import jax
import jax.numpy as jnp
from einops import rearrange
from forward_utils import RMSnorm, feed_forward, rope, embed_tokens
from setup_utils import LlamaParams
from llama_types import (
  LlamaParams, LangModelSelfAttentionLayer, 
  Tokens, TensorBTC, LogProbsBT, 
)


# OPTIMIZATION: do matmul first, then break into heads just before softmax
def GQA_attention(layer_params, Q, K, V, padding_mask, B, T, H_Q, H_KV):
  # GQA
  d_Q = Q.shape[-1]//H_Q
  d_KV = K.shape[-1]//H_KV
  Q = jnp.reshape(Q, (B, T, H_Q, d_Q)) # (B, T, d_k) => (B, T, H_Q, d_Q)
  Q = jnp.einsum("BTHD->BHTD", Q)
  K = jnp.reshape(K, (B, T, H_KV, d_KV)) # (B, T, d_k) => (B, T, H_KV, d_KV)
  K = jnp.einsum("BThd->BhTd", K)
  V = jnp.reshape(V, (B, T, H_KV, d_KV)) # (B, T, d_v) => (B, T, H_KV, d_KV)
  V = jnp.einsum("BThd->BhTd", V) # (B, T, H, D) => (B, H, T, D)
  
  G = H_Q // H_KV # groups
  Q = rearrange(Q, "B (h G) T D -> B G h T D", G=G)
  # Compute attention scores
  attention_table = jnp.einsum("BGhTD,Bhtd->BGhTt", Q, K) / jnp.sqrt(d_KV) # (B,H_Q,T,d_Q) @ (B,H_Q,d_KV,T) => (B, H_Q, T, T)
  causal_mask_shape = attention_table.shape[-2:]
  causal_mask = jnp.triu(jnp.ones(causal_mask_shape), k=1).astype(bool)
  causal_mask = jnp.where(causal_mask, -jnp.inf, 0)
  query_padding_mask = jnp.where(padding_mask, -jnp.inf, 0) # add pre-softmax
  key_padding_mask = jnp.where(padding_mask, 0, 1) # multiply post-softmax along the t dim in BHTt
  attention_scores = jax.nn.softmax((query_padding_mask + causal_mask + attention_table).astype(jnp.float32), axis=-1).astype("bfloat16") # ASSUMPTION: does not need axis=-1, -2
  attention_scores = key_padding_mask[jnp.newaxis, ..., jnp.newaxis] * attention_scores # the final n rows should be all 0 (padding tokens)
  Z = jnp.einsum("BGhTt,Bhtd->BGhtd", attention_scores, V)   # (B,H,T,T) @ (B,H,T,d_v//H_KV) => (B, H, T, d_v//H_KV)
  # bug: "BHTT,BhTd->BHTd" breaks this. future: don't use the same letter twice. be specific about the dimensions in the operation

  # reshape back into xBTC_residual
  Z = rearrange(Z, "B G h T D -> B (G h) T D") # group the 4x8 heads into 32 heads
  Z = rearrange(Z, "B H T D -> B T H D") # Swap head and time dim
  Z = jnp.reshape(Z, (B, T, Z.shape[-1]*H_Q))  # (B, H_Q, T, d_KV) => (B, T, Z)

  # project BHTd_v back to BHTd
  xBTC_attn_output = Z @ jnp.transpose(layer_params.self_attn_o_proj_weight) # (B, T, Z) @ (Z, C) => B, T, C # ASSUMPTION does not need transposition

  return xBTC_attn_output



# TODO make the mask different per-batch.
# TODO simplify GQA by doing the matmul first and not overcomplicating it
@jax.jit
def self_attention_layer(layer_params: LangModelSelfAttentionLayer, xBTC: TensorBTC, padding_mask: jax.Array) -> TensorBTC:
  # input layernorm
  xBTC = RMSnorm(xBTC, layer_params.input_layernorm_weight)

  # Reshape input into attention heads
  # xBTC => xBHTd
  B, T, C  = xBTC.shape
  H_Q = 32 # TODO replace the hardcoding
  H_KV = 8 # GQA kv-heads

  Q = xBTC @ jnp.transpose(layer_params.self_attn_q_proj_weight)  # (BTC) @ (d, d_k) => (B, T, d_q)
  K = xBTC @ jnp.transpose(layer_params.self_attn_k_proj_weight) # (BTC) @ (d, d_k) => (B, T, d_k)
  V = xBTC @ jnp.transpose(layer_params.self_attn_v_proj_weight) # (BTC) @ (d, d_v) => (B, T, d_v) 
  # RoPE 
  Q = rope(Q)
  K = rope(K)
  # attention heads and KV-groups are introduced at the QKV stage
  
  # GQA
  xBTC_attn_residual = GQA_attention(layer_params, Q, K, V, padding_mask, B, T, H_Q, H_KV)

  return xBTC_attn_residual




# trainable if needed
# TODO: implement text llama 3.1 first and test it
# model -> TensorBTC -> TensorBTC -> float -> TensorBTC
@jax.jit
def text_forward(model_params: LlamaParams, context_tokens: Tokens, temp: float) -> LogProbsBT:
  ### PADDING MASK
  non_padding_tokens = (context_tokens != 128004)
  padding_mask = ~non_padding_tokens
  
  ### TEXT EMBEDDINGS
  xBTC_text = embed_tokens(model_params.language_model, context_tokens)
  xBTC = xBTC_text

  ### TRANSFORMER
  # iterate over attention layers
  # replacing for loops w scans reduces compile time and memory from the jitted compute graph
  # set up the scan
  def scan_fn(xBTC, layer_params):
    xBTC_attn_residual = self_attention_layer(layer_params, xBTC, padding_mask)
    xBTC = xBTC + xBTC_attn_residual 
    # layer norm
    xBTC_postattn_residual = RMSnorm(xBTC, layer_params.post_attention_layernorm_weight)
    # swiglu
    xBTC_postattn_residual = feed_forward(xBTC_postattn_residual, layer_params)
    xBTC = xBTC + xBTC_postattn_residual
    return xBTC, None
  # do the scan
  xBTC, _ = jax.lax.scan(scan_fn, xBTC, model_params.language_model.model.self_attention_layers)
  
  # layernorm and project to logits
  xBTC = RMSnorm(xBTC, model_params.language_model.model.norm_weight)
  # ffw
  yBTC_logits = xBTC @ jnp.transpose(model_params.language_model.lm_head_weight)

  # logits => logprobs
  yBTC_logprobs = jax.nn.log_softmax(yBTC_logits/(temp + 1e-5), axis=-1)
  
  return yBTC_logprobs, yBTC_logits



