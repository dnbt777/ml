import jax
import jax.numpy as jnp
import jax.random as jrand
import functools
import PIL

from load_params import LlamaParams
from typing import TypeAlias
from llama_types import (
  LlamaParams, LangModel,
  LangModelSelfAttentionLayer, LangModelCrossAttentionLayer, 
  Text, Tokens
)

# Type aliases for better static typing
TensorBTC: TypeAlias = jax.Array
TensorBC: TypeAlias = jax.Array
TensorBT: TypeAlias = jax.Array
TensorBTHd: TypeAlias = jax.Array
LogProbsBT: TypeAlias = jax.Array
ImageFloat32: TypeAlias = jax.Array # PIL uses float32 for F mode.



# def vision_encoder
# ImageFloat32 -> TensorBTC


# def text_encoder
# text -> embeddings


def layer_norm(x, weight):
  # ASSUMPTION: layer norm is axis=-1
  # ASSUMPTION: we dont need to add eta to the std in this divide
  x = weight * (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.std(x, axis=-1, keepdims=True)
  return x


# TODO make the mask different per-batch.
@jax.jit
def self_attention_layer(layer_params: LangModelSelfAttentionLayer, xBTC: TensorBTC, mask: jax.Array) -> TensorBTC:
  # input layernorm
  xBTC = layer_norm(xBTC, layer_params.input_layernorm_weight)

  # Reshape input into attention heads
  # xBTC => xBHTd
  B, T, C  = xBTC.shape
  H_Q = 32 # TODO replace the hardcoding
  H_KV = 8 # GQA kv-heads

  Q = xBTC @ jnp.transpose(layer_params.self_attn_q_proj_weight)  # (BTC) @ (d, d_k) => (B, T, d_k)
  K = xBTC @ jnp.transpose(layer_params.self_attn_k_proj_weight) # (BTC) @ (d, d_k) => (B, T, d_k)
  V = xBTC @ jnp.transpose(layer_params.self_attn_v_proj_weight) # (BTC) @ (d, d_v) => (B, T, d_v) 
  # attention heads and KV-groups are introduced at the QKV stage
  Q = jnp.reshape(Q, (B, H_Q, T, Q.shape[-1]//H_Q)) # (B, T, d_k) => (B, H, T, d_k//H_Q)
  K = jnp.reshape(K, (B, H_KV, T, K.shape[-1]//H_KV)) # (B, T, d_k) => (B, G, T, d_k//H_KV)
  V = jnp.reshape(V, (B, H_KV, T, V.shape[-1]//H_KV)) # (B, T, d_v) => (B, G, T, d_v//H_KV)
  Kt = jnp.swapaxes(K, 2, 3)  # (B, H, T, d_k) => (B, H, d_k//H, T)
  d_k = Q.shape[-1]
  # Compute attention scores. H = H_Q, h = H_KV. h is smaller so is broadcast
  attention_table = jnp.einsum("BHTd,Bhdt->BHTt", Q, Kt) / jnp.sqrt(d_k) # (B,H_Q,T,d_k//H_Q) @ (B,H_KV,T,d_K//H_KV) => (B, H_Q, T, T)
  attention_scores = jax.nn.softmax(mask + attention_table, axis=(-2, -1)) # ASSUMPTION: does not need axis=-1, -2
  Z = jnp.einsum("BHtT,BhTd->BHtd", attention_scores, V)   # (B,H,T,T) @ (B,H,T,d_v//H_KV) => (B, H, T, d_v//H_KV)
  # bug: "BHTT,BhTd->BHTd" breaks this. future: don't use the same letter twice. be specific about the dimensions in the operation

  # reshape back into xBTC_residual
  Z = jnp.reshape(Z, (B, T, Z.shape[-1]*H_Q))  # (B, H, T, d_v//H_KV) => (B, T, d_v)

  # project BHTd_v back to BHTd
  xBTC_residual = Z @ layer_params.self_attn_o_proj_weight # (B, T, d_v) @ (d_v, d) => B, T, d   i.e. (B,T,C)
  
  # layer norm
  xBTC_residual = layer_norm(xBTC_residual, layer_params.post_attention_layernorm_weight)

  # feed forward w SwiGLU
  # https://arxiv.org/pdf/2002.05202
  # ASSUMPTION: for ffn, its down(silu(gate(up(x)))), not silu(down(gate(up(x))))
  x1 = xBTC_residual @ jnp.transpose(layer_params.mlp_up_proj_weight)
  x2 = jax.nn.swish(xBTC_residual @ jnp.transpose(layer_params.mlp_gate_proj_weight))
  xBTC_residual = (x1*x2) @ jnp.transpose(layer_params.mlp_down_proj_weight)
  
  # skip connection
  xBTC = xBTC + xBTC_residual
  return xBTC


def embed_tokens(lang_model_params: LangModel, batch_tokens: TensorBT) -> TensorBTC:
  vocab_size = lang_model_params.lm_head_weight.shape[0]
  one_hot_tokens = jax.nn.one_hot(batch_tokens, vocab_size, axis=-1)
  token_embeddings = one_hot_tokens @ lang_model_params.lm_head_weight
  return token_embeddings


def rope():
  pass

# trainable if needed
# TODO: implement text llama 3.1 first and test it
# model -> TensorBTC -> TensorBTC -> float -> TensorBTC
@jax.jit
def text_forward(model_params: LlamaParams, context_tokens: Tokens, temp: float) -> LogProbsBT:
  ### PADDING MASK
  non_padding_tokens = (context_tokens != 128004)
  padding_mask = jnp.where(~jnp.outer(non_padding_tokens, non_padding_tokens), -jnp.inf, 0)
  
  ### TEXT EMBEDDINGS
  xBTC_text = embed_tokens(model_params.language_model, context_tokens)
  xBTC = xBTC_text

  ### TRANSFORMER
  # iterate over attention layers
  # replacing for loops w scans reduces compile time and memory from the jitted compute graph
  # set up the scan
  def scan_fn(xBTC, layer_params):
    xBTC = self_attention_layer(layer_params, xBTC, padding_mask)
    return xBTC, None
  # do the scan
  xBTC, _ = jax.lax.scan(scan_fn, xBTC, model_params.language_model.model.self_attention_layers)
  
  # layernorm and project to logits
  xBTC = layer_norm(xBTC, model_params.language_model.model.norm_weight)
  yBTC = xBTC @ jnp.transpose(model_params.language_model.lm_head_weight)

  # logits => logprobs
  # conditional branching is fine in jax.jit if the conditional is static
  # ASSUMPTION: 1e-7 is not too large of an eta in the output of logprobs
  yBTC_logprobs = jax.nn.log_softmax(yBTC/(temp + 1e-7), axis=-1)
  
  return yBTC_logprobs 




# trainable if needed
# TODO: implement text llama 3.1 first and test it
# model -> TensorBTC -> TensorBTC -> float -> TensorBTC
@functools.partial(jax.jit, static_argnames=["temp"])
def forward(model_params: LlamaParams, xBTC: TensorBTC, xBTC_image: TensorBTC, temp: float) -> LogProbsBT:
  ### VISION ENCODING
  ## local


  ## global


  ### -> xBTC_image
  
  ### TEXT EMBEDDINGS
  ## embed tokens
  ### -> xBTC_text

  # xBTC = xBTC_text

  ### TRANSFORMER
  # for each attention layer:
    # if self attention:
      # xBTC = self_attention_layer(attention_layer_params, xBTC)

    # if cross attention:
      # reshape input xBTC and input xBTC_image into attention heads
      # xBTC => xBTHd, xBTC_image => xBTHd_image
      # do cross attention: Q = xBTHd, K = xBTHd_image

      # reshape output into xBTC_residual

      # xBTC_residual = layer_norm(xBTC_residual)
      # forward mlp layer. xBTC_residual = ffn(xBTC_residual)

      # xBTC = xBTC + xBTC_residual
  
  
  # layernorm and project to logits
  xBTC = layer_norm(xBTC, model_params.language_model.model.norm_weight)
  yBTC = xBTC @ model_params.language_model.lm_head_weight

  # logits => logprobs
  # conditional branching is fine in jax.jit if the conditional is static
  if temp == 0:
    yBTC_logprobs = jax.nn.log_softmax(xBTC, axis=-1)
  else:
    # ASSUMPTION: 1e-7 is not too large of an eta in the output of logprobs
    yBTC_logprobs = jax.nn.log_softmax(xBTC/(temp + 1e-7), axis=-1)
  
  return yBTC_logprobs 


# model -> str -> ImageFloat32 -> float -> str
def inference(model_params: LlamaParams, prompt: str, image: ImageFloat32, temperature: float) -> str:
  # yBTC_probs = forward(model_params, encode(prompt), temperature) 

  # yBT = jnp.argmax(yBTC_probs, axis=-1)
  
  # return yBT
  pass