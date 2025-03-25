import jax
import jax.numpy as jnp
import jax.random as jrand
import functools
from einops import rearrange
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


def layer_norm(x, weight):
  # ASSUMPTION: layer norm is axis=-1
  x = weight * (x - jnp.mean(x, axis=-1, keepdims=True, dtype="bfloat16")) / (jnp.std(x, axis=-1, keepdims=True, dtype="bfloat16") + 1e-5)
  return x


def RMSnorm(x, weight):
  rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True, dtype="bfloat16"))
  return weight*(x / (rms + 1e-5))


def feed_forward(
        x: jax.Array,
        layer_params: LangModelSelfAttentionLayer
        ) -> jax.Array:
    gate = x @ jnp.transpose(layer_params.mlp_gate_proj_weight)
    up = x @ jnp.transpose(layer_params.mlp_up_proj_weight)
    out = up * jax.nn.silu(gate)
    out = out @ jnp.transpose(layer_params.mlp_down_proj_weight)
    return out


# TODO break up large functions into smaller testable functions
# create attention mask
# etc


# TODO make the mask different per-batch.
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
  xBTC_attn_residual = Z @ jnp.transpose(layer_params.self_attn_o_proj_weight) # (B, T, Z) @ (Z, C) => B, T, C # ASSUMPTION does not need transposition
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


def embed_tokens(lang_model_params: LangModel, batch_tokens: TensorBT) -> TensorBTC:
  #vocab_size = lang_model_params.lm_head_weight.shape[0]
  #one_hot_tokens = jax.nn.one_hot(batch_tokens, vocab_size, axis=-1)
  #token_embeddings = one_hot_tokens @ lang_model_params.lm_head_weight
  token_embeddings = jnp.take(lang_model_params.model.embed_tokens, batch_tokens, axis=0)
  token_embeddings = lang_model_params.model.embed_tokens[batch_tokens]
  return token_embeddings


def rope(channels: TensorBTC) -> TensorBTC:
  """
  channels: tensor of shape (B, T, Q)
  rope(channels) applies rope_channel across each channel at each position T
  example: rotated_channels = rope(Q)
  """
  B, T, Q = channels.shape
  rope_each_channel = jax.vmap(rope_channel, in_axes=(0, 0))
  rope_each_batch = jax.vmap(rope_each_channel, in_axes=(0, None))
  positions = jnp.arange(T)
  rotated_channels = rope_each_batch(channels, positions)
  return rotated_channels


def rope_channel(channel: jax.Array, position: int) -> jax.Array:
  """
  channel: the C in B, T, C.
  frequencies = [th1, th1, th2, th2, ...]
  output: channel, but rotated.
  """
  d = channel.shape[-1]
  i = jnp.arange(d, dtype="bfloat16")
  N = 50_000 # value from llama 3.1 #jnp.repeat(i, 2)
  frequencies = jnp.float_power(N, -2*i/d).astype("bfloat16") # can and should be precomputed. probably will be stored when compiled
  cos = jnp.cos(position*frequencies).astype("bfloat16")
  sin = jnp.sin(position*frequencies).astype("bfloat16")

  even_indices = (jnp.arange(d) % 2 == 0).astype(int)
  signs = jnp.where(even_indices.astype(bool), -jnp.ones(d, dtype="bfloat16"), jnp.ones(d, dtype="bfloat16")).astype("bfloat16")
  indices = jnp.arange(d, dtype=int) + 2*(even_indices)
  channel_rotated = cos*channel + sin*channel[indices]*signs
  
  return channel_rotated



# https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf
def vision_encoding():
    pass


def cross_attention_layer():
    pass


# trainable if needed
# TODO: implement text llama 3.1 first and test it
# model -> TensorBTC -> TensorBTC -> float -> TensorBTC
@functools.partial(jax.jit, static_argnames=["temp"])
def mm_forward(model_params: LlamaParams, xBTC: TensorBTC, xBTC_image: TensorBTC, temp: float) -> LogProbsBT:
  ### PADDING MASK
  non_padding_tokens = (context_tokens != 128004)
  padding_mask = ~non_padding_tokens
  
  ### TEXT EMBEDDINGS
  xBTC_text = embed_tokens(model_params.language_model, context_tokens)
  xBTC = xBTC_text

  ### VISION ENCODING
  ## local TODO

  ## global TODO

  ### TRANSFORMER LAYERS
  def scan_self_attn_layers(xBTC, layer_params):
    xBTC_attn_residual = self_attention_layer(layer_params, xBTC, padding_mask)
    xBTC = xBTC + xBTC_attn_residual 
    # layer norm
    xBTC_postattn_residual = RMSnorm(xBTC, layer_params.post_attention_layernorm_weight)
    # swiglu
    xBTC_postattn_residual = feed_forward(xBTC_postattn_residual, layer_params)
    xBTC = xBTC + xBTC_postattn_residual
    return xBTC, None

  def scan_cross_attn_layers(todo, TODO):
      pass

  cross_attn_layers = [3, 8, 13, 18, 23, 28, 33, 38] # ASSUMPTION double check these layers
  
  xBTC, _ = jax.lax.scan(scan_self_attn_layers, xBTC, model_params.language_model.model.self_attention_layers[0:5])
  xBTC, _ = jax.lax.scan(scan_cross_attn_layers ... .self_attention_layers[5])
  xBTC, _ = ... # just do this all manually. manually pick the layers
  
  
  ## OUTPUT
  #layernorm and project to logits
  xBTC = RMSnorm(xBTC, model_params.language_model.model.norm_weight)
  # ffw
  yBTC_logits = xBTC @ jnp.transpose(model_params.language_model.lm_head_weight)
  # logits => logprobs
  yBTC_logprobs = jax.nn.log_softmax(yBTC_logits/(temp + 1e-5), axis=-1)
  return yBTC_logprobs, yBTC_logits




# TODO make the mask different per-batch.
@jax.jit
def self_attention_layer_old(layer_params: LangModelSelfAttentionLayer, xBTC: TensorBTC, padding_mask: jax.Array) -> TensorBTC:
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
  attention_scores = jax.nn.softmax(query_padding_mask + causal_mask + attention_table, axis=-1) # ASSUMPTION: does not need axis=-1, -2
  attention_scores = key_padding_mask[jnp.newaxis, ..., jnp.newaxis] * attention_scores # the final n rows should be all 0 (padding tokens)
  Z = jnp.einsum("BGhTt,Bhtd->BGhtd", attention_scores, V)   # (B,H,T,T) @ (B,H,T,d_v//H_KV) => (B, H, T, d_v//H_KV)
  # bug: "BHTT,BhTd->BHTd" breaks this. future: don't use the same letter twice. be specific about the dimensions in the operation

  # reshape back into xBTC_residual
  Z = rearrange(Z, "B G h T D -> B (G h) T D") # group the 4x8 heads into 32 heads
  Z = rearrange(Z, "B H T D -> B T H D") # Swap head and time dim
  Z = jnp.reshape(Z, (B, T, Z.shape[-1]*H_Q))  # (B, H_Q, T, d_KV) => (B, T, Z)

  # project BHTd_v back to BHTd
  xBTC_attn_residual = Z @ jnp.transpose(layer_params.self_attn_o_proj_weight) # (B, T, Z) @ (Z, C) => B, T, C # ASSUMPTION does not need transposition
  return xBTC_attn_residual
