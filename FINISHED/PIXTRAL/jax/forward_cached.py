import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from model_types import *
from einops import rearrange
import jax.random as jrand
from functools import partial

# mm_preprocessor :: [Image] -> [image_tiles or whatever]
from PIL import Image
import cv2

from forward_common import *

# here, we compute the rope freqs for a single token at position i
# this is used after prefill
@partial(jax.jit, static_argnames=["d"])
def precompute_specific_position_rope_freqs_1d(i, d):
    rope_theta = 1_000_000_000 # params.json
    max_j = d//2
    j = jnp.arange(max_j)
    freqs_ij = i*rope_theta**(-2.0*j/d) # (d//2,)
    cos, sin = jnp.cos(freqs_ij)[None, :], jnp.sin(freqs_ij)[None, :]
    return jax.lax.complex(cos, sin).astype(jnp.complex64)


@partial(jax.jit, static_argnames=["query_heads","kv_heads","head_dim"])
def pixtral_attention_cached(block_params: TransformerBlock, hidden_state_BTC, freqs, query_heads, kv_heads, head_dim,
                             K_cache, V_cache, next_token_mask, padding_mask):
  ## COMPUTE NEXT Q, K, and V FOR THE SINGLE TOKEN INPUT
  # compute qkv
  hidden_state_BTC = hidden_state_BTC
  Q = hidden_state_BTC @ block_params.attention_wq_weight.T
  K = hidden_state_BTC @ block_params.attention_wk_weight.T
  V = hidden_state_BTC @ block_params.attention_wv_weight.T

  # split into heads (GQA)
  Hk=kv_heads
  Hq=query_heads
  repeats = Hq // Hk 
  Q = rearrange(Q, "B T (Hk r d) -> B Hk r T d", Hk=Hk, r=repeats, d=head_dim)
  K = rearrange(K, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)
  V = rearrange(V, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)

  # rope1D AFTER splitting into GQA heads
  # TODO MAKE THE FREQ PROPER POSITION
    # this will require a new rope that just computes freq(pos, d_static)
  Q = apply_rope(Q, freqs)
  K = apply_rope(K, freqs) # both have the same head dim

  K = K[:, :, None, :, :] # broadcast over r
  V = V[:, :, None, :, :] # broadcast over r

  # repeat kv to match query (optimization: view or broadcast instead of repeat)

  ## LOAD KV CACHE
  next_token_mask = next_token_mask[None, None, None, :, None] # (T,) => (B, Hk, r, T, d)
  K = jnp.where(next_token_mask, K, K_cache) #           cache      t    padding
  V = jnp.where(next_token_mask, V, V_cache) # mask= [00000000000000100000000000000]

  # uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  #scale = jax.lax.rsqrt(Q.shape[-1]) # rqrt does not accept int64
  scale = 1.0 / jnp.sqrt(Q.shape[-1])
  Q = jnp.float32(Q * scale)
  attn = Q @ jnp.swapaxes(K, -1, -2)
  attn = attn + jnp.where(padding_mask, -jnp.inf, 0) # mask out padding tokens that keep the shape constant. Q is just 1, but K and V (after kv cache) is max_tokens
  attn = jax.nn.softmax(attn, axis=-1) # kernel does this
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn @ block_params.attention_wo_weight.T

  return out.astype(jnp.bfloat16), K, V


# TODO
# make cached
# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
@partial(jax.jit, static_argnames=["query_heads","kv_heads","head_dim"])
def transformer_block_cached(block_params: TransformerBlock, hidden_state_BTC: jax.Array, freqs_1d, query_heads, kv_heads, head_dim,
                             K_cache, V_cache, next_token_mask, padding_mask):
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC, K_cache, V_cache = pixtral_attention_cached(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim,
                                                            K_cache, V_cache, next_token_mask, padding_mask)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC, K_cache, V_cache


@jax.jit
def next_token_embedding(model_params: PixtralModel, next_token_batch) -> jax.Array:
  embeddings = text_embedding(model_params, next_token_batch)
  return embeddings.astype(jnp.bfloat16)


@jax.jit
def mm_forward_cached(model_params: PixtralModel, next_token_batch, kvcache, next_token_index):
  # get embeddings
  hidden_state_BTC = next_token_embedding(model_params, next_token_batch)[None, :] # fake batch for now

  head_dim = 128 # params.json
  freqs = precompute_specific_position_rope_freqs_1d(next_token_index, head_dim)

  # attention
  Hq = 32 # params.json
  Hk = 8 # params.json
  # create masks
  _, B, _, _, T, d = kvcache.K.shape
  next_token_mask = (jnp.arange(T) == next_token_index) # demarcate location of next token
  padding_mask = (jnp.arange(T) >= next_token_index) # mask out future tokens in attn mechanism
  # loop through attention layers
  def scanf(hidden_state, xfmr_block_data):
    block_params, block_kvcache = xfmr_block_data
    hidden_state, K, V = transformer_block_cached(block_params, hidden_state, freqs, Hq, Hk, head_dim,
                                                  block_kvcache.K, block_kvcache.V, next_token_mask, padding_mask)
    return hidden_state, (K, V)
  xfmr_blocks_data = (
      model_params.transformer.transformer_layers, # (40,...)
      kvcache # (40, ...)
  )
  hidden_state_BTC, kv_pairs = jax.lax.scan(scanf, hidden_state_BTC, xfmr_blocks_data)
  hidden_state_BTC = layernorm(hidden_state_BTC, model_params.norm_weight, jnp.zeros_like(hidden_state_BTC))
  hidden_state_BTC = hidden_state_BTC @ model_params.output_weight.T # lm head

  post_scan_remap = lambda xs_tuple: jnp.stack(xs_tuple) # (K0, K1, K2..Kn) => jax.Array of shape (n,*K.shape)
  Ks, Vs = kv_pairs # [(K, V), ... (K, V)] => [(K, K, ...), (V, V, ...)]
  kvcache = KVCache(
      K=post_scan_remap(Ks), # (xfmr block, B, Hk, r=1, T, d)
      V=post_scan_remap(Vs),
  )
  return hidden_state_BTC, kvcache


@jax.jit
def inference_cached(key, pixtral_params, next_token_batch, kvcache, next_token_index) -> str:
  # get logits
  next_token_batch_logits, kvcache = mm_forward_cached(pixtral_params, next_token_batch,
                                                       kvcache, next_token_index) # B, 1, C
  # random sample
  next_token_batch = jrand.categorical(key, next_token_batch_logits, axis=-1) # (B,1)
  return next_token_batch, kvcache
