import jax
import jax.numpy as jnp
import jax.random as jrand
from einops import rearrange
from functools import partial

from typing import List
from model_types import *
from forward_common import *



def pixtral_attention_prefill(block_params: TransformerBlock, hidden_state_BTC, freqs,
                              query_heads, kv_heads, head_dim, attn_mask):
  # compute qkv
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
  Q = apply_rope(Q, freqs)
  K = apply_rope(K, freqs) # both have the same head dim

  K = K[:, :, None, :, :] # broadcast over r
  V = V[:, :, None, :, :] # broadcast over r

  # mistral uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  scale = jnp.bfloat16(1.0 / jnp.sqrt(Q.shape[-1]))
  Q = Q*scale
  attn = Q @ jnp.swapaxes(K, -1, -2)
  attn = jnp.where(attn_mask, jnp.bfloat16(-jnp.inf), attn) 
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(jnp.bfloat16) # do attn softmax in float32
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn @ block_params.attention_wo_weight.T

  return out.astype(jnp.bfloat16), K, V



# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block_prefill(
    block_params: TransformerBlock,
    hidden_state_BTC: jax.Array,
    freqs_1d, query_heads, kv_heads, head_dim,
    attn_mask) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC, K, V = pixtral_attention_prefill(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim, attn_mask)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC, K, V



def text_forward_prefill(model_params: PixtralModel, message_tokens):
  hidden_state_BTC = text_embedding(model_params, message_tokens)[jnp.newaxis, :]
  return forward_prefill(model_params, hidden_state_BTC)



def mm_forward_prefill(model_params: PixtralModel, message_tokens, processed_images, intext_image_start_indices):
  hidden_state_BTC = multimodal_embedding(model_params, message_tokens, processed_images, intext_image_start_indices)[jnp.newaxis, :] # fake batch for now
  return forward_prefill(model_params, hidden_state_BTC)



def forward_prefill(model_params, hidden_state_BTC):
  B, T, C = hidden_state_BTC.shape
  head_dim = 128 # params.json
  max_pos, d = T, head_dim
  freqs = precompute_rope_freqs_1d(max_pos, d) # mistral does rope after splitting k and q into gqa heads. q and k are split into the same channel size per head

  # attention layers
  Hq = 32 # params.json
  Hk = 8 # params.json
  attn_mask = get_causal_mask(T)
  # head dim defined above - it's used to calculate rope1d frequencies
  # scan compiles faster than a for loop
  def scanf(hidden_state, block_params):
    hidden_state, K, V = transformer_block_prefill(block_params, hidden_state, freqs, Hq, Hk, head_dim, attn_mask)
    return hidden_state, (K, V)
  hidden_state_BTC, kv_pairs = jax.lax.scan(scanf, hidden_state_BTC, model_params.transformer.transformer_layers)

  # remap KVCache to appropriate structure
  post_scan_remap = lambda xs_tuple: jnp.stack(xs_tuple) # (K0, K1, K2..Kn) => jax.Array of shape (n,*K.shape)
  Ks, Vs = kv_pairs # [(K, V), ... (K, V)] => [(K, K, ...), (V, V, ...)]
  kvcache = KVCache(
      K=post_scan_remap(Ks), # (xfmr block, B, Hk, r=1, T, d)
      V=post_scan_remap(Vs),
  )

  # inference - we only care about the final token past this point
  hidden_state_BC = hidden_state_BTC[:, -1, :]
  # layernorm
  hidden_state_BC = layernorm(hidden_state_BC, model_params.norm_weight, jnp.zeros((1, hidden_state_BTC.shape[-1])))
  # lm_head: channel -> vocab logits
  hidden_state_BC = hidden_state_BC @ model_params.output_weight.T # (B, C) @ (C, vocab) => (B, vocab)
  return hidden_state_BC, kvcache



def inference_prefill(key, pixtral_params, tokens, images, intext_image_start_indices) -> str:
  if images:
      next_token_logit, kvcache = mm_forward_prefill(pixtral_params, tokens, images, intext_image_start_indices)
  else:
      next_token_logit, kvcache = text_forward_prefill(pixtral_params, tokens)
  next_token = jrand.categorical(key, next_token_logit, axis=-1)
  return next_token, kvcache
