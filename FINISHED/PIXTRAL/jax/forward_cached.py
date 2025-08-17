import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from model_types import *
from einops import rearrange
import jax.random as jrand
from functools import partial

from PIL import Image
import cv2

from forward_common import *




# use jit to cache the majority of the freqs computation
@partial(jax.jit, static_argnames=["d"])
def precompute_rope_channel_vector(d):
    rope_theta = 1_000_000_000 # params.json
    max_j = d//2
    j = jnp.arange(max_j, dtype=jnp.float32)
    channel_vec = rope_theta**(-2.0*j/jnp.float32(d))
    return channel_vec
    
# compute the rope freqs for a single token at position i
# only used in cached forward
@partial(jax.jit, static_argnames=["d"])
def precompute_specific_position_rope_freqs_1d(i, d):
    channel_vec = precompute_rope_channel_vector(d) # jit will cache channel vec
    freqs_ij = i[:, None] * channel_vec[None, :] # (B, d//2,)  
    cos, sin = jnp.cos(freqs_ij), jnp.sin(freqs_ij)
    return cos[:, None, None, None, :], sin[:, None, None, None, :] # B, Hk, r, T, d (for Q and K in GQA)



@jax.jit
def apply_rope_batches(hidden_state, cos, sin):
    original_shape = hidden_state.shape # ..., T, C
    T, C = hidden_state.shape[-2:]
    d = C
    hidden_state_pairs = jnp.reshape(hidden_state, (*hidden_state.shape[:-2], T, d//2, 2))
    re, im = hidden_state_pairs[..., 0], hidden_state_pairs[..., 1] # B, i, d//2
    # rotate each pair by its corresponding theta:
    # rotating a vector <x,y> by theta == multiplying (x + iy) by e^(i*theta)
        # where i = sqrt(-1), and x = re(al), and y = im(aginary) components
    # derivation:
    # complex rotation factor = e^(i*theta) = cos(theta) + i*sin(theta)
    # so (x + i*y) * e^(i*theta) =
    # = (x + i*y) * (cos(theta) + i*sin(theta))
    # = x*cos(theta) + i*y*cos(theta) + x*i*sin(theta) + i*y*i*sin(theta)
    # = x*cos(theta) - y*sin(theta) + i*(y*cos(theta) + x*sin(theta))
    # = <x*cos(theta) - y*sin(theta), y*cos(theta) + x*sin(theta)>
    #cos, sin = jnp.real(freqs)[:, None, None, None, :], jnp.imag(freqs)[:, None, None, None, :] # B, 1, d//2
    re_rot = re*cos - im*sin
    im_rot = im*cos + re*sin
    hidden_state_pairs_rot = jnp.stack([re_rot, im_rot], axis=-1) # (B, T, d//2, 2)
     # [re, im, re, im, re, im, ... ]
    return jnp.reshape(hidden_state_pairs_rot, hidden_state.shape).astype(jnp.bfloat16)




@partial(jax.jit, static_argnames=["query_heads","kv_heads","head_dim"], donate_argnames=["K_cache", "V_cache"])
def pixtral_attention_cached(block_params: TransformerBlock, hidden_state_BTC, rope_cos, rope_sin, query_heads, kv_heads, head_dim, attn_scale,
                             K_cache, V_cache, batch_next_token_indices, padding_mask, block_lora_params=None):
  ## COMPUTE NEXT Q, K, and V FOR THE SINGLE TOKEN INPUT
  # compute qkv
  hidden_state_BTC = hidden_state_BTC
  Q = hidden_state_BTC @ block_params.attention_wq_weight.T
  K = hidden_state_BTC @ block_params.attention_wk_weight.T
  V = hidden_state_BTC @ block_params.attention_wv_weight.T
  if block_lora_params:
      Q = Q + block_lora_params.alpha_q*((hidden_state_BTC @ block_lora_params.in_q) @ block_lora_params.out_q)
      K = K + block_lora_params.alpha_k*((hidden_state_BTC @ block_lora_params.in_k) @ block_lora_params.out_k)
      V = V + block_lora_params.alpha_v*((hidden_state_BTC @ block_lora_params.in_v) @ block_lora_params.out_v)

  # split into heads (GQA)
  Hk=kv_heads
  Hq=query_heads
  repeats = Hq // Hk 
  Q = rearrange(Q, "B T (Hk r d) -> B Hk r T d", Hk=Hk, r=repeats, d=head_dim)
  K = rearrange(K, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)
  V = rearrange(V, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)

  K = K[:, :, None, :, :] # broadcast over r
  V = V[:, :, None, :, :] # broadcast over r

  # rope1D AFTER splitting into GQA heads
  Q = apply_rope_batches(Q, rope_cos, rope_sin)
  K = apply_rope_batches(K, rope_cos, rope_sin)

  ## UPDATE KV CACHE
  def update_cache(G_cache, G, t_index):
      pos = (0, 0, t_index, 0) # Hk, r, T, d (vmapped over batch)
      return jax.lax.dynamic_update_slice(G_cache, G, pos)
  batch_update_cache = jax.vmap(update_cache, in_axes=(0, 0, 0))

  K_cache = batch_update_cache(K_cache, K, batch_next_token_indices)
  V_cache = batch_update_cache(V_cache, V, batch_next_token_indices)
  # K = jnp.where(next_token_mask, K, K_cache) #           cache      t    padding
  # V = jnp.where(next_token_mask, V, V_cache) # mask= [00000000000000100000000000000]

  padding_mask = padding_mask[:, None, None, None, :]

  # uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  #scale = jax.lax.rsqrt(Q.shape[-1]) # rqrt does not accept int64
  Q = Q*attn_scale
  #attn = Q @ jnp.swapaxes(K, -1, -2)
  attn = Q @ jnp.swapaxes(K_cache, -1, -2)
  attn = jnp.where(padding_mask, -jnp.bfloat16(jnp.inf), attn) # mask out padding tokens that keep the shape constant. Q is just 1, but K and V (after kv cache) is max_tokens
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(jnp.bfloat16) # kernel does this
  #attn = attn @ V
  attn = attn @ V_cache

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn @ block_params.attention_wo_weight.T
  if block_lora_params:
      out = out + block_lora_params.alpha_o*((attn @ block_lora_params.in_o) @ block_lora_params.out_o)

  return out.astype(jnp.bfloat16), K_cache, V_cache



# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
@partial(jax.jit, static_argnames=["query_heads","kv_heads","head_dim"], donate_argnames=["K_cache", "V_cache"])
def transformer_block_cached(block_params: TransformerBlock, hidden_state_BTC: jax.Array, rope_cos, rope_sin, query_heads, kv_heads, head_dim, attn_scale,
                             K_cache, V_cache, batch_next_token_indices, padding_mask, block_lora_params=None):
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC, K_cache, V_cache = pixtral_attention_cached(block_params, residual_BTC, rope_cos, rope_sin, query_heads, kv_heads, head_dim, attn_scale,
                                                            K_cache, V_cache, batch_next_token_indices, padding_mask, block_lora_params=block_lora_params)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC, K_cache, V_cache



@jax.jit
def next_token_embedding(model_params: PixtralModel, next_token_batch) -> jax.Array:
  embeddings = text_embedding(model_params, next_token_batch)
  return embeddings.astype(jnp.bfloat16)[:, None, :] # return B, 1, C instead of B, C



@partial(jax.jit, donate_argnames=["kvcache"])
def forward_cached(model_params: PixtralModel, next_token_batch, kvcache, batch_next_token_indices, lora_params=None):
  # token embeddings
  hidden_state_BTC = next_token_embedding(model_params, next_token_batch)

  # rope freqs
  head_dim = 128 # params.json
  rope_cos, rope_sin = precompute_specific_position_rope_freqs_1d(batch_next_token_indices, head_dim)

  ## attention
  Hq = 32 # params.json
  Hk = 8 # params.json
  attn_scale = jnp.bfloat16(1.0 / jnp.sqrt(head_dim))
  # create masks
  _, B, _, _, T, d = kvcache.K.shape
  #next_token_mask = (jnp.arange(T)[None, :] == batch_next_token_indices[:, None]) # demarcate location of next token
  padding_mask = (jnp.arange(T)[None, :] > batch_next_token_indices[:, None]) # mask out future, currently-unused padding tokens in attn mechanism
  # loop through attention layers
  # setup scan data and funcs
  layer_count = kvcache.K.shape[0]
  if lora_params:
      def scanf(hidden_state, xfmr_block_data):
        block_params, block_kvcache, block_lora_params = xfmr_block_data
        hidden_state, K, V = transformer_block_cached(block_params, hidden_state, rope_cos, rope_sin, Hq, Hk, head_dim, attn_scale,
                                                      block_kvcache.K, block_kvcache.V, batch_next_token_indices, padding_mask, block_lora_params=block_lora_params)
        return hidden_state, (K, V)
      xfmr_blocks_data = (
          model_params.transformer.transformer_layers, # (40, ...)
          kvcache, # (40, ...)
          lora_params.attention_lora.layers # (40, ...)
      )
  else:
      def scanf(state, xfmr_block_data):
        hidden_state, kvcache = state
        block_idx, block_params = xfmr_block_data
        hidden_state, K, V = transformer_block_cached(block_params, hidden_state, rope_cos, rope_sin, Hq, Hk, head_dim, attn_scale,
                                                      kvcache.K[block_idx],
                                                      kvcache.V[block_idx],
                                                      batch_next_token_indices, padding_mask, block_lora_params=None)
        kvcache = kvcache._replace(
            K=kvcache.K.at[block_idx].set(K),
            V=kvcache.V.at[block_idx].set(V),
        ) # optimization: just update the next token, not the whole block's kvcache
        return (hidden_state, kvcache), None
      xfmr_blocks_data = (
          jnp.arange(layer_count), # layer count
          model_params.transformer.transformer_layers, # (40, ...)
      )
  # do the scan
  state = (hidden_state_BTC, kvcache)
  state, _ = jax.lax.scan(scanf, state, xfmr_blocks_data)
  hidden_state_BTC, kvcache = state
  hidden_state_BTC = layernorm(hidden_state_BTC, model_params.norm_weight, jnp.zeros_like(hidden_state_BTC))
  hidden_state_BTC = hidden_state_BTC @ model_params.output_weight.T # lm head
  # rearrange KVcache to the right shape
  #post_scan_remap = lambda xs_tuple: jnp.stack(xs_tuple) # (K0, K1, K2..Kn) => jax.Array of shape (n,*K.shape)
  #Ks, Vs = kv_pairs # [(K, V), ... (K, V)] => [(K, K, ...), (V, V, ...)]
  #kvcache = KVCache(
  #    K=Ks, # (xfmr block, B, Hk, r=1, T, d)
  #    V=Vs,
  #)
  return hidden_state_BTC, kvcache




@partial(jax.jit, static_argnames=["temperature"], donate_argnames=["kvcache"])
def inference_cached(key, pixtral_params, next_token_batch, kvcache, batch_next_token_indices, temperature, lora_params=None) -> str:
  # get logits
  next_token_batch_logits, kvcache = forward_cached(pixtral_params, next_token_batch,
                                                       kvcache, batch_next_token_indices, lora_params=lora_params) # B, 1, C
  # random sample
  next_token_batch = jrand.categorical(key, next_token_batch_logits/max(temperature, 1e-5), axis=-1) # (B,1)
  return jnp.squeeze(next_token_batch, axis=-1), kvcache
