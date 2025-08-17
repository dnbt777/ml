import jax
import jax.numpy as jnp
import jax.random as jrand
from einops import rearrange
from functools import partial

from typing import List
from model_types import *
from forward_common import *

# the code is best read from bottom funcs to top funcs

def pixtral_attention_prefill(block_params: TransformerBlock, hidden_state_BTC, freqs,
                              query_heads, kv_heads, head_dim, attn_mask, block_lora_params=None):
  # compute qkv
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
  #attn_mask = rearrange(attn_mask, "(B Hk r) Tq Tk -> B Hk r Tq Tk", Hk=1, r=1)
  attn = jnp.where(attn_mask, jnp.bfloat16(-jnp.inf), attn) 
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(jnp.bfloat16) # do attn softmax in float32
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn @ block_params.attention_wo_weight.T
  if block_lora_params:
      out = out + block_lora_params.alpha_o*((attn @ block_lora_params.in_o) @ block_lora_params.out_o)

  return out.astype(jnp.bfloat16), K, V



# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block_prefill(
    block_params: TransformerBlock,
    hidden_state_BTC: jax.Array,
    freqs_1d, query_heads, kv_heads, head_dim,
    attn_mask, block_lora_params=None) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC, K, V = pixtral_attention_prefill(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim, attn_mask, block_lora_params=block_lora_params)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC, K, V




def forward_prefill(model_params, hidden_state_BTC, batch_next_token_indices, batch_attn_mask, lora_params=None):
  B, T, C = hidden_state_BTC.shape
  head_dim = 128 # params.json
  max_pos, d = T, head_dim
  freqs = precompute_rope_freqs_1d(max_pos, d) # mistral does rope after splitting k and q into gqa heads. q and k are split into the same channel size per head

  # attention layers
  Hq = 32 # params.json
  Hk = 8 # params.json
  attn_mask = get_causal_mask(T)[None, None, :, :]
  attn_mask = jnp.logical_or(batch_attn_mask[:, None, None, None, :], attn_mask)  # if True in either mask, mask out token
  # head dim defined above - it's used to calculate rope1d frequencies
  # scan compiles faster than a for loop
  if lora_params:
    def scanf(hidden_state, carry):
      xfmr_block_params, block_lora_params = carry
      hidden_state, K, V = transformer_block_prefill(xfmr_block_params, hidden_state, freqs, Hq, Hk, head_dim, attn_mask, block_lora_params=block_lora_params)
      return hidden_state, (K, V)
    hidden_state_BTC, kv_pairs = jax.lax.scan(scanf, hidden_state_BTC, (model_params.transformer.transformer_layers, lora_params.attention_lora.layers))
  else:
      def scanf(hidden_state, xfmr_block_params):
        hidden_state, K, V = transformer_block_prefill(xfmr_block_params, hidden_state, freqs, Hq, Hk, head_dim, attn_mask, block_lora_params=None)
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
  # USE NEXT TOKEN INDICES
  B = hidden_state_BTC.shape[0]
  hidden_state_BC = hidden_state_BTC[jnp.arange(B, dtype=int), batch_next_token_indices-1, :]
  # layernorm
  hidden_state_BC = layernorm(hidden_state_BC, model_params.norm_weight, jnp.zeros((1, hidden_state_BTC.shape[-1])))
  # lm_head: channel -> vocab logits
  hidden_state_BC = hidden_state_BC @ model_params.output_weight.T # (B, C) @ (C, vocab) => (B, vocab)
  return hidden_state_BC, kvcache



def text_forward_prefill(model_params: PixtralModel, batch_tokens, batch_next_token_indices, batch_attn_mask, lora_params=None):
  hidden_state_BTC = text_embedding(model_params, batch_tokens)
  return forward_prefill(model_params, hidden_state_BTC, batch_next_token_indices, batch_attn_mask, lora_params=lora_params)



def mm_forward_prefill(model_params: PixtralModel, batch_tokens, batch_image_sets,
                       batch_intext_image_start_indices, batch_next_token_indices, batch_attn_mask, lora_params=None):
  hidden_state_BTC = multimodal_embedding(model_params, batch_tokens, batch_image_sets, batch_intext_image_start_indices)
  return forward_prefill(model_params, hidden_state_BTC, batch_next_token_indices, batch_attn_mask, lora_params=lora_params)



def inference_prefill(key,
                      pixtral_params, batch_tokens, batch_image_sets, batch_intext_image_start_indices,
                      batch_next_token_indices, batch_attn_mask, temperature, lora_params=None) -> str:
  if any([len(image_set) > 0 for image_set in batch_image_sets]):
      batch_next_token_logit, kvcache = mm_forward_prefill(
          pixtral_params, batch_tokens, batch_image_sets,
          batch_intext_image_start_indices, batch_next_token_indices, batch_attn_mask,
          lora_params=lora_params
      )
  else:
      batch_next_token_logit, kvcache = text_forward_prefill(
          pixtral_params, batch_tokens,
        batch_next_token_indices, batch_attn_mask, lora_params=lora_params
      )
  batch_next_token = jrand.categorical(key, batch_next_token_logit/max(temperature, 1e-5), axis=-1)
  return batch_next_token, kvcache



