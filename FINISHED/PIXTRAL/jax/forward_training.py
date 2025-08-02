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
from forward_prefill import vision_encoder, vision_language_adapter, embedding

from typing import NamedTuple, List
# Lora: trains the model to have different facts/instructions
# small matrix that goes over lm head
class LoRA(NamedTuple):
    in_lm_head: jax.Array # (channel, lora_dim)
    out_lm_head: jax.Array # (lora_dim, vocab)

@jax.jit
def apply_lora(model_params: PixtralModel, lora: LoRA):
    # add lora to dense and return the params
    # this would be called in the model's training code. just slip in 'model_params = apply_lora(model_params, lora) before calling forward(model_params, ...)
    lora_mat = in_lm_head @ out_lm_head
    model_params = model_params._replace(lm_head = lm_head + lora_mat) # PSEUDOCODE
    return model_params


def init_lora(key: jax.Array, channel_dim: int, vocab_size: int, lora_dim: int) -> LoRA:
    # create a new lora
    # xiavier initialization?
    in_key, out_key = jrand.split(key)
    return LoRA(
        in_lm_head=jrand.uniform(in_key, (channel_dim, lora_dim), dtype=jnp.bfloat16),
        out_lm_head=jrand.uniform(out_key, (lora_dim, out_lm_head), dtype=jnp.bfloat16),
    )
    


# Custom
# AbstractionLoRA: trains the model to attend to tokens differently
# list of QLoRA, KLoRA, and VLoRA (QKV - NOT 'quantized')
class AbstractionLoRA(NamedTuple):
    in_q:  jax.Array
    out_q: jax.Array
    in_k:  jax.Array
    out_k: jax.Array
    in_v:  jax.Array
    out_v: jax.Array
    in_o:  jax.Array
    out_o: jax.Array



def pixtral_attention_training(
    block_params: TransformerBlock,
    hidden_state_BTC, 
    freqs,
    query_heads,
    kv_heads,
    head_dim
):
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
  Q = apply_rope(Q, freqs)
  K = apply_rope(K, freqs) # both have the same head dim

  K = K[:, :, None, :, :] # broadcast over r
  V = V[:, :, None, :, :] # broadcast over r

  # mistral uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  scale = jnp.bfloat16(1.0 / jnp.sqrt(Q.shape[-1]))
  Q = Q*scale
  attn = Q @ jnp.swapaxes(K, -1, -2)
  attn = jax.nn.softmax(attn, axis=-1)
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn @ block_params.attention_wo_weight.T
  return out.astype(jnp.bfloat16)


# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block_training(
    block_params: TransformerBlock,
    hidden_state_BTC: jax.Array,
    freqs_1d,
    query_heads,
    kv_heads,
    head_dim
) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC = pixtral_attention_training(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC


def mm_forward_training(
    model_params: PixtralModel,
    message_tokens,
    processed_images,
    image_start_indices
):
  hidden_state_BTC = embedding(model_params, message_tokens, processed_images, image_start_indices)[jnp.newaxis, :] # fake batch for now

  B, T, C = hidden_state_BTC.shape
  head_dim = 128 # params.json
  max_pos, d = T, head_dim
  freqs = precompute_rope_freqs_1d(max_pos, d) # mistral does rope after splitting k and q into gqa heads. q and k are split into the same channel size per head

  # attention
  # loop through attention layers
  Hq = 32 # params.json
  Hk = 8 # params.json
  # head dim defined above - it's used to calculate rope1d frequencies
  # scan compiles faster
  def scanf(hidden_state, block_params):
    hidden_state = transformer_block_training(block_params, hidden_state, freqs, Hq, Hk, head_dim)
    return hidden_state, None
  hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.transformer.transformer_layers)
  hidden_state_BTC = layernorm(hidden_state_BTC, model_params.norm_weight, jnp.zeros_like(hidden_state_BTC))
  hidden_state_BTC = hidden_state_BTC @ model_params.output_weight.T # lm head

  post_scan_remap = lambda xs_tuple: jnp.stack(xs_tuple) # (K0, K1, K2..Kn) => jax.Array of shape (n,*K.shape)
  Ks, Vs = kv_pairs # [(K, V), ... (K, V)] => [(K, K, ...), (V, V, ...)]
  kvcache = KVCache(
      K=post_scan_remap(Ks), # (xfmr block, B, Hk, r=1, T, d)
      V=post_scan_remap(Vs),
  )
  return hidden_state_BTC, kvcache


def inference_training(
    key: jax.Array,
    pixtral_params,
    tokens,
    images,
    image_start_indice
) -> str:
  # get logits
  next_token_logits = mm_forward_training(pixtral_params, tokens, images, image_start_indices) # B, T, C
  # random weighted sample
  next_token_probs = jrand.categorical(key, next_token_logits, axis=-1) # (B,T)
  return next_token_probs


def loss(
     lora: LoRA,
     pixtral_params,
     tokens,
     images,
     image_start_indices,
     y,
     key
) -> float:
    # take all inputs
    # get the grads of the lora
    pixtral_params = apply_lora(pixtral_params)

    next_token_logits = mm_forward_training(pixtral_params, tokens, images, image_start_indices)
    next_token_probs = jrand.categorical(key, next_token_logits, axis=-1) # (B,T)

    yprobs = one_hot(y, axis=-1)
    loss = cross_entropy_loss(next_token_probs, y_probs)
    return loss


# experiments
# train a simple lora that writes in all caps
# train a simple lora that 
# do in context learning


# experiments to consider
# sparse autoencoder for chesstral (constantly steers the conversation towards chess)


# write a blog post on how to implement this
# write little blog posts about experiments (make it fun)