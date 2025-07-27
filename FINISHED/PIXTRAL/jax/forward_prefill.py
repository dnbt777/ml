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




def pixtral_attention_prefill(block_params: TransformerBlock, hidden_state_BTC, freqs, query_heads, kv_heads, head_dim):
  # compute qkv
  hidden_state_BTC = hidden_state_BTC
  Q = hidden_state_BTC @ block_params.attention_wq_weight.T
  K = hidden_state_BTC @ block_params.attention_wk_weight.T
  V = hidden_state_BTC @ block_params.attention_wv_weight.T

  # kv cache optional ig
  # for now dont use this. wont be used in training I dont think

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

  # repeat kv to match query (optimization: view or broadcast instead of repeat)

  # uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  #scale = jax.lax.rsqrt(Q.shape[-1]) # rqrt does not accept int64
  scale = 1.0 / jnp.sqrt(Q.shape[-1])
  Q = jnp.float32(Q * scale)
  attn = Q @ jnp.swapaxes(K, -1, -2)
  attn = jax.nn.softmax(attn, axis=-1) # kernel does this
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn @ block_params.attention_wo_weight.T

  return out.astype(jnp.bfloat16), K, V



# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block_prefill(block_params: TransformerBlock, hidden_state_BTC: jax.Array, freqs_1d, query_heads, kv_heads, head_dim) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC, K, V = pixtral_attention_prefill(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC, K, V


def embedding(model_params: PixtralModel, message_tokens, processed_images, image_start_indices) -> jax.Array:
  # gets the embeddings of the tokens
  # already the exact length needed for images. contains img tokens including img_br and img_end
  embeddings = text_embedding(model_params, message_tokens)

  # get image embeddings
  # (call vision encoder) 
  image_embeddings = [vision_encoder(model_params, processed_image) for processed_image in processed_images]  # N, H, W, C

  # replace img tokens with images
  for image_data, start_idx in zip(image_embeddings, image_start_indices):
    image_embedding_TC, H, W, C = image_data 
    image_width = W 
    image_embedding_HWC = jnp.reshape(image_embedding_TC, (H, W, C))
    # add the embeddings for img_break and img_end
    img_break_id = 1
    img_end_id = 2
    img_break_embed = model_params.tok_embeddings_weight[img_break_id]
    img_end_embed = model_params.tok_embeddings_weight[img_end_id]
    breaks_and_img_end = jnp.repeat(img_break_embed[None, None, :], H, axis=0)
    breaks_and_img_end = breaks_and_img_end.at[-1, 0, :].set(img_end_embed)
    padded_image_embedding_HWC = jnp.concatenate([image_embedding_HWC, breaks_and_img_end], axis=1)
    image_embedding_TC = jnp.reshape(padded_image_embedding_HWC, (H*(W+1), C))

    # overwritten_embeddings = embeddings.at[start_idx:image_embedding_TC.shape[0]].set(image_embedding_TC)
    embeddings = jax.lax.dynamic_update_slice(
      embeddings, # dest
      image_embedding_TC, # source # bfloat16 just for now
      (start_idx, 0) # start overwrite index
    )

  return embeddings.astype(jnp.bfloat16)


# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
def vision_encoder(model_params: PixtralModel, processed_image):
  # conv2d on image 
  patch_embeddings_CHW = conv2d(model_params, processed_image)[0]

  C, H, W = patch_embeddings_CHW.shape
  #rope2d_positions = position_meshgrid(patch_embeddings_CHW.shape[1], patch_embeddings_CHW.shape[2])
  num_attention_heads = 16 # params.json
  hidden_dim = C
  head_dim = hidden_dim // num_attention_heads
  freqs_2d = precompute_rope_freqs_2d(H, W, head_dim) # in the future - precompute with max H and W ONCE, and adjust func to deal with any H W array

  freqs_2d = rearrange(freqs_2d, "B H W c -> B (H W) c") # flatten
  
  # flatten patch embeddings
  # (C, H, W) => (P, C)
  flattened_patch_embeddings_PC = rearrange(patch_embeddings_CHW, "C H W -> (H W) C")

  # ln pre (RMSnorm)
  flattened_patch_embeddings_PC = RMSnorm(flattened_patch_embeddings_PC, model_params.vision_encoder.ln_pre_weight)

  # block diagonal mask
  # see vision_encoder.py
  # in a text transformer, keys for future tokens are masked out
  # here, we want to construct a mask that masks out other images' keys
  # so that attention doesn't leak between images
  # for a single image, there is no mask
  # for now we will use placeholders and a single image
  patch_sequences = [flattened_patch_embeddings_PC.shape[0]] # for image in images - placeholder for now that only uses the first image
  #block_diagonal_mask = jnp.zeros((sum(patch_sequences),sum(patch_sequences)), dtype=jnp.bfloat16)
  #left_idx = 0
  #for patch_sequence in patch_sequences:
  #    block_diagonal_mask.at[left_idx:patch_sequence, left_idx:patch_sequence].set(1.0)
  #    left_idx = left_idx + patch_sequence

  # vision transformer blocks
  # loop through attention layers
  hidden_state_BTC = flattened_patch_embeddings_PC[jnp.newaxis, ...]
  _, T, C = hidden_state_BTC.shape
  num_attention_heads = 16 # params.json
  hidden_dim = hidden_state_BTC.shape[-1]
  head_dim = hidden_dim // num_attention_heads
  #freqs_1d = precompute_rope_freqs_1d(T, head_dim) # in pixtral, rope is done after viewing q and k broken into heads

  def scanf(hidden_state, block_params):
    hidden_state, _, K = transformer_block_prefill(block_params, hidden_state, freqs_2d, num_attention_heads, num_attention_heads, head_dim)
    return hidden_state, None
  hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.vision_encoder.vision_encoder_layers)
  hidden_state_TC = hidden_state_BTC[0] # un-batch fake batch

  # call vision language adapter
  hidden_state_TC = vision_language_adapter(model_params.vision_language_adapter, hidden_state_TC)

  return hidden_state_TC, H, W, hidden_state_TC.shape[-1]


def vision_language_adapter(vla_params: VisionLanguageAdapter, hidden_state_TC):
  hidden_state_TC = jax.nn.gelu(hidden_state_TC @ vla_params.w_in_weight.T + vla_params.w_in_bias)
  hidden_state_TC = hidden_state_TC @ vla_params.w_out_weight.T + vla_params.w_out_bias
  return hidden_state_TC


def mm_forward_prefill(model_params: PixtralModel, message_tokens, processed_images, image_start_indices):
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
    hidden_state, K, V = transformer_block_prefill(block_params, hidden_state, freqs, Hq, Hk, head_dim)
    return hidden_state, (K, V)
  hidden_state_BTC, kv_pairs = jax.lax.scan(scanf, hidden_state_BTC, model_params.transformer.transformer_layers)
  hidden_state_BTC = layernorm(hidden_state_BTC, model_params.norm_weight, jnp.zeros_like(hidden_state_BTC))
  hidden_state_BTC = hidden_state_BTC @ model_params.output_weight.T # lm head

  post_scan_remap = lambda xs_tuple: jnp.stack(xs_tuple) # (K0, K1, K2..Kn) => jax.Array of shape (n,*K.shape)
  Ks, Vs = kv_pairs # [(K, V), ... (K, V)] => [(K, K, ...), (V, V, ...)]
  kvcache = KVCache(
      K=post_scan_remap(Ks), # (xfmr block, B, Hk, r=1, T, d)
      V=post_scan_remap(Vs),
  )
  return hidden_state_BTC, kvcache


def inference_prefill(key, pixtral_params, tokens, images, image_start_indices) -> str:
  # get logits
  next_token_logits, kvcache = mm_forward_prefill(pixtral_params, tokens, images, image_start_indices) # B, T, C
  # random weighted sample
  next_token = jrand.categorical(key, next_token_logits[:, -1], axis=-1) # (B,T)
  return next_token, kvcache
