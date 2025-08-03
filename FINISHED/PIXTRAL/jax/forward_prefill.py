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
def transformer_block_prefill(block_params: TransformerBlock, hidden_state_BTC: jax.Array, freqs_1d, query_heads, kv_heads, head_dim, attn_mask) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC, K, V = pixtral_attention_prefill(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim, attn_mask)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC, K, V


def multimodal_embedding(model_params: PixtralModel, message_tokens, processed_images, image_intext_start_indices) -> jax.Array:
  # gets the embeddings of the tokens
  # already the exact length needed for images. contains img tokens including img_br and img_end
  text_embeddings = text_embedding(model_params, message_tokens)

  # get image embeddings
  image_embeddings = vision_encoder(model_params, processed_images)

  # replace img tokens with images
  patches_C = image_embeddings.shape[-1]
  patch_size = 16 # params.json
  inimg_start_idx = 0
  for i, intext_start_idx in enumerate(image_intext_start_indices):
    pixels_C, pixels_H, pixels_W = processed_images[i].shape
    patches_H, patches_W = pixels_H//patch_size, pixels_W//patch_size
    inimg_end_idx = inimg_start_idx + patches_H*patches_W # size(unformatted patches) = img patches H * img patches W
    intext_end_idx = intext_start_idx + patches_H*(patches_W + 1) # size(final patches) = img patches + break tokens + end token
    #print(image_embeddings.shape, intext_start_idx, intext_end_idx, patches_H, patches_W)
    # there are two start indexes we will need. one is for the image inside of the image_embeddings. the other is where the image starts in the text_embeddings.
    image_embedding_TC = image_embeddings[inimg_start_idx:inimg_end_idx, ...]
    image_embedding_HWC = jnp.reshape(image_embedding_TC, (patches_H, patches_W, patches_C))
    # add the embeddings for img_break and img_end
    img_break_token_id, img_end_token_id = 1, 2
    img_break_embed = model_params.tok_embeddings_weight[img_break_token_id]
    img_end_embed = model_params.tok_embeddings_weight[img_end_token_id]
    img_formatting_tokens = jnp.repeat(img_break_embed[None, None, :], patches_H, axis=0) # add img break tokens to each row
    img_formatting_tokens = img_formatting_tokens.at[-1, 0, :].set(img_end_embed) # replace last row's break token with an end token
    formatted_image_embedding_HWC = jnp.concatenate([image_embedding_HWC, img_formatting_tokens], axis=1)
    formatted_image_embedding_TC = jnp.reshape(formatted_image_embedding_HWC, (patches_H*(patches_W+1), patches_C))

    text_embeddings = jax.lax.dynamic_update_slice(
      text_embeddings, # dest
      formatted_image_embedding_TC, # source # bfloat16 just for now
      (intext_start_idx, 0) # overwrite index = starting index in text tokens
    )
    inimg_start_idx = inimg_end_idx

  return text_embeddings.astype(jnp.bfloat16)



def embeddings_and_freqs(model_params, processed_image):
  patch_embeddings_CHW = conv2d(model_params, processed_image)[0]
  C, H, W = patch_embeddings_CHW.shape
  num_attention_heads = 16 # params.json
  hidden_dim = C
  head_dim = hidden_dim // num_attention_heads
  freqs_2d = precompute_rope_freqs_2d(H, W, head_dim) # in the future - precompute with max H and W ONCE, and adjust func to deal with any H W array
  freqs_2d = rearrange(freqs_2d, "B H W c -> B (H W) c") # flatten
  # flatten patch embeddings
  flattened_patch_embeddings_PC = rearrange(patch_embeddings_CHW, "C H W -> (H W) C")
  # ln pre (RMSnorm)
  flattened_patch_embeddings_PC = RMSnorm(flattened_patch_embeddings_PC, model_params.vision_encoder.ln_pre_weight)
  return flattened_patch_embeddings_PC, freqs_2d
    


def create_block_diagonal_mask(flattened_patch_embeddings_list):
  patch_counts = [embeds.shape[0] for embeds in flattened_patch_embeddings_list]
  total_patch_count = sum(patch_counts)
  block_diagonal_mask = jnp.ones((total_patch_count, total_patch_count), dtype=bool)
  start_patch_idx = 0
  for patch_count in patch_counts:
      block_diagonal_mask = block_diagonal_mask.at[start_patch_idx:start_patch_idx+patch_count, start_patch_idx:start_patch_idx+patch_count].set(False)
      start_patch_idx = start_patch_idx + patch_count
  return block_diagonal_mask



# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
def vision_encoder(model_params: PixtralModel, processed_images):
  flattened_patch_embeddings_list, rope_2d_freqs = zip(*[embeddings_and_freqs(model_params, processed_image) for processed_image in processed_images])
  flattened_patch_embeddings_PC = jnp.concatenate(flattened_patch_embeddings_list, axis=0)
  rope_2d_freqs = jnp.concatenate(rope_2d_freqs, axis=1)

  # block diagonal mask
  attn_mask = create_block_diagonal_mask(flattened_patch_embeddings_list)

  # vision transformer blocks
  hidden_state_BTC = flattened_patch_embeddings_PC[jnp.newaxis, ...] # fake batch
  _, T, C = hidden_state_BTC.shape
  num_attention_heads = 16 # params.json
  hidden_dim = hidden_state_BTC.shape[-1]
  head_dim = hidden_dim // num_attention_heads

  def scanf(hidden_state, block_params):
    hidden_state, _, _ = transformer_block_prefill(block_params, hidden_state, rope_2d_freqs, num_attention_heads, num_attention_heads, head_dim, attn_mask)
    return hidden_state, None
  hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.vision_encoder.vision_encoder_layers)
  hidden_state_TC = hidden_state_BTC[0] # un-batch fake batch

  # vision language adapter
  hidden_state_TC = vision_language_adapter(model_params.vision_language_adapter, hidden_state_TC)
  return hidden_state_TC#, H, W, hidden_state_TC.shape[-1]


    
def vision_language_adapter(vla_params: VisionLanguageAdapter, hidden_state_TC):
  hidden_state_TC = jax.nn.gelu(hidden_state_TC @ vla_params.w_in_weight.T + vla_params.w_in_bias)
  hidden_state_TC = hidden_state_TC @ vla_params.w_out_weight.T + vla_params.w_out_bias
  return hidden_state_TC



def get_causal_mask(T: int) -> jax.Array:
    mask = jnp.ones((T, T), dtype=bool)
    mask = jnp.triu(mask, k=1)
    return mask


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
