import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from model_types import *
from einops import rearrange


# mm_preprocessor :: [Image] -> [image_tiles or whatever]
from PIL import Image
import cv2

def convert_to_rgb(image: Image) -> Image:
  image = image.convert("RGBA")
  bg = Image.new("RGBA", image.size, "WHITE")
  bg.paste(image, (0, 0), image)
  rgb_img = bg.convert("RGB")
  return rgb_img


def process_image(image: Image):
  ## get embeddings of img
  # convert to rgb
  image = convert_to_rgb(image)

  # resize img
  max_image_size = 1024 # from params.json
  patch_size = 16 # from params.json
  height, width = image.size
  resize_factor = max(height / max_image_size, width / max_image_size)
  if resize_factor > 1:
    height = round(height / resize_factor)
    width = round(width / resize_factor)
  height_tokens = int(jnp.ceil(height / patch_size))
  width_tokens = int(jnp.ceil(width / patch_size))

  new_size = (height_tokens*patch_size, width_tokens*patch_size) # no padding! nice!
  image = cv2.resize(np.array(image, dtype=np.float32), new_size, interpolation=cv2.INTER_CUBIC)

  # rescale/normalize
  # https://github.com/mistralai/mistral-common/blob/9a38768468fe012aac04bea4d3c33fdd0dd1fd59/src/mistral_common/tokens/tokenizers/multimodal.py#L67
  DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
  DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
  image = image / 255.0
  image = (image - DATASET_MEAN) / DATASET_STD
  
  # CHANNEL_FIRST format
  image = jnp.transpose(image, (2, 0, 1))

  ## get tokens

  # create tokens
  img_token_id = 0
  img_break_id = 1
  img_end_id = 2 # placeholders
  image_tokens = [[img_token_id for _ in range(width_tokens)] + [img_break_id] for _ in range(height_tokens)]
  image_tokens = [item for sublist in image_tokens for item in sublist] # wtf python weird ass syntax
  image_tokens = image_tokens + [img_end_id]

  return (image_tokens, image)





# tokenizer :: [string] -> [[token]]
#  - preprocess
#  - tokenize
# https://docs.mistral.ai/guides/tokenization/
import json
#def load_tokenizer(path: str) -> dict:
#  with open(path, 'r') as file:
#    tokenizer_config = json.load(file)
#  return tokenizer_config
#
#def tokenizer():
 # # loads tekken (mistrals tiktoken based bpe)
 # load_tokenizer("./pixtral/tekken.json")
#
#  pass # return tokens, image_idxs 
# for now just use mistrals. come back to this later
from mistral_common.tokens.tokenizers.tekken import Tekkenizer 
tok = Tekkenizer.from_file("./pixtral/tekken.json")

def encode(string):
  return tok.encode(string, bos=True, eos=True)

def decode(string):
  return tok.decode(string)



def pixtral_attention(block_params: TransformerBlock, hidden_state_BTC, freqs, query_heads, kv_heads, head_dim):
  # kv cache (ignore for now lol)
  # scaled dot prod attention
  
  # compute qkv
  Q = hidden_state_BTC @ block_params.attention_wq_weight.T
  K = hidden_state_BTC @ block_params.attention_wk_weight.T
  V = hidden_state_BTC @ block_params.attention_wv_weight.T

  # kv cache optional ig
  
  # rope1D
  Q = rope1d(Q)
  K = rope1d(K)

  # split into heads (GQA)
  Hk=kv_heads
  Hq=query_heads
  repeats = Hq // Hk 
  Q = rearrange(Q, "B T (Hk r d) -> B Hk r T d", Hk=Hk, r=repeats, d=head_dim)
  K = rearrange(K, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)
  V = rearrange(V, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)

  K = K[:, :, None, :, :] # broadcast over r
  V = V[:, :, None, :, :] # broadcast over r

  # repeat kv to match query (imo, this is a waste, and there should be an op that does gqa attn w/o duping memory)

  # uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  # equivalent to normal attention. no sliding window ig? yaaay
  scale = 1 / jnp.sqrt(Q.shape[-1])
  Q = Q * scale
  attn = Q @ jnp.swapaxes(K, -1, -2) 
  attn = jax.nn.softmax(attn, axis=-1)
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)") 
  out = attn @ block_params.attention_wo_weight.T

  return out


def feed_forward(block_params: TransformerBlock, hidden_state_BTC: jax.Array) -> jax.Array:
  x = hidden_state_BTC
  x1 = jax.nn.silu(x @ block_params.feed_forward_w1_weight.T)
  x3 = x @ block_params.feed_forward_w3_weight.T
  x2 = (x1 * x3) @ block_params.feed_forward_w2_weight.T # element-wise multiplication followed by matmul
  return x2


# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block(block_params: TransformerBlock, hidden_state_BTC: jax.Array, freqs, query_heads, kv_heads, head_dim) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight, jnp.zeros_like(hidden_state_BTC)) ## attention norm
  residual_BTC = pixtral_attention(block_params, residual_BTC, freqs, query_heads, kv_heads, head_dim)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight, jnp.zeros_like(hidden_state_BTC)) ## ff norm
  residual_BTC = feed_forward(block_params, hidden_state_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC



def degen_conv2d(model_params: PixtralModel, image_CHW) -> jax.Array:
  # for now just break into patches and matmul
  patch_size = 16
  H, W = image_CHW.shape[1:]
  h, w = H//patch_size, W//patch_size
  patches_HWI = rearrange(image_CHW, "C (h hp) (w wp) -> h w (C hp wp)", h=h, w=w, hp=patch_size, wp=patch_size)
  # matmul I => O
  patch_embeddings_HWO = patches_HWI @ jnp.reshape(
    model_params.vision_encoder.patch_conv_weight,
    (model_params.vision_encoder.patch_conv_weight.shape[0], -1)).T # may need to reshape conv weight to be collapsed (probably tbh...)
  # if errors - try doing the actual conv operation. maybe degen conv is not the way to go
  return patch_embeddings_HWO


def RMSnorm(hidden_state, weight, bias):
  eps = 1e-5
  hidden_state = hidden_state / jnp.sqrt(jnp.mean(hidden_state**2) + eps)
  hidden_state = hidden_state*weight
  hidden_state = hidden_state + bias
  return hidden_state 


def compute_freqs_1d(dim, max_length):
  theta = 10_000.0 # params.json
  inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
  position = jnp.arange(0, max_length)
  angles = position[:, jnp.newaxis] * inv_freq[jnp.newaxis, :]
  re, im = jnp.sin(angles), jnp.cos(angles)
  return re, im 


def compute_freqs_2d(dim, max_height, max_width):
  re_h, im_h = compute_freqs_1d(dim//2, max_height)
  re_w, im_w = compute_freqs_1d(dim//2, max_width)
  

  re_h = jnp.repeat(re_h[:, jnp.newaxis, :], max_width, axis=1) # add a W dim to the H-based freqs
  im_h = jnp.repeat(im_h[:, jnp.newaxis, :], max_width, axis=1) # add a W dim to the H-based freqs
  re_w = jnp.repeat(re_w[jnp.newaxis, :, :], max_height, axis=0) # add an H dim to the W-based freqs
  im_w = jnp.repeat(im_w[jnp.newaxis, :, :], max_height, axis=0)

  re = jnp.stack([re_h, re_w], axis=-1)
  re = rearrange(re, "H W D n -> H W (D n)")

  im = jnp.stack([im_h, im_w], axis=-1)
  im = rearrange(im, "H W D n -> H W (D n)")
  return re, im


def rope1d(hidden_state_BTC):
  re, im = compute_freqs_1d(hidden_state_BTC.shape[-1], hidden_state_BTC.shape[0])
  return rope(hidden_state_BTC, re, im)


def position_meshgrid(height, width):
  rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
  return jnp.stack([rows.ravel(), cols.ravel()], axis=1)


# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py
def rope2d(hidden_state_BTC, positions, H, W, C):
  # rope1D, but the frequencies are interleaved h and w positions
  # rope1D: [f(pos, dim0), g(pos, dim0), f(pos, dim1), g(pos, dim1)]
  # rope2D: [f(posh, dim0), g(posh, dim0), f(posw, dim1), g(posw, dim1)]
  re, im = compute_freqs_2d(C, H, W)
  rows, cols = positions[:, 0], positions[:, 1]
  re = re[rows, cols]
  im = im[rows, cols]

  return rope(hidden_state_BTC, re, im)


def rope(hidden_state_BTC, re, im):
  hidden_even = hidden_state_BTC[..., ::2]
  hidden_odd = hidden_state_BTC[..., 1::2]
  
  hidden_rotated_even = hidden_even * re - hidden_odd * im
  hidden_rotated_odd = hidden_even * im + hidden_odd * re

  hidden_state_2BTC = jnp.array([hidden_rotated_even, hidden_rotated_odd])
  hidden_state_BTC = rearrange(hidden_state_2BTC, "n B T C -> B T (n C)")
  return hidden_state_BTC
  


def vision_language_adapter(vla_params: VisionLanguageAdapter, hidden_state_TC):
  hidden_state_TC = jax.nn.gelu(hidden_state_TC @ vla_params.w_in_weight.T + vla_params.w_in_bias)
  hidden_state_TC = hidden_state_TC @ vla_params.w_out_weight.T + vla_params.w_out_bias
  return hidden_state_TC



# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
def vision_encoder(model_params: PixtralModel, processed_image):
  # conv2d on image 
  patch_embeddings_HWC = degen_conv2d(model_params, processed_image)
  H, W, C = patch_embeddings_HWC.shape
  rope2d_positions = position_meshgrid(patch_embeddings_HWC.shape[0], patch_embeddings_HWC.shape[1])

  # flatten patch embeddings
  # (H, W, C) => (P, C)
  flattened_patch_embeddings_PC = jnp.reshape(patch_embeddings_HWC, (-1, patch_embeddings_HWC.shape[-1]))

  # ln pre (RMSnorm)
  flattened_patch_embeddings_PC = RMSnorm(flattened_patch_embeddings_PC, model_params.vision_encoder.ln_pre_weight, jnp.zeros_like(flattened_patch_embeddings_PC))

  # rope
  flattened_patch_embeddings_PC = rope2d(flattened_patch_embeddings_PC[None, :], rope2d_positions, H, W, C)[0] # intermediate fake batch
  
  # vision transformer blocks
  # loop through attention layers
  hidden_state_BTC = flattened_patch_embeddings_PC[None, :]
  freqs = compute_freqs_2d(C, H, W)
  num_attention_heads = 16 # params.json
  hidden_dim = hidden_state_BTC.shape[-1]
  head_dim = hidden_dim // num_attention_heads
  def scanf(hidden_state, block_params):
    hidden_state = transformer_block(block_params, hidden_state, freqs, num_attention_heads, num_attention_heads, head_dim)#, kvcache)
    return hidden_state, None
  
  hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.vision_encoder.vision_encoder_layers)
  hidden_state_TC = hidden_state_BTC[0] # un-batch fake batch
  
  # call vision language adapter
  hidden_state_TC = vision_language_adapter(model_params.vision_language_adapter, hidden_state_TC)

  return hidden_state_TC, H, W, hidden_state_TC.shape[-1]


def embedding(model_params: PixtralModel, message_tokens, processed_images, image_start_indices) -> jax.Array:
  # gets the embeddings of the tokens
  # already the exact length needed for images. contains img tokens including img_br and img_end
  message_tokens = jnp.array(message_tokens, dtype=int)
  embeddings = jnp.take(model_params.tok_embeddings_weight, message_tokens, axis=0) # one-hot but faster ig
  
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
      image_embedding_TC.astype(jnp.bfloat16), # source # bfloat16 just for now
      (start_idx, 0) # start overwrite index
    )

  return embeddings


def layernorm(hidden_state_BTC, weight, bias):
  mean = jnp.mean(hidden_state_BTC)
  std = jnp.std(hidden_state_BTC)
  hidden_state_BTC = (hidden_state_BTC - mean)
  hidden_state_BTC = hidden_state_BTC/std
  hidden_state_BTC = hidden_state_BTC*weight # element-wise
  hidden_state_BTC = hidden_state_BTC + bias # element-wise
  return hidden_state_BTC
  


@jax.jit
def mm_forward(model_params: PixtralModel, message_tokens, processed_images, image_start_indices) -> jax.Array:
  # get embeddings
  hidden_state_BTC = embedding(model_params, message_tokens, processed_images, image_start_indices)[jnp.newaxis, :] # fake batch for now

  freqs = compute_freqs_1d(hidden_state_BTC.shape[-1], hidden_state_BTC.shape[1])

  # attention
  # loop through attention layers
  Hq = 32
  Hk = 8
  head_dim = 128 # params.json
  def scanf(hidden_state, block_params):
    hidden_state = transformer_block(block_params, hidden_state, freqs, Hq, Hk, head_dim).astype(jnp.bfloat16)#, kvcache)
    return hidden_state, None
  
  hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.transformer.transformer_layers)
    # ffw - gelu gated idk
  
  hidden_state_BTC = layernorm(hidden_state_BTC, model_params.norm_weight, jnp.zeros_like(hidden_state_BTC))
  hidden_state_BTC = hidden_state_BTC @ model_params.output_weight.T # lm head

  return hidden_state_BTC


#inference :: [Union(Image, string)] -> [string]
# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
import jax.random as jrand
from functools import partial
def inference(key, pixtral_params, messages) -> str:
  # token IDS
  START = -1 # <s> # placeholder values for now
  INS_START = 0 # [INS]
  INS_END = 1 # [/INS]

  tokens = [START]
  images = []
  image_start_indices = []
  for message in messages:
    if message["role"] == "user":
      tokens.append(INS_START)
      if type(message["content"]) == str:
        tokens = tokens + encode(message["content"])
      else:
        for content in message["content"]:
          if content["type"] == "text":
            tokens = tokens + encode(content["text"])
          elif content["type"] == "image_url":
            image_start_index = len(tokens)
            image_start_indices.append(image_start_index)
            source = content["image_url"]["url"]
            if ('https://' in source) or ('http://' in source):
              pass # requests.get
            elif 'data:image' in source:
              pass # base64
            else:
              # file
              image = Image.open(source)
              image_tokens, processed_image = process_image(image)
              tokens = tokens + image_tokens
              images.append(processed_image)
          else:
            raise NameError(f"Error processing messages. Unknown content type {content["type"]}")
      tokens.append(INS_END)
    elif message["role"] == "assistant": # assistant?
      tokens = tokens + encode(message["content"])
    else:
      raise NameError(f"Error processing messages. Unknown role {message["role"]}")
  # get logits
  logits = mm_forward(pixtral_params, tokens, images, image_start_indices) # (B, T)
  next_token = logits[..., -1] # (B,)
  # random sample
  next_token = jrand.categorical(key, next_token, axis=-1) # (B,)
  # return
  return next_token # (B,)


