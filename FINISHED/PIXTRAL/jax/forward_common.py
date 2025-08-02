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
  width, height = image.size
  resize_factor = max(height / max_image_size, width / max_image_size)
  if resize_factor > 1:
    height = round(height / resize_factor)
    width = round(width / resize_factor)
  height_tokens = int(jnp.ceil(height / patch_size))
  width_tokens = int(jnp.ceil(width / patch_size))

  new_size = (width_tokens*patch_size, height_tokens*patch_size) # no padding! nice!
  image = cv2.resize(np.array(image, dtype=np.float32), new_size, interpolation=cv2.INTER_CUBIC)

  # rescale/normalize
  # https://github.com/mistralai/mistral-common/blob/9a38768468fe012aac04bea4d3c33fdd0dd1fd59/src/mistral_common/tokens/tokenizers/multimodal.py#L67
  DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
  DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
  image = image / 255.0
  image = (image - DATASET_MEAN) / DATASET_STD
  
  # CHANNEL_FIRST format
  image = jnp.array(image, dtype=jnp.bfloat16) # explicitly convert to jnp array w dtype float32. when x64 is enabled, implicit => float64 and breaks this
  image = jnp.transpose(image, (2, 0, 1))

  # create tokens
  img_token_id = 10 # from params.json
  img_break_id = 12 # from params.json
  img_end_id = 13 # from params.json
  image_tokens = [[img_token_id for _ in range(width_tokens)] + [img_break_id] for _ in range(height_tokens)]
  image_tokens = [item for sublist in image_tokens for item in sublist] # flatten img token list. 2d->1d. also wtf python weird ass syntax
  image_tokens[-1] = img_end_id # replace final row's break token with img end token

  return (image_tokens, image)


# https://docs.mistral.ai/guides/tokenization/
# Just use mistral's tokenizer for now
import json
from mistral_common.tokens.tokenizers.tekken import Tekkenizer 
tok = Tekkenizer.from_file("../pixtral/tekken.json")

def encode(string, add_special=True):
  return tok.encode(string, bos=add_special, eos=add_special)

def decode(string):
  return tok.decode(string)

#inference :: [Union(Image, string)] -> [string]
# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
def tokenize_messages_dict(messages, add_special=False):
  # token IDS https://github.com/mistralai/mistral-common/issues/105#issuecomment-2997200779
  BOS = 1 # <s>
  INS_START = 3 # [INS]
  INS_END = 4 # [/INS]
  EOS = 2 # </s>

  tokens = []
  images = []
  image_start_indices = []
  for message in messages:
    if message["role"] == "user":
      tokens.append(BOS)
      tokens.append(INS_START)
      if type(message["content"]) == str:
        tokens = tokens + encode(message["content"], add_special=add_special)
      else:
        for content in message["content"]:
          if content["type"] == "text":
            tokens = tokens + encode(content["text"], add_special=add_special)
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
      # https://huggingface.co/mistral-community/pixtral-12b
      # append at end of user prompt
      # see 'usage example' in the link above
      tokens.append(INS_END)
    elif message["role"] == "assistant":
      tokens = tokens + encode(message["content"]) + [EOS]
    else:
      raise NameError(f"Error processing messages. Unknown role {message["role"]}")
    
  return tokens, images, image_start_indices


@jax.jit
def feed_forward(block_params: TransformerBlock, hidden_state_BTC: jax.Array) -> jax.Array:
  x = hidden_state_BTC
  x1 = jax.nn.silu(x @ block_params.feed_forward_w1_weight.T)
  x3 = x @ block_params.feed_forward_w3_weight.T
  x2 = (x1 * x3) @ block_params.feed_forward_w2_weight.T
  return x2

@jax.jit
def conv2d(model_params: PixtralModel, image_CHW) -> jax.Array:
    patch_size = 16 # params.json
    H, W = image_CHW.shape[1:]
    h, w = H//patch_size, W//patch_size
    patch_embeddings_HWO = jax.lax.conv_general_dilated(
        image_CHW[jnp.newaxis, ...],
        model_params.vision_encoder.patch_conv_weight,
        (patch_size, patch_size),
        'SAME',
        preferred_element_type=jnp.float32, # OPTIMIZATION: test in bfloat16 (do a grep on float32 actually...)
        precision=jax.lax.Precision.DEFAULT
    )
    return patch_embeddings_HWO.astype(jnp.bfloat16)

@jax.jit
def RMSnorm(hidden_state, weight):
  eps = 1e-5
  hidden_state = hidden_state
  weight = weight
  squared = jax.lax.pow(hidden_state, 2)
  mean = (jnp.mean(squared, axis=-1, keepdims=True) + eps)
  rsqrt = jax.lax.rsqrt(mean)
  hidden_state = jnp.multiply(hidden_state, rsqrt)
  hidden_state = hidden_state
  hidden_state = jnp.multiply(hidden_state, weight)
  return hidden_state

@partial(jax.jit, static_argnames=["max_i", "d"])
def precompute_rope_freqs_1d(max_i, d):
    rope_theta = 1_000_000_000 # params.json
    max_j = d//2
    i, j = jnp.arange(max_i), jnp.arange(max_j)
    freqs_ij = jnp.outer(i, rope_theta**(-2.0*j/d)) # i, d//2
    cos, sin = jnp.cos(freqs_ij)[None, :], jnp.sin(freqs_ij)[None, :]
    return jax.lax.complex(cos, sin).astype(jnp.complex64)


@jax.jit
def apply_rope(hidden_state, freqs):
    original_shape = hidden_state.shape # ..., T, C
    T, C = hidden_state.shape[-2:]
    d = C
    hidden_state_pairs = jnp.reshape(hidden_state, (*hidden_state.shape[:-2], T, d//2, 2))
    re, im = hidden_state_pairs[..., 0], hidden_state_pairs[..., 1] # b, i, d//2
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
    cos, sin = jnp.real(freqs), jnp.imag(freqs)
    re_rot = re*cos - im*sin
    im_rot = im*cos + re*sin
    hidden_state_pairs_rot = jnp.concatenate([re_rot, im_rot], axis=-1) # (B, T, d//2, 2)
    hidden_state_rot = jnp.zeros_like(hidden_state)
    hidden_state_rot = hidden_state_rot.at[..., ::2].set(re_rot)
    hidden_state_rot = hidden_state_rot.at[..., 1::2].set(im_rot) # [re, im, re, im, re, im, ... ]
    return hidden_state_rot.astype(jnp.bfloat16)

@partial(jax.jit, static_argnames=["max_h", "max_w", "d"])
def precompute_rope_freqs_2d(max_h, max_w, d):
    rope_theta = 10_000 # params.json
    max_j = d//2
    h, w, j = jnp.arange(max_h), jnp.arange(max_w), jnp.arange(0, d, 2, dtype=jnp.bfloat16) # mimics -2.0 * (j from 0 to d//2)
    aten_div = j/d
    aten_pow = rope_theta**aten_div
    aten_pow = aten_pow
    base_freqs = jnp.reciprocal(aten_pow)
    thetas_hj = jnp.outer(h, base_freqs[::2]) # H, d//2
    thetas_hj = thetas_hj[:, jnp.newaxis, :] # H, d//2 => H, 1, d//2
    thetas_wj = jnp.outer(w, base_freqs[1::2]) # W, d//2
    thetas_wj = thetas_wj[jnp.newaxis, :, :] # 1, W, d//2
    
    # calculate hpos based rotations
    # pixtral uses aten::polar, which is just 1.0*cos(theta), 1.0*sin(theta)
    cos_h, sin_h = jnp.cos(thetas_hj)[jnp.newaxis, :].astype(jnp.float32), jnp.sin(thetas_hj)[jnp.newaxis, :].astype(jnp.float32) # add batch dim to both
    freqs_h = jax.lax.complex(cos_h, sin_h).astype(jnp.complex64) # 1, H, 1, d//4 # take the evens
    cos_w, sin_w = jnp.cos(thetas_wj)[jnp.newaxis, :].astype(jnp.float32), jnp.sin(thetas_wj)[jnp.newaxis, :].astype(jnp.float32)
    freqs_w = jax.lax.complex(cos_w, sin_w).astype(jnp.complex64) # 1, 1, W, d//4 # take the odds

    # make frequency grid
    freqs_w = jnp.repeat(freqs_w, repeats=max_h, axis=1)
    freqs_h = jnp.repeat(freqs_h, repeats=max_w, axis=2)
    freqs_2d = jnp.concatenate([freqs_h, freqs_w], axis=-1) # 1, H, W, d//2
    return freqs_2d.astype(jnp.complex64) # explicit

@jax.jit
def text_embedding(model_params: PixtralModel, text_tokens) -> jax.Array:
  text_tokens = jnp.array(text_tokens, dtype=int)
  embeddings = jnp.take(model_params.tok_embeddings_weight, text_tokens, axis=0) # one-hot but faster ig
  return embeddings

@jax.jit
def layernorm(hidden_state_BTC, weight, bias):
  mean = jnp.mean(hidden_state_BTC, axis=-1, keepdims=True)
  std = jnp.std(hidden_state_BTC, axis=-1, keepdims=True)
  hidden_state_BTC = (hidden_state_BTC - mean)
  hidden_state_BTC = hidden_state_BTC/std
  hidden_state_BTC = hidden_state_BTC*weight # element-wise
  hidden_state_BTC = hidden_state_BTC + bias # element-wise
  return hidden_state_BTC






