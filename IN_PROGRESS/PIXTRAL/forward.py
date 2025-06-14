import jax
import jax.numpy as jnp
from typing import List
from model_types import *
from einops import rearrange


# mm_preprocessor :: [Image] -> [image_tiles or whatever]
from PIL import Image
import cv2

def convert_to_rgb(image: Image) -> Image:
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
  height_tokens = jnp.ceil(height / patch_size) 
  width_tokens = jnp.ceil(width / patch_size) 

  new_size = (height_tokens*patch_size, width_tokens*patch_size) # no padding! nice!
  image = cv2.resize(jnp.array(image, dtype=jnp.float32), new_size, interpolation=cv2.INTER_CUBIC)

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
  image_tokens = [[img_token_id for _ in width_tokens] + [img_break_id] for _ in height_tokens]
  image_tokens = [item for sublist in image_tokens for item in sublist] # wtf python weird ass syntax
  image_tokens = image_tokens + [img_end_id]

  return (image_tokens, image)





# tokenizer :: [string] -> [[token]]
#  - preprocess
#  - tokenize
# https://docs.mistral.ai/guides/tokenization/
import json
def load_tokenizer(path: str) -> dict:
  with open(path, 'r') as file:
    tokenizer_config = json.load(file)
  return tokenizer_config

def tokenizer():
  # loads tekken (mistrals tiktoken based bpe)
  load_tokenizer("./pixtral/tekken.json")

  pass # return tokens, image_idxs 
# for now just use mistrals. come back to this later
from mistral_common.tokens.tokenizers.tekken import Tekkenizer 
tok = Tekkenizer.from_file("./pixtral/tekken.json")

def encode(string):
  return tok.encode(string, add_bos=True, add_eos=True)

def decode(string):
  return tok.decode(string)



def pixtral_attention(block_params: TransformerBlock, hidden_state_BTC, freqs, kvcache):
  # kv cache (ignore for now lol)
  # scaled dot prod attention
  
  # compute qkv
  Q = hidden_state_BTC @ block_params.attention_wq_weight
  K = hidden_state_BTC @ block_params.attention_wk_weight
  V = hidden_state_BTC @ block_params.attention_wv_weight 

  # kv cache optional ig
  

  # rope1D


  # split into heads (GQA)
  key_head_count=8
  query_head_count=32
  Q = rearrange(Q, "B T (Hq Q) -> B T Hq Q", Hq=query_head_count)
  K = rearrange(K, "B T (Hk K) -> B T Hk K", Hk=key_head_count)
  V = rearrange(V, "B T (Hv V) -> B T Hv V", Hv=key_head_count)
  
  # repeat kv to match query (imo, this is a waste, and there should be an op that does gqa attn w/o duping memory)


  # uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  # equivalent to normal attention. no sliding window ig? yaaay
  scale = 1 / jnp.sqrt(Q.shape[-1])
  Q = Q * scale
  attn = Q @ jnp.transpose(K, -1, -2)
  attn = jnp.softmax(attn, axis=-1)
  attn = attn @ V


  # outproject
  out = attn @ block_params.attention_wo_weight

  # reshape
  out = rearrange(out, "B T Hq O -> B T (Hq O)")

  return out


def feed_forward(block_params: TransformerBlock, hidden_state_BTC: jax.Array) -> jax.Array:
  x = hidden_state_BTC
  x1 = jax.nn.silu(x @ block_params.feed_forward_w1_weight)
  x3 = x @ block_params.feed_forward_w3_weight
  x2 = (x1 * x3) @ block_params.feed_forward_w2_weight # element-wise multiplication followed by matmul
  return x2


# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block(block_params: TransformerBlock, hidden_state_BTC: jax.Array, freqs, kvcache) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight, jnp.zeros_like(hidden_state_BTC)) ## attention norm
  residual_BTC = pixtral_attention(residual_BTC, freqs, kvcache)
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
  patch_embeddings_HWO = patches_HWI @ model_params.vision_encoder.patch_conv_weight # may need to reshape conv weight to be collapsed (probably tbh...)
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
  inv_freq = (1.0 / (theta ** jnp.arange(0, dim, 2))) / dim
  position = jnp.arange(0, max_length)
  angles = position[:, jnp.newaxis] * inv_freq[jnp.newaxis, :]
  re, im = jnp.sin(angles), jnp.cos(angles)
  return re, im 


def compute_freqs_2d(max_height, max_width, dim):
  freqs_h = compute_freqs_1d(dim//2, max_height)
  freqs_w = compute_freqs_1d(dim//2, max_width)


def rope1d(hidden_state_TC):
  re, im = compute_freqs_1d(hidden_state_TC.shape[-1], hidden_state_TC.shape[0])
  
  hidden_even = hidden_state_TC[..., ::2]
  hidden_odd = hidden_state_TC[..., 1::2]
  
  hidden_rotated_even = hidden_even * re - hidden_even * im
  hidden_rotated_odd = hidden_odd * re + hidden_odd * im

  hidden_state_2TC = jnp.array([hidden_rotated_even, hidden_rotated_odd])
  hidden_state_TC = rearrange(hidden_state_2TC, "n T C -> T (n C)")
  return hidden_state_TC
  
# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py
def rope2d(hidden_state):
  pass


# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
def vision_encoder(model_params: PixtralModel, processed_image):
  # conv2d on image 
  patch_embeddings_HWC = degen_conv2d(model_params, processed_image)

  # flatten patch embeddings
  flattened_patch_embeddings_PC = jnp.reshape(patch_embeddings_HWC, (-1, patch_embeddings_HWC[-1]))

  # ln pre (RMSnorm)
  flattened_patch_embeddings_PC = RMSnorm(flattened_patch_embeddings_PC, model_params.vision_encoder.ln_pre_weight, jnp.zeros_like(flattened_patch_embeddings_PC))

  # rope
  flattened_patch_embeddings_PC = rope2d(flattened_patch_embeddings_PC)
  
  # vision transformer blocks
  return


def embedding(model_params: PixtralModel, message_tokens, processed_images, image_start_indices) -> jax.Array:
  # gets the embeddings of the tokens
  embeddings = jnp.take_along_axis(model_params.tok_embeddings_weight, message_tokens) # one-hot but faster ig

  # get image embeddings
  # (call vision encoder) 
  image_embeddings = [vision_encoder(processed_image) for processed_image in processed_images]  # N, H, W, C
  
  # replace img tokens with images
  for image_embedding, start_idx in zip(image_embeddings, image_start_indices):
    image_width = image_embedding.shape[1] # H, W
    padded_image_embedding = jnp.concatenate([image_embedding.T, [0]], axis=0)
    flattened_image_embedding = jnp.reshape(padded_image_embedding, (-1,))
    
    overwritten_embeddings = embeddings.at[start_idx:flattened_image_embedding.shape[0]].set(flattened_image_embedding)
    overwrite_mask = jnp.ones_like(overwritten_embeddings).at[start_idx:flattened_image_embedding.shape[0]::image_width+1].set(0)
    embeddings = jnp.where(overwrite_mask, overwritten_embeddings, embeddings) # dont overwrite br and end img tokens

  return embeddings


def layernorm(hidden_state_BTC, weight, bias):
  mean = jnp.mean(hidden_state_BTC)
  std = jnp.std(hidden_state_BTC)
  hidden_state_BTC = (hidden_state_BTC - mean)
  hidden_state_BTC = hidden_state_BTC/std
  hidden_state_BTC = hidden_state_BTC*weight # element-wise
  hidden_state_BTC = hidden_state_BTC + bias # element-wise
  return hidden_state_BTC
  



def mm_forward(model_params: PixtralModel, message_tokens, images) -> jax.Array:
  # get embeddings
  hidden_state_BTC = embedding(message_tokens, images)[jnp.newaxis, None] # fake batch for now

  freqs = compute_freqs_1d(hidden_state_BTC.shape[-1], hidden_state_BTC.shape[1])

  # attention
  # loop through attention layers
  for layer in layers:
    hidden_state = transformer_block(layer, hidden_state, freqs, kvcache)
    # ffw - gelu gated idk
  
  hidden_state_BTC = layernorm(hidden_state_BTC, weight_out, bias_out)
  hidden_state_BTC = hidden_state_BTC @ dense_weight # lm head

  return hidden_state_BTC


#inference :: [Union(Image, string)] -> [string]
# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
def inference(messages) -> str:
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
      
  # get logprobs
  logprobs = mm_forward(tokens, images, image_start_indices)

  # random sample

  # return
  return

