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


"""
# https://docs.mistral.ai/guides/tokenization/
# mistral's tokenizer
import json
from mistral_common.tokens.tokenizers.tekken import Tekkenizer 
tok = Tekkenizer.from_file("./pixtral/tekken.json")

def encode(string, add_special=False):
  return tok.encode(string, bos=add_special, eos=add_special)

def decode(ids):
  return tok.decode(ids)
"""


import json
#import re
import regex as re
import base64

def load_tokenizer(config_path: str, special_tokens_config_path: str) -> dict:
    # load tekken config (comes with pixtral)
    with open(config_path, 'rb') as config_file:
        tekken_config = json.load(config_file)
    vocab = tekken_config["vocab"]
    regex_pattern = tekken_config["config"]["pattern"]
    special_token_count = tekken_config["config"]["default_num_special_tokens"]

    # load special tokens config (had to make this. not sure where it is on the internet)
    with open(special_tokens_config_path, 'r') as specials_config_file:
        special_tokens_config = json.load(specials_config_file)
    special_tokens = special_tokens_config["vocab"]

    # ids -> token string
    special_tokens = [(token["rank"], token["token_str"]) for token in special_tokens]
    special_tokens = sorted(special_tokens, key=lambda kv: kv[0])
    special_tokens = [token_str for _, token_str in special_tokens]
    while len(special_tokens) < special_token_count:
        padding = {"rank": len(special_tokens), "token_str": f"<err{len(special_tokens)}>", "is_control": True}
        special_tokens.append(padding)
    assert len(special_tokens) == special_token_count # will fail for now

    # id -> token string
    ids_to_bytes = [( token["rank"],
                      base64.b64decode(token["token_bytes"]),
                    ) for token in vocab]
    ids_to_bytes = sorted(ids_to_bytes, key=lambda kv: kv[0])
    ids_to_bytes = [token_bytes for _, token_bytes in ids_to_bytes]
    ids_to_bytes = special_tokens + ids_to_bytes

    # token bytes -> id
    # does not include specials
    vocab_size = tekken_config["config"]["default_vocab_size"]
    bytes_to_ids = {
        base64.b64decode(token["token_bytes"]): token["rank"] + special_token_count
        for token in vocab if token["rank"] + special_token_count < vocab_size
        # limit size to runtime max
    }

    tokenizer = {
        "ids_to_bytes": ids_to_bytes,
        "bytes_to_ids": bytes_to_ids,
        "special_tokens": special_tokens,
        "regex": re.compile(regex_pattern, re.UNICODE),
        "special_token_count": special_token_count,
        "MAX_ID": special_token_count + vocab_size
    }
    return tokenizer


path = "./pixtral/tekken.json"
specials_path = "./pixtral/special_tokens.json"
tokenizer = load_tokenizer(path, specials_path)


def bpe_encode_bytes(b):
    # rank correlates to token_id
    # rank = token_id - 1000 (specials)
    # this preserves order, though, so we can just use token_id in place of rank
    tokens = [bytes([x]) for x in b]
    while True:
        pairs = {
            (i, tokenizer["bytes_to_ids"][tokens[i] + tokens[i+1]], tokens[i]+tokens[i+1])
            for i in range(len(tokens)-1) if tokens[i] + tokens[i+1] in tokenizer["bytes_to_ids"].keys()
        }
        if not pairs:
            break
        i, _, merged = min(pairs, key=lambda x: x[1])
        tokens[i:i+2] = [merged]
    return [tokenizer["bytes_to_ids"][t] for t in tokens]


def encode(string):
    ids = []
    for piece in tokenizer["regex"].findall(string):
        ids += bpe_encode_bytes(piece.encode("utf-8"))
    return ids


def decode(ids, keep_special=True):
    byte_chunks = []
    for i in ids:
        if i < tokenizer["special_token_count"] and not keep_special:
            continue
        if i >= tokenizer["special_token_count"]:
            byte_chunks.append(tokenizer["ids_to_bytes"][i])
        else:
            byte_chunks.append(tokenizer["special_tokens"][i].encode())
    return b"".join(byte_chunks).decode("utf-8", errors="replace")






#inference :: [Union(Image, string)] -> [string]
# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
def tokenize_messages_dict(messages, add_eos=True):
  # token IDS https://github.com/mistralai/mistral-common/issues/105#issuecomment-2997200779
  BOS = 1 # <s>
  INS_START = 3 # [INS]
  INS_END = 4 # [/INS]
  EOS = 2 # </s>

  tokens = [BOS]
  images = []
  image_start_indices = []
  previous_role = None
  for message in messages:
    if message["role"] == "user":
      if previous_role != "user":
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
      previous_role = "user"
      # https://huggingface.co/mistral-community/pixtral-12b
      # append INS_END at end of user prompt
      # see 'usage example' in the link above
    elif message["role"] == "assistant":
      if previous_role == "user":
          tokens.append(INS_END)
      assert len(message["content"]) == 1 # assistant will only produce 1 response
      tokens = tokens + encode(message["content"][0]["text"])
      if add_eos:
          tokens = tokens + [EOS]
      previous_role = "assistant"
    else:
      raise NameError(f"Error processing messages. Unknown role {message["role"]}")
  if previous_role == "user":
      tokens.append(INS_END)
  return tokens, images, image_start_indices


# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
def tokenize_messages_dict_with_masks(messages, add_eos=True):
  """
  context mask: masks out everything except the assistant's response. used for fine-tuning
  image_mask: masks out image tokens. used for full training
  """
  # token IDS https://github.com/mistralai/mistral-common/issues/105#issuecomment-2997200779
  BOS = 1 # <s>
  INS_START = 3 # [INS]
  INS_END = 4 # [/INS]
  EOS = 2 # </s>

  tokens = []
  images = []
  image_start_indices = []
  image_mask = []
  context_mask = []
  previous_role = None

  # append initial <s>
  tokens.append(BOS)
  context_mask.append((True, 1)) # hide these tokens
  image_mask.append((False, 1)) # dont hide these tokens
  for i, message in enumerate(messages):
    if message["role"] == "user":
      if previous_role != "user":
          tokens.append(INS_START)
          context_mask.append((True, 1)) # hide these tokens
          image_mask.append((False, 1)) # dont hide these tokens
      if type(message["content"]) == str:
        new_tokens = encode(message["content"])
        tokens = tokens + new_tokens
        context_mask.append((True, len(new_tokens)))
        image_mask.append((False, len(new_tokens)))
      else:
        for content in message["content"]:
          if content["type"] == "text":
            new_tokens = encode(content["text"])
            tokens = tokens + new_tokens
            context_mask.append((True, len(new_tokens)))
            image_mask.append((False, len(new_tokens)))
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
              context_mask.append((True, len(image_tokens)))
              image_mask.append((True, len(image_tokens)))
          else:
            raise NameError(f"Error processing messages. Unknown content type {content["type"]}")
      previous_role = "user"
      # https://huggingface.co/mistral-community/pixtral-12b
      # append INS_END at end of user prompt
      # see 'usage example' in the link above
    elif message["role"] == "assistant":
      if previous_role == "user":
        tokens.append(INS_END)
        context_mask.append((True, 1))
        image_mask.append((False, 1))
      assert len(message["content"]) == 1 # assistant should only produce 1 response
      new_tokens = encode(message["content"][0]["text"])
      if add_eos:
          new_tokens = new_tokens + [EOS]
      tokens = tokens + new_tokens
      is_final_response = i == len(messages) - 1
      context_mask.append((not is_final_response, len(new_tokens)))
      image_mask.append((False, len(new_tokens)))
      previous_role = "assistant"
    else:
      raise NameError(f"Error processing messages. Unknown role {message["role"]}")
  if previous_role == "user":
      tokens.append(INS_END)
      context_mask.append((True, 1))
      image_mask.append((False, 1))
  # unpack context_mask and image_masks
  # [(x, 2), (y, 3), (z, 4)] => [x, x, y, y, y, z, z, z, z]
  def unpack_list(packed_list):
    unpacked_list = [[val for _ in range(count)] for val, count in packed_list]
    unpacked_list = [item for sublist in unpacked_list for item in sublist]
    return unpacked_list

  context_mask = unpack_list(context_mask)
  image_mask = unpack_list(image_mask)
    
  return tokens, images, image_start_indices, context_mask, image_mask




def pixtral_attention(block_params: TransformerBlock, hidden_state_BTC, freqs,
                      query_heads, kv_heads, head_dim, attn_mask,
                      block_lora_params=None):      
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
  attn = jnp.where(attn_mask, jnp.bfloat16(-jnp.inf), attn) 
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(jnp.bfloat16) # do attn softmax in float32
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn @ block_params.attention_wo_weight.T
  if block_lora_params:
      out = out + block_lora_params.alpha_o*((attn @ block_lora_params.in_o) @ block_lora_params.out_o)

  return out.astype(jnp.bfloat16)



# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block(
    block_params: TransformerBlock,
    hidden_state_BTC: jax.Array,
    freqs_1d, query_heads, kv_heads, head_dim,
    attn_mask, block_lora_params=None) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
  residual_BTC = pixtral_attention(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim, attn_mask, block_lora_params=block_lora_params)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
  residual_BTC = feed_forward(block_params, residual_BTC)
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC




# probably the biggest bottleneck. needs heavy optimization
# this only runs once for prefill
# in the future, when we want to add prefill on top of an existing kvcache (i.e. user -> assistant -> 2nd user response w images),
# this will need to be optimized
def multimodal_embedding(model_params: PixtralModel, batch_message_tokens, batch_image_sets, image_intext_start_indices_batches) -> jax.Array:
  # gets the embeddings of the tokens
  # already the exact length needed for images. contains img tokens including img_br and img_end
  text_embeddings_batch = text_embedding(model_params, batch_message_tokens) # BTC

  # get image embeddings
  image_embeddings_batch = [vision_encoder(model_params, image_set) for image_set in batch_image_sets]

  # replace img token placeholders with images
  patches_C = image_embeddings_batch[0].shape[-1]
  patch_size = 16 # params.json
  img_break_token_id, img_end_token_id = 1, 2
  img_break_embed = model_params.tok_embeddings_weight[img_break_token_id]
  img_end_embed = model_params.tok_embeddings_weight[img_end_token_id]
  for image_batch in range(len(image_intext_start_indices_batches)):
      inimg_start_idx = 0
      image_intext_start_indices = image_intext_start_indices_batches[image_batch]
      image_embeddings = image_embeddings_batch[image_batch]
      for i, intext_start_idx in enumerate(image_intext_start_indices):
        pixels_C, pixels_H, pixels_W = batch_image_sets[image_batch][i].shape
        patches_H, patches_W = pixels_H//patch_size, pixels_W//patch_size
        inimg_end_idx = inimg_start_idx + patches_H*patches_W # size(unformatted patches) = img patches H * img patches W
        intext_end_idx = intext_start_idx + patches_H*(patches_W + 1) # size(final patches) = img patches + break tokens + end token
        # there are two start indexes we will need. one is for the image inside of the image_embeddings. the other is where the image starts in the text_embeddings.
        image_embedding_TC = image_embeddings[inimg_start_idx:inimg_end_idx, ...]
        image_embedding_HWC = jnp.reshape(image_embedding_TC, (patches_H, patches_W, patches_C))
        # add the embeddings for img_break and img_end
        img_formatting_tokens = jnp.repeat(img_break_embed[None, None, :], patches_H, axis=0) # add img break tokens to each row
        img_formatting_tokens = img_formatting_tokens.at[-1, 0, :].set(img_end_embed) # replace last row's break token with an end token
        formatted_image_embedding_HWC = jnp.concatenate([image_embedding_HWC, img_formatting_tokens], axis=1)
        formatted_image_embedding_TC = jnp.reshape(formatted_image_embedding_HWC, (patches_H*(patches_W+1), patches_C))
    
        text_embeddings_batch = jax.lax.dynamic_update_slice(
          text_embeddings_batch, # dest
          formatted_image_embedding_TC[None, :], # source # bfloat16 just for now
          (image_batch, intext_start_idx, 0) # overwrite index = starting index in text tokens
        )
        inimg_start_idx = inimg_end_idx

  return text_embeddings_batch.astype(jnp.bfloat16)





# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
def vision_encoder(model_params: PixtralModel, processed_images, prefill=False):
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
    hidden_state = transformer_block(block_params, hidden_state, rope_2d_freqs, num_attention_heads, num_attention_heads, head_dim, attn_mask)
    return hidden_state, None
  hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.vision_encoder.vision_encoder_layers)
  hidden_state_TC = hidden_state_BTC[0] # un-batch fake batch

  # vision language adapter
  hidden_state_TC = vision_language_adapter(model_params.vision_language_adapter, hidden_state_TC)
  return hidden_state_TC




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
        precision=jax.lax.Precision.HIGHEST
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
    #hidden_state_pairs_rot = jnp.concatenate([re_rot, im_rot], axis=-1) # (B, T, d//2, 2)
    hidden_state_rot = jnp.zeros_like(hidden_state)
    hidden_state_rot = hidden_state_rot.at[..., ::2].set(re_rot.astype(jnp.bfloat16))
    hidden_state_rot = hidden_state_rot.at[..., 1::2].set(im_rot.astype(jnp.bfloat16)) # [re, im, re, im, re, im, ... ]
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
def text_embedding(model_params: PixtralModel, text_tokens_batch) -> jax.Array:
  text_tokens_batch = jnp.array(text_tokens_batch, dtype=int) # B, T
  embeddings_BTC = jnp.take(model_params.tok_embeddings_weight, text_tokens_batch, axis=0) # C, B, T
  return embeddings_BTC



@jax.jit
def layernorm(hidden_state_BTC, weight, bias):
  mean = jnp.mean(hidden_state_BTC, axis=-1, keepdims=True)
  std = jnp.std(hidden_state_BTC, axis=-1, keepdims=True)
  hidden_state_BTC = (hidden_state_BTC - mean)
  hidden_state_BTC = hidden_state_BTC/std
  hidden_state_BTC = hidden_state_BTC*weight # element-wise
  hidden_state_BTC = hidden_state_BTC + bias # element-wise
  return hidden_state_BTC



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



    
def vision_language_adapter(vla_params: VisionLanguageAdapter, hidden_state_TC):
  hidden_state_TC = jax.nn.gelu(hidden_state_TC @ vla_params.w_in_weight.T + vla_params.w_in_bias)
  hidden_state_TC = hidden_state_TC @ vla_params.w_out_weight.T + vla_params.w_out_bias
  return hidden_state_TC



def get_causal_mask(T: int) -> jax.Array:
    mask = jnp.ones((T, T), dtype=bool)
    mask = jnp.triu(mask, k=1)
    return mask



"""
def text_forward(model_params: PixtralModel, batch_tokens, batch_next_token_indices, batch_attn_mask, lora_params=None):
  hidden_state_BTC = text_embedding(model_params, batch_tokens)
  return forward(model_params, hidden_state_BTC, batch_next_token_indices, batch_attn_mask)



def mm_forward(model_params: PixtralModel, batch_tokens, batch_image_sets, batch_intext_image_start_indices, batch_next_token_indices, batch_attn_mask, lora_params=None):
  hidden_state_BTC = multimodal_embedding(model_params, batch_tokens, batch_image_sets, batch_intext_image_start_indices)
  return forward(model_params, hidden_state_BTC, batch_next_token_indices, batch_attn_mask)



def inference(key, pixtral_params, batch_tokens, batch_image_sets, batch_intext_image_start_indices, batch_next_token_indices, batch_attn_mask, temperature, lora_params=None) -> str:
  if any([len(image_set) > 0 for image_set in batch_image_sets]):
      batch_next_token_logit, kvcache = mm_forward(pixtral_params, batch_tokens, batch_image_sets, batch_intext_image_start_indices, batch_next_token_indices, batch_attn_mask)
  else:
      batch_next_token_logit, kvcache = text_forward(pixtral_params, batch_tokens, batch_next_token_indices, batch_attn_mask)
  batch_next_token = jrand.categorical(key, batch_next_token_logit/max(temperature, 1e-5), axis=-1)
  return batch_next_token, kvcache
"""




