import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from model_types import *
from einops import rearrange

# mm_preprocessor :: [Image] -> [image_tiles or whatever]
from PIL import Image
import cv2


# debug
import tbug.tbug as tbug

debug = False
if not debug:
    donothing = lambda *args, **kwargs: None
    tbug.capture = donothing
    #print = donothing


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
  image = jnp.array(image, dtype=jnp.float32) # explicitly convert to jnp array w dtype float32. when x64 is enabled, implicit => float64 and breaks this
  image = jnp.transpose(image, (2, 0, 1))

  ## get tokens

  # create tokens
  img_token_id = 10 # from params.json
  img_break_id = 12 # from params.json
  img_end_id = 13 # from params.json
  image_tokens = [[img_token_id for _ in range(width_tokens)] + [img_break_id] for _ in range(height_tokens)]
  image_tokens = [item for sublist in image_tokens for item in sublist] # flatten img token list. 2d->1d. also wtf python weird ass syntax
  image_tokens[-1] = img_end_id # replace final row's break token with img end token

  return (image_tokens, image.astype(jnp.bfloat16))



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
tok = Tekkenizer.from_file("../pixtral/tekken.json")

def encode(string, add_special=True):
  return tok.encode(string, bos=add_special, eos=add_special)

def decode(string):
  return tok.decode(string)



def pixtral_attention(block_params: TransformerBlock, hidden_state_BTC, freqs, query_heads, kv_heads, head_dim, capturenum=None):
  # scaled dot prod attention
  # compute qkv
  hidden_state_BTC = hidden_state_BTC.astype(jnp.bfloat16)
  Q = hidden_state_BTC @ block_params.attention_wq_weight.T
  K = hidden_state_BTC @ block_params.attention_wk_weight.T
  V = hidden_state_BTC @ block_params.attention_wv_weight.T

  # kv cache optional ig


  # split into heads (GQA)
  Hk=kv_heads
  Hq=query_heads
  repeats = Hq // Hk 
  Q = rearrange(Q, "B T (Hk r d) -> B Hk r T d", Hk=Hk, r=repeats, d=head_dim) # not sure if repeats is needed in pixtral
  K = rearrange(K, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)
  V = rearrange(V, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)

  if not (capturenum is None):
      tbug.capture(rearrange(Q, "B Hk r T d -> B r T Hk d")[0][0], f"vision_encoder.layer{capturenum:02d}_attn_Q", project="jax")

  
  # rope1D AFTER splitting into GQA heads
  Q = apply_rope(Q, freqs).astype(jnp.float32)
  K = apply_rope(K, freqs).astype(jnp.float32) # both have the same head dim

  if not (capturenum is None):
      tbug.capture(rearrange(Q, "B Hk r T d -> B r T Hk d")[0][0], f"vision_encoder.layer{capturenum:02d}_attn_ropedQ", project="jax")

  K = K[:, :, None, :, :] # broadcast over r
  V = V[:, :, None, :, :] # broadcast over r

  # repeat kv to match query (imo, this is a waste, and there should be an op that does gqa attn w/o duping memory)

  # uses xformers memory efficient attention
  # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
  # equivalent to normal attention. no sliding window ig? yaaay
  #scale = jax.lax.rsqrt(Q.shape[-1]) # rqrt does not accept int64
  scale = 1.0 / jnp.sqrt(Q.shape[-1])
  Q = jnp.float32(Q * scale)
  attn = Q @ jnp.swapaxes(K, -1, -2)
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1) # kernel does this
  attn = attn @ V

  # collapse heads and outproject
  attn = rearrange(attn, "B H r T d -> B T (H r d)")
  out = attn.astype(jnp.bfloat16) @ block_params.attention_wo_weight.T

  return out.astype(jnp.bfloat16)


def feed_forward(block_params: TransformerBlock, hidden_state_BTC: jax.Array) -> jax.Array:
  x = hidden_state_BTC.astype(jnp.bfloat16)
  x1 = jax.nn.silu(x @ block_params.feed_forward_w1_weight.T)
  x3 = x @ block_params.feed_forward_w3_weight.T
  x2 = (x1 * x3) @ block_params.feed_forward_w2_weight.T # element-wise multiplication followed by matmul
  return x2


# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block(block_params: TransformerBlock, hidden_state_BTC: jax.Array, freqs_1d, query_heads, kv_heads, head_dim, capturenum=None) -> jax.Array:
  # same block for vision encoder AND transformer
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight, jnp.zeros_like(hidden_state_BTC)) ## attention norm
  if not (capturenum is None):
      tbug.capture(residual_BTC[0], f"vision_encoder.layer{capturenum:02d}_postrms1", project="jax")
      residual_BTC = pixtral_attention(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim, capturenum=capturenum)
  else:
      residual_BTC = pixtral_attention(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim)
  if not (capturenum is None):
      tbug.capture(residual_BTC[0], f"vision_encoder.layer{capturenum:02d}_postattn", project="jax")
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight, jnp.zeros_like(hidden_state_BTC)) ## ff norm
  if not (capturenum is None):
      tbug.capture(residual_BTC[0], f"vision_encoder.layer{capturenum:02d}_postrms2", project="jax")
  residual_BTC = feed_forward(block_params, residual_BTC)
  if not (capturenum is None):
      tbug.capture(residual_BTC[0], f"vision_encoder.layer{capturenum:02d}_postffout", project="jax")
  hidden_state_BTC = hidden_state_BTC + residual_BTC
  return hidden_state_BTC


def conv2d(model_params: PixtralModel, image_CHW) -> jax.Array:
    patch_size = 16
    H, W = image_CHW.shape[1:]
    h, w = H//patch_size, W//patch_size
    patch_embeddings_HWO = jax.lax.conv_general_dilated(
        image_CHW[jnp.newaxis, ...],
        model_params.vision_encoder.patch_conv_weight,
        (patch_size, patch_size),
        'SAME',
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT
    )
    return patch_embeddings_HWO.astype(jnp.bfloat16)


def RMSnorm(hidden_state, weight, bias):
  eps = 1e-5
  hidden_state = hidden_state.astype(jnp.float32)
  weight = weight.astype(jnp.float32)
  bias = bias.astype(jnp.float32)
  squared = jax.lax.pow(hidden_state, 2)
  mean = (jnp.mean(squared, axis=-1, keepdims=True) + jnp.float64(eps)).astype(jnp.float32) # float, double -> ?
  rsqrt = jax.lax.rsqrt(mean.astype(jnp.float32)) # float, float -> ?
  hidden_state = jnp.multiply(hidden_state, rsqrt) # float, float -> 
  hidden_state = hidden_state.astype(jnp.bfloat16)
  hidden_state = jnp.multiply(hidden_state, weight) # bfloat16, bfloat16 -> ?
  # hidden_state = hidden_state + bias # not actually used
  return hidden_state.astype(jnp.bfloat16)


def precompute_rope_freqs_1d(max_i, d):
    rope_theta = 1_000_000_000 # params.json
    max_j = d//2
    i, j = jnp.arange(max_i), jnp.arange(max_j)
    freqs_ij = jnp.outer(i, rope_theta**(-2.0*j/d)) # i, d//2
    cos, sin = jnp.cos(freqs_ij)[None, :], jnp.sin(freqs_ij)[None, :] # precompute these so we dont have to write jnp.cos(... over and over again
    return jax.lax.complex(cos, sin).astype(jnp.complex64)


def apply_rope(hidden_state, freqs):
    original_shape = hidden_state.shape # ..., T, C
    T, C = hidden_state.shape[-2:]
    d = C
    hidden_state_pairs = jnp.reshape(hidden_state, (*hidden_state.shape[:-2], T, d//2, 2))
    re, im = hidden_state_pairs[..., 0], hidden_state_pairs[..., 1] # b, i, d//2
    # rotate each pair by its corresponding theta:
    # rotating a vector <x,y> by theta == multiplying (x + iy) by e^(i*theta)
        # where i = sqrt(-1), and x = re(al), and y = im(aginary) components
    # e^(i*theta) = cos(theta) + i*sin(theta)
    # so (x + i*y) * e^(i*theta) =
    # = (x + i*y) * (cos(theta) + i*sin(theta))
    # = x*cos(theta) + i*y*cos(theta) + x*i*sin(theta) + i*y*i*sin(theta)
    # = x*cos(theta) - y*sin(theta) + i*(y*cos(theta) + x*sin(theta))
    # = <x*cos(theta) - y*sin(theta), y*cos(theta) + x*sin(theta)>
    cos, sin = jnp.real(freqs).astype(jnp.float32), jnp.imag(freqs).astype(jnp.float32)
    re_rot = re*cos - im*sin
    im_rot = im*cos + re*sin
    hidden_state_pairs_rot = jnp.concatenate([re_rot, im_rot], axis=-1) # (B, T, d//2, 2)
    hidden_state_rot = jnp.zeros_like(hidden_state)
    hidden_state_rot = hidden_state_rot.at[..., ::2].set(re_rot)
    hidden_state_rot = hidden_state_rot.at[..., 1::2].set(im_rot) # [re, im, re, im, re, im, ... ]
    return hidden_state_rot


def precompute_rope_freqs_2d(max_h, max_w, d):
    rope_theta = 10_000 # params.json
    max_j = d//2
    h, w, j = jnp.arange(max_h), jnp.arange(max_w), jnp.arange(0, d, 2, dtype=jnp.float64) # mimics -2.0 * (j from 0 to d//2)
    aten_div = j/jnp.float64(d) # float32, (long int => Scalar => float32)
    #aten_div.astype(jnp.float32)
    # anytime you see 'scalar' in pytorch profiling, it gets auto-converted to a dtype.
    # do precise ops in float64, then cast back to float32
    aten_pow = jnp.float64(rope_theta)**jnp.float64(aten_div) # (Scalar => float32), float32
    aten_pow = aten_pow.astype(jnp.float32)
    base_freqs = jnp.reciprocal(aten_pow.astype(jnp.float64)).astype(jnp.float32) # float32
    thetas_hj = jnp.outer(jnp.float64(h), base_freqs[::2]).astype(jnp.float32) # H, d//2
    thetas_hj = thetas_hj[:, jnp.newaxis, :] # H, d//2 => H, 1, d//2
    thetas_wj = jnp.outer(jnp.float64(w), base_freqs[1::2]).astype(jnp.float32) # W, d//2
    thetas_wj = thetas_wj[jnp.newaxis, :, :] # 1, W, d//2
    
    # calculate hpos based rotations
    # pixtral uses aten::polar, which is just 1.0*cos(theta), 1.0*sin(theta)
    cos_h, sin_h = jnp.cos(thetas_hj.astype(jnp.float64))[jnp.newaxis, :].astype(jnp.float64), jnp.sin(thetas_hj.astype(jnp.float64))[jnp.newaxis, :].astype(jnp.float64) # add batch dim to both
    freqs_h = jax.lax.complex(cos_h, sin_h).astype(jnp.complex64) # 1, H, 1, d//4 # take the evens
    cos_w, sin_w = jnp.cos(thetas_wj.astype(jnp.float64))[jnp.newaxis, :].astype(jnp.float64), jnp.sin(thetas_wj.astype(jnp.float64))[jnp.newaxis, :].astype(jnp.float64)
    freqs_w = jax.lax.complex(cos_w, sin_w).astype(jnp.complex64) # 1, 1, W, d//4 # take the odds

    # make frequency grid
    freqs_w = jnp.repeat(freqs_w, repeats=max_h, axis=1) # jnp.repeat(freqs_w, (1, max_h, 1, 1))
    freqs_h = jnp.repeat(freqs_h, repeats=max_w, axis=2) # jnp.repeat(freqs_h, (1, 1, max_w, 1)) # make a grid
    freqs_2d = jnp.concatenate([freqs_h, freqs_w], axis=-1) # 1, H, W, d//2
    #freqs_2d = jnp.reshape(freqs_2d, (1, max_h, max_w, d//2))
    return freqs_2d.astype(jnp.complex64) # explicit


"""is this used??
def rope2d(hidden_state_BTC, freqs_2d):
    rope_theta = 10_000 # params.json
    B, T, C = hidden_state_BTC.shape
    d = C
    _, H, W, _ = freqs_2d.shape
    hidden_state_pairs = jnp.reshape(hidden_state_BTC, (B, T, d//2, 2))
    re, im = hidden_state_pairs[..., 0], hidden_state_pairs[..., 1] # b, H, W, d//2
    # calculate rotations
        # could just be done with one mul (of both complex nums) but whatever
    cos, sin = jnp.real(freqs_2d).astype(jnp.float32), jnp.imag(freqs_2d).astype(jnp.float32) # 1, H, W, d//2
    # flatten cos and sin grids
    cos, sin = jnp.reshape(cos, (1, H*W, d//2)), jnp.reshape(sin, (1, H*W, d//2))
    # do rotations
    re_rot = re*cos - im*sin
    im_rot = im*cos + re*sin
    
    hidden_state_rot2d = jnp.concatenate([re_rot[..., None], im_rot[..., None]], axis=-1)
    hidden_state_rot2d = jnp.reshape(hidden_state_rot2d, (B, T, C)) # ..., d//2, 2 => ..., d 
    return hidden_state_rot2d
"""


def vision_language_adapter(vla_params: VisionLanguageAdapter, hidden_state_TC):
  hidden_state_TC = jax.nn.gelu(hidden_state_TC @ vla_params.w_in_weight.T + vla_params.w_in_bias)
  hidden_state_TC = hidden_state_TC @ vla_params.w_out_weight.T + vla_params.w_out_bias
  return hidden_state_TC



# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
def vision_encoder(model_params: PixtralModel, processed_image):
  tbug.capture(processed_image, name="vision_enc.processed_image", project="jax")
  # conv2d on image 
  patch_embeddings_CHW = conv2d(model_params, processed_image)[0]
  tbug.capture(patch_embeddings_CHW, name="vision_enc.post_conv.first_image", project="jax")
  
  C, H, W = patch_embeddings_CHW.shape
  #rope2d_positions = position_meshgrid(patch_embeddings_CHW.shape[1], patch_embeddings_CHW.shape[2])
  num_attention_heads = 16 # params.json
  hidden_dim = C
  head_dim = hidden_dim // num_attention_heads
  freqs_2d = precompute_rope_freqs_2d(H, W, head_dim) # in the future - precompute with max H and W ONCE, and adjust func to deal with any H W array
  tbug.capture(freqs_2d[0], name="vision_encoder.rope_freqs2d", project="jax") # capture pre-flattened (like mistral)
  freqs_2d = rearrange(freqs_2d, "B H W c -> B (H W) c") # flatten
  
    
  # flatten patch embeddings
  # (C, H, W) => (P, C)
  flattened_patch_embeddings_PC = rearrange(patch_embeddings_CHW, "C H W -> (H W) C")

  # ln pre (RMSnorm)
  flattened_patch_embeddings_PC = RMSnorm(flattened_patch_embeddings_PC, model_params.vision_encoder.ln_pre_weight, jnp.zeros_like(flattened_patch_embeddings_PC))
  tbug.capture(flattened_patch_embeddings_PC, name="vision_enc.post_ln_pre.patch_embeds", project="jax")

  # block diagonal mask
  # see vision_encoder.py
  # in a text transformer, keys for future tokens are masked out
  # here, we want to construct a mask that masks out other images' keys
  # so that attention doesn't leak between images
  # for a single image, there is no mask
  # for now we will use placeholders and a single image
  patch_sequences = [flattened_patch_embeddings_PC.shape[0]] # for image in images - placeholder for now that only uses the first image
  block_diagonal_mask = jnp.zeros((sum(patch_sequences),sum(patch_sequences)), dtype=jnp.float32)
  left_idx = 0
  for patch_sequence in patch_sequences:
      block_diagonal_mask.at[left_idx:patch_sequence, left_idx:patch_sequence].set(1.0)
      left_idx = left_idx + patch_sequence
  tbug.capture(block_diagonal_mask, name="vision_encoder.blockdiagonalmask", project="jax")
    
  # vision transformer blocks
  # loop through attention layers
  hidden_state_BTC = flattened_patch_embeddings_PC[jnp.newaxis, ...]
  _, T, C = hidden_state_BTC.shape
  num_attention_heads = 16 # params.json
  hidden_dim = hidden_state_BTC.shape[-1]
  head_dim = hidden_dim // num_attention_heads
  #freqs_1d = precompute_rope_freqs_1d(T, head_dim) # in pixtral, rope is done after viewing q and k broken into heads
  #tbug.capture(freqs_1d[0], name="vision_encoder.rope_freqs1d", project="jax") # this is wrong!!! 1d is for the text transformer. do rope1d with the 2d freqs
  def scanf(data, block_params):
    hidden_state, i = data
    if i < 3:
        tbug.capture(hidden_state[0], f"vision_transformer.layer{i:02d}_input", project="jax")
        hidden_state = transformer_block(block_params, hidden_state, freqs_2d, num_attention_heads, num_attention_heads, head_dim, capturenum=i)
    else:
        hidden_state = transformer_block(block_params, hidden_state, freqs_2d, num_attention_heads, num_attention_heads, head_dim)
    i = i + 1
    data = hidden_state, i
    return data, None
  data_in = hidden_state_BTC, 0
  data_out, _ = jax.lax.scan(scanf, data_in, model_params.vision_encoder.vision_encoder_layers)
  hidden_state_BTC, _ = data_out
  hidden_state_TC = hidden_state_BTC[0] # un-batch fake batch
  tbug.capture(hidden_state_TC, name="post_vision_encoder.hidden_state", project="jax")

  # call vision language adapter
  hidden_state_TC = vision_language_adapter(model_params.vision_language_adapter, hidden_state_TC)
  tbug.capture(hidden_state_TC, name="post_vl_adapter.hidden_state", project="jax")
  
  return hidden_state_TC, H, W, hidden_state_TC.shape[-1]


def embedding(model_params: PixtralModel, message_tokens, processed_images, image_start_indices) -> jax.Array:
  # gets the embeddings of the tokens
  # already the exact length needed for images. contains img tokens including img_br and img_end
  message_tokens = jnp.array(message_tokens, dtype=int)
  embeddings = jnp.take(model_params.tok_embeddings_weight, message_tokens, axis=0) # one-hot but faster ig
  tbug.capture(embeddings, name="combined_embeddings.text_features", project="jax")
  
  # get image embeddings
  # (call vision encoder) 
  image_embeddings = [vision_encoder(model_params, processed_image) for processed_image in processed_images]  # N, H, W, C
  tbug.capture(image_embeddings[0][0], name="combined_embeddings.image_features", project="jax")
  
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
  tbug.capture(embeddings, name="combined_embeddings.hidden_state", project="jax")
  return embeddings


def layernorm(hidden_state_BTC, weight, bias):
  mean = jnp.mean(hidden_state_BTC, axis=-1, keepdims=True)
  std = jnp.std(hidden_state_BTC, axis=-1, keepdims=True)
  hidden_state_BTC = (hidden_state_BTC - mean)
  hidden_state_BTC = hidden_state_BTC/std
  hidden_state_BTC = hidden_state_BTC*weight # element-wise
  hidden_state_BTC = hidden_state_BTC + bias # element-wise
  return hidden_state_BTC
  


@jax.jit
def mm_forward(model_params: PixtralModel, message_tokens, processed_images, image_start_indices) -> jax.Array:
  # get embeddings
  hidden_state_BTC = embedding(model_params, message_tokens, processed_images, image_start_indices)[jnp.newaxis, :] # fake batch for now
  tbug.capture(hidden_state_BTC, name="post_embedding.hidden_state", project="jax")

  B, T, C = hidden_state_BTC.shape
  # in GQA there are more queries than keys/values
  # therefore the channel is larger
  # so the max dim is .. wait..
  head_dim = 128 # params.json
  max_pos, d = T, head_dim
  freqs = precompute_rope_freqs_1d(max_pos, d) # mistral does rope after splitting k and q into gqa heads. q and k are split into the same channel size per head
  tbug.capture(freqs, name="post_embedding.freqs1d", project="jax")

  # attention
  # loop through attention layers
  Hq = 32
  Hk = 8
  # head dim defined above - it's used to calculate rope1d frequencies
  def scanf(hidden_state, block_params):
    hidden_state = transformer_block(block_params, hidden_state, freqs, Hq, Hk, head_dim).astype(jnp.bfloat16)#, kvcache)
    return hidden_state, None
  
  hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.transformer.transformer_layers)
    # ffw - gelu gated idk
  tbug.capture(hidden_state_BTC[:, -1, :], name="post_lang_transformer.hidden_state", project="jax")
  
  hidden_state_BTC = layernorm(hidden_state_BTC, model_params.norm_weight, jnp.zeros_like(hidden_state_BTC))
  tbug.capture(hidden_state_BTC[:, -1, :], name="post_lang_transformer_ln.hidden_state", project="jax")
  hidden_state_BTC = hidden_state_BTC @ model_params.output_weight.T # lm head
  return hidden_state_BTC


#inference :: [Union(Image, string)] -> [string]
# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
import jax.random as jrand
from functools import partial

def tokenize_messages_dict(messages, add_special=False):
  # token IDS
  START = -1 # <s> # placeholder values for now
  INS_START = 0 # [INS]
  INS_END = 1 # [/INS]

  tokens = []
  #tokens = [START]
  images = []
  image_start_indices = []
  for message in messages:
    if message["role"] == "user":
      #tokens.append(INS_START)
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
      #tokens.append(INS_END)
    elif message["role"] == "assistant": # assistant?
      tokens = tokens + encode(message["content"])
    else:
      raise NameError(f"Error processing messages. Unknown role {message["role"]}")
    
  return tokens, images, image_start_indices
    



def inference(key, pixtral_params, tokens, images, image_start_indices) -> str:
  # get logits
  next_token_logits = mm_forward(pixtral_params, tokens, images, image_start_indices)[:, -1, :] # (B, T, vocab) => (B, 1, vocab)
  tbug.capture(next_token_logits, name="post_forward.logits", project="jax")
  # random sample
  next_token = jrand.categorical(key, next_token_logits, axis=-1) # (B,)
  # return
  return next_token # (B,)


