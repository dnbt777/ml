import jax
import jax.numpy as jnp
from einops import rearrange
from PIL import Image
from forward_utils import RMSnorm, feed_forward, rope, rope_channel, vision_model_local_feed_forward, vision_model_global_feed_forward
from llama_types import (
  Text, Tokens, TensorBTC
)
from text_forward import GQA_attention # OPTIMIZATION move to separate file
from llama_types import *

# img -> (h, w) -> (p, p) -> 2D array of patches(patch = 2D array of pixels)
def image_to_tiles(image: Image, tile_resolution) -> jax.Array:
    ## PIL resize image to target_resolution
    width, height = image.size
    tile_height, tile_width = tile_resolution
    aspect_ratio = width / height 
    buffer = 0.25
    if aspect_ratio >= 4 - buffer:
      new_width = 4*tile_width 
      new_height = tile_height
      aspect_ratio_id = 7
    elif aspect_ratio >= 3 - buffer:
      new_width = 3*tile_width 
      new_height = tile_height
      aspect_ratio_id = 5
    elif aspect_ratio >= 2 - buffer:
      new_width = 2*tile_width 
      new_height = tile_height
      aspect_ratio_id = 3
    elif aspect_ratio <= (1 / (2 - buffer)):
      new_width = tile_width 
      new_height = 2*tile_height
      aspect_ratio_id = 2
    elif aspect_ratio <= (1 / (3 - buffer)):
      new_width = tile_width 
      new_height = 3*tile_height
      aspect_ratio_id = 4
    elif aspect_ratio <= (1 / (4 - buffer)):
      new_width = tile_width 
      new_height = 4*tile_height
      aspect_ratio_id = 6
    else:
      # HD
      if width / tile_width >= 2 - 0.25:
          new_width = 2*tile_width
          new_height = 2*tile_height
          aspect_ratio_id = 8
      else: # Not HD 
        new_width = tile_width 
        new_height = tile_height
        aspect_ratio_id = 1
    # 8 would be a 2x2 probably 
    # 0 would be no image
       
    image = image.resize((new_width, new_height))
    image = jnp.array(image, dtype="bfloat16")
    vtiles, htiles = (new_height // tile_height, new_width // tile_width)  # hardcoded tile size for now
    tiles = rearrange(image, "(Th h) (Tw w) C -> Th Tw h w C", Th=vtiles, Tw=htiles)
    tiles = tiles[..., :3] # keep only R, G, B, and no A 
    return tiles, aspect_ratio_id





def embed_tiles(vision_model_params: VisionModel, image_tiles: jax.Array, aspect_ratio_id: int) -> TensorBTC:
  """
    Takes pre-tiled/patched images (B, T, T, P, P, H, W, C) and embeds them
  """
  B, Th, Tw, H, W, RGB_C = image_tiles.shape
  ## Tile pixels -> channel
  # shape patches back into tiles and convolve
  image_tiles = rearrange(image_tiles, "B hT wT H W C -> B (hT wT) C H W")
  image_tiles = rearrange(image_tiles, "B T C H W -> (B T) C H W")
  print("shapes: ", image_tiles.shape, vision_model_params.patch_embedding_weight.shape)
  # (B*T, 3, 224, 224) convolve (1280, 3, 14, 14) => (B*T, 16, 16, 1280)
  patch_resolution = (14, 14)
  imgBTPC = jax.lax.conv_general_dilated(
      image_tiles,
      vision_model_params.patch_embedding_weight,
      patch_resolution,
      padding="valid",
      dimension_numbers=("NCHW", "OIHW", "NCHW")
  ) # ((B, T), C, Ph, Pw) == (B*T, 1280, 16, 16)

  imgBTPC = rearrange(imgBTPC, "(B T) C Ph Pw -> B T (Ph Pw) C", B=B)
  B, T, P, C = imgBTPC.shape

  ## pre tile positional embeddings
  pre_tile_positional_embedding = vision_model_params.pre_tile_positional_embedding_embedding_weight[aspect_ratio_id] # (5120)
  pre_tile_positional_embedding = jnp.reshape(pre_tile_positional_embedding, (1, 4, 1, -1)) # (1, 4, 1, 1280)
  pre_tile_positional_embedding = pre_tile_positional_embedding*vision_model_params.pre_tile_positional_embedding_gate
  pre_tile_positional_embedding = pre_tile_positional_embedding[:, :T, ...]
  
  # the positional embeddings are of shape (9, 5120). this '9' corresponds to the 9 aspect ratio types
  # that tiles can be arranged in. there are 4 max tiles. 1280*4 = 5120.
  # this is a positional embedding for each tile. NOT each patch, but each tile
  imgBTPC = imgBTPC + pre_tile_positional_embedding 

  # add cls token before gated positional embedding
  cls_token = jnp.zeros(shape=(B, T, 1, C), dtype="bfloat16")
  imgBTPC = jax.lax.concatenate([cls_token, imgBTPC], dimension=2) # add class token as first 'patch' for each tile

  ## gated positional embedding
  # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py#L150
  # Patch wise
  gate = vision_model_params.gated_positional_embedding_gate
  patch_embedding = vision_model_params.gated_positional_embedding_embedding 
  gated_patch_embedding = (1 - jnp.tanh(gate)) * patch_embedding # (T*P + 1, C) == (4*256 + 1, 1280)
  gated_patch_embedding = jnp.reshape(gated_patch_embedding, (1, -1, C))
  gated_patch_embedding = gated_patch_embedding[:, :1 + P*T, :] # only add embed to  existing patches
  imgBTPC = imgBTPC + gated_patch_embedding
  # Tile wise
  tile_embedding = vision_model_params.gated_positional_embedding_tile_embedding_weight[aspect_ratio_id] # (5248000)
  gated_tile_embedding = gate * tile_embedding
  max_tiles = 4 # hardcode for now
  gated_tile_embedding = jnp.reshape(gated_tile_embedding, (1, max_tiles, 1+P, C))
  gated_tile_embedding = gated_tile_embedding[:, :T, ...] # only take the tiles needed
  imgBTPC = imgBTPC + gated_tile_embedding 

  return imgBTPC




def unmasked_multihead_attention(attention_layer: AttentionLayer, xBTC: TensorBTC) -> TensorBTC:
    # unmasked multi-head attention
    # ASSUMPTION: no GQA here
    # ASSUMPTION: not transposed (probably incorrect)
    q_weight = attention_layer.self_attn_q_proj_weight
    k_weight = attention_layer.self_attn_k_proj_weight
    v_weight = attention_layer.self_attn_v_proj_weight

    Q = xBTC @ q_weight # B, T, Q
    K = xBTC @ k_weight # B, T, KV
    V = xBTC @ v_weight # B, T, KV

    # multi head
    attention_heads = 32 # hardcoded
    Q = rearrange(Q, "B T (H q) -> B H T q", H=attention_heads)
    Kt = rearrange(K, "B T (H k) -> B H k T", H=attention_heads)
    V = rearrange(V, "B T (H v) -> B H T v", H=attention_heads)

    d = xBTC.shape[-1]/attention_heads # channel size per attention head
    attention_scores = jnp.einsum("BHTc,BHct->BHTt", Q, Kt)/jnp.sqrt(d) # B H T T
    attention_table = jnp.einsum("BHTt,BHTv->BHtv", attention_scores, V)
    Z = attention_table
    
    # rearrange heads at Z and project back to B T C
    Z = rearrange(Z, "B H T v -> B T (H v)") # B T V
    xBTC = Z @ attention_layer.self_attn_o_proj_weight # B T C

    return xBTC



# page 3 https://arxiv.org/pdf/2010.11929
def local_encoder(vision_model_params: VisionModel, image_tiles: jax.Array, aspect_ratio_id: int) -> jax.Array:
    ## embed patches
    imgBTPC = embed_tiles(vision_model_params, image_tiles, aspect_ratio_id)
    B, T, P, C = imgBTPC.shape
    imgBTC = rearrange(imgBTPC, "B T P C -> (B T) P C") # ASSUMPTION: inter-tile attention

    ## LOCAL TRANSFORMER - scan through layers
    def scan_fn(imgBTC, layer_params):
        ### ATTENTION_RESIDUAL
        ## norm -> attn -> reconnect
        attn_residual = RMSnorm(imgBTC, layer_params.input_layernorm_weight, layer_params.input_layernorm_bias) # ASSUMPTION: this is RMSnorm
        attn_residual = unmasked_multihead_attention(layer_params, attn_residual) # unmasked
        imgBTC = imgBTC + attn_residual
        ### POST ATTN RESIDUAL
        ## norm -> MLP -> reconnect
        post_attn_residual = RMSnorm(imgBTC, layer_params.post_attention_layernorm_weight, layer_params.post_attention_layernorm_bias)
        post_attn_residual = vision_model_local_feed_forward(post_attn_residual, layer_params)
        imgBTC = imgBTC + post_attn_residual
        ### send imgBTC to next block and output the layer activation
        layer_activation = imgBTC
        return imgBTC, layer_activation # carry_over, output
    imgBTC, layer_activations = jax.lax.scan(scan_fn, imgBTC, vision_model_params.transformer.layers)
    
    imgBTC = rearrange(imgBTC, "(B T) P C -> B (T P) C", B=B, T=T, P=P)
    
    ## return layers 3, 7, 15, 23, 30
    layers = jnp.array([3, 7, 15, 23, 30]) # mlamma uses 0 indexing, so no + 1 
    selected_layer_activations = layer_activations[layers, ...] # (L, BT, P, C)

    selected_layer_activations = rearrange(selected_layer_activations, "L (B T) P C -> B L T P C", B=B, T=T, P=P)
    return imgBTC, selected_layer_activations




def global_encoder(vision_model_params: VisionModel, key_features) -> jax.Array:
  def scan_fn(xBTC, layer_params):
      ### ATTENTION_RESIDUAL
      ## norm -> attn -> reconnect
      attn_residual = RMSnorm(xBTC, layer_params.input_layernorm_weight, layer_params.input_layernorm_bias) # ASSUMPTION: this is RMSnorm
      attn_residual = unmasked_multihead_attention(layer_params, attn_residual) # unmasked
      xBTC = xBTC + attn_residual
      ### POST ATTN RESIDUAL
      ## norm -> MLP -> reconnect
      post_attn_residual = RMSnorm(xBTC, layer_params.post_attention_layernorm_weight, layer_params.post_attention_layernorm_bias)
      post_attn_residual = vision_model_global_feed_forward(post_attn_residual, layer_params) # replace with fully connected, i think
      xBTC = xBTC + post_attn_residual
      return xBTC, None 
  global_features, _ = jax.lax.scan(scan_fn, key_features, vision_model_params.global_transformer.layers)
  return global_features




# https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf
# page 56 https://arxiv.org/pdf/2407.21783
def vision_processing(vision_model_params: VisionModel, patches: jax.Array, aspect_ratio_id: int) -> TensorBTC:
  ## process through 32 layer local encoder https://arxiv.org/abs/2010.11929
  ## save key intermediate features (layers 3 7 15 23 30
  imgBPC, intermediate_features = local_encoder(vision_model_params, patches, aspect_ratio_id) # (B L T P C)
  B, L, T, P, C = intermediate_features.shape
  
  ## process these through 8 layer global encoder
  imgBPC = RMSnorm(imgBPC, vision_model_params.layernorm_pre_weight, vision_model_params.layernorm_pre_bias)
  
  ## Post tile embeddings
  post_tile_embedding_weight = vision_model_params.post_tile_positional_embedding_embedding_weight[aspect_ratio_id]
  gate = vision_model_params.post_tile_positional_embedding_gate
  post_tile_embedding = post_tile_embedding_weight*gate
  post_tile_embedding = jnp.reshape(post_tile_embedding, (1, 4, 1, C)) # 4 = MAX_TILES
  #imgBPC = jnp.reshape(imgBPC, (B, T, P, C))
  # add padding to imgBPC to make it divisible by 4. then split into 4 tiles. then treat every
  # padding_patch_count = (8 - (imgBPC.shape[-2] % 8)) % 8
  imgBTPC = rearrange(imgBPC, "B (T P) C -> B T P C", T=T)
  imgBTPC = imgBTPC + post_tile_embedding # B T P C + 1 MAX_T 1 C => B T P C

  imgBPC = rearrange(imgBTPC, "B T P C -> B (T P) C")
  global_features = global_encoder(vision_model_params, imgBPC) # B TP C
  global_features = RMSnorm(global_features, vision_model_params.layernorm_post_weight, vision_model_params.layernorm_post_bias)

  ## concatenate layers into 7680 channel size
  global_features = jnp.reshape(global_features, (B, 1, T*P, C)) # single layer. append to the key features layers.
  intermediate_features = rearrange(intermediate_features, "B L T P C -> B L (T P) C", T=T, P=P)
  imgBLPC = jax.lax.concatenate([intermediate_features, global_features], dimension=1) # (B, Layer, TP, C)
  imgBPC = rearrange(imgBLPC, "B L (T P) C -> B (T P) (L C)", L=L+1, T=T, P=P) # L += 1 from concatenation

  return imgBPC 




# same as regular attention BUT K anv V are from vision and Q is from text. I assume Q dim in attention_table is masked.
def cross_attention_layer(layer_params: LangModelCrossAttentionLayer, xBTC: TensorBTC, xBTC_image: TensorBTC, padding_mask: jax.Array) -> TensorBTC:
  # vision => k, v
  # text => q
  B, T, C  = xBTC.shape
  H_Q = 32 # TODO replace the hardcoding
  H_KV = 8 # GQA kv-heads
  
  q_weight = jnp.transpose(layer_params.cross_attn_k_proj_weight)
  k_weight = jnp.transpose(layer_params.cross_attn_k_proj_weight)
  v_weight = jnp.transpose(layer_params.cross_attn_v_proj_weight)
  
  Q = xBTC @ q_weight
  K = xBTC_image @ k_weight
  V = xBTC_image @ v_weight

  # rope?
  Q = rope(Q)
  K = rope(K)

  Q = RMSnorm(Q, layer_params.cross_attn_q_norm_weight)
  K = RMSnorm(K, layer_params.cross_attn_k_norm_weight)

  xBTC_attn_residual = GQA_attention(layer_params, Q, K, V, padding_mask, B, T, H_Q, H_KV)
  
  return xBTC_attn_residual


