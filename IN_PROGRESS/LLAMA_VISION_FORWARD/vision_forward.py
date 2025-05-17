import jax
import jax.numpy as jnp
from einops import rearrange
from PIL import Image
from forward_utils import (
    RMSnorm, RMSnorm_bias,
    layer_norm, layer_norm_bias,
    feed_forward, rope, rope_channel,
    vision_model_local_feed_forward, vision_model_global_feed_forward
)
from llama_types import (
  Text, Tokens, TensorBTC
)
from text_forward import GQA_attention # OPTIMIZATION move to separate file
from llama_types import *
from typing import Tuple


def aspect_ratio_mask(aspect_ratio_id: int) -> jax.Array:
    aspect_ratios = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)]
    aspect_ratio = aspect_ratios[aspect_ratio_id - 1]
    tw, th = aspect_ratio
    tile_count = tw*th
    max_tiles = 4 # hardcoded
    aspect_ratio_mask = jnp.where(jnp.arange(max_tiles) < tile_count, 1, 0)
    return aspect_ratio_mask


def aspect_ratio_attention_mask(aspect_ratio_id: int) -> jax.Array:
    """
    input: aspect ratio id
    output: the attention mask (jax.numpy array of 1s and 0s)
    1 == identity
    0 == values to hide/mask out
    the attention mask hides the unused tiles' keys during attention
    """
    visible_tiles = aspect_ratio_mask(aspect_ratio_id) # [1, 0, 0, 0] for example
    max_tiles = 4 # hardcoded for now. if this code supported all mllama models this would need to be loaded from a config
    patches_per_tile = 1032 # including CLS and padding
    tile_mask = jnp.ones((max_tiles, patches_per_tile)) # (4, 1032)
    tile_mask = tile_mask * visible_tiles[:, None] # T, P (x) T, 1
    patch_mask = rearrange(tile_mask, "T P -> (T P)")
    # attention_mask = patch_mask[:, None] * patch_mask[None, :] # P, P # this doesn't work. if an entire row of keys is masked, softmax outputs nans
    attention_mask = jnp.tile(patch_mask, (patch_mask.shape[0], 1))
    print("ATTN MASK SHAPE", attention_mask.shape, attention_mask)
    return attention_mask[jnp.newaxis, :, :] # batch, patch (query), patch (key)


def padded_aspect_ratio_attention_mask(aspect_ratio_id: int) -> jax.Array:
    # padded patches = 1032 - 1025
    visible_tiles = aspect_ratio_mask(aspect_ratio_id) # [1, 0, 0, 0] for example
    max_tiles = 4 # hardcoded for now. if this code supported all mllama models this would need to be loaded from a config
    patches_per_tile = 1032 # including CLS and padding
    tile_mask = jnp.ones((max_tiles, patches_per_tile)) # (4, 1032)
    tile_mask = tile_mask * visible_tiles[:, None] # T, P (x) T, 1
    padded_patches = 1032 - 1025 # hardcoded for now
    tile_mask = tile_mask.at[:, -padded_patches:].set(0)
    patch_mask = rearrange(tile_mask, "T P -> (T P)")
    # now we have a 1D mask of the tokens. for the attention mask, we make an OR table out of it.
    # 1 = keep, 0 = mask.
    # 1D mask: 1 1 0 0 0
    # 2D masK:
    #     1 1 0 0 0
    #     ----------
    # 1 | 1 1 1 1 1
    # 1 | 1 1 1 1 1
    # 0 | 1 1 0 0 0
    # 0 | 1 1 0 0 0
    # 0 | 1 1 0 0 0
    patch_mask = patch_mask.astype(bool)
    print("PATCH MASK", patch_mask)
    attention_mask = jnp.logical_or(patch_mask[:, None], patch_mask[None, :])
    return attention_mask


def preprocess_image(image: jax.Array) -> jax.Array:
  image_mean = jnp.array([
    0.48145466,
    0.4578275,
    0.40821073
  ], jnp.float32)
  image_std = jnp.array([
    0.26862954,
    0.26130258,
    0.27577711
  ], jnp.float32)
  rescale_factor = jnp.float32(0.00392156862745098) # 1 / 255.0
  rgb_img = jnp.array(image, dtype=jnp.float32)[..., :3]
  pixel_values = (rgb_img*rescale_factor - image_mean)/image_std
  return pixel_values


# img -> (h, w) -> (p, p) -> 2D array of patches(patch = 2D array of pixels)
def image_to_tiles(image: Image, tile_resolution) -> jax.Array:
    tile_width, tile_height = tile_resolution
    max_tiles = 4 # hardcoded for now
    ## get aspect ratio
    width, height = image.size
    aspect_ratio = int(jnp.ceil(height/min(width, height))), int(jnp.ceil(width/min(width, height))) # convention here is (h, w)
    if aspect_ratio == (1, 1) and (width > tile_width or height > tile_height):
        aspect_ratio = (2, 2)
    
    ## get aspect ratio, scale, and aspect ratio id
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L71
    aspect_ratios = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)] # 9 is 'none'
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L134
    scaling_options = [
        min(
            tile_height*canvas_height/height, # (pixels/tile x tiles) / pixels 
            tile_width*canvas_width/width     # (pixels/tile x tiles) / pixels 
        )
        for canvas_height, canvas_width in aspect_ratios   # convention here is (h, w)
    ] # min scaling needed for each aspect ratio to fill its tile canvas
    # prefer upscaling. pick the smallest option, if there is one.
    upscaling_options = [scale for scale in scaling_options if scale >= 1]
    if len(upscaling_options) > 0:
        scale = min(upscaling_options)
    # otherwise pick the largest downscaling (minimizing changes)
    else:
        downscaling_options = [scale for scale in scaling_options if scale < 1]
        scale = max(scaling_options)
    # pick the canvas size with the smallest area
    smallest_area, best_idx = 100000000, None
    for idx, scaling_option in enumerate(scaling_options):
        if scaling_option != scale:
            continue
        else:
            tiles_h, tiles_w = aspect_ratios[idx]
            tile_width, tile_height = tile_resolution
            area = tiles_h*tile_height + tiles_w*tile_width
            if area < smallest_area:
                best_idx, smallest_area = idx, area
    aspect_ratio = aspect_ratios[best_idx]
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L451
    aspect_ratio_id = aspect_ratios.index(aspect_ratio) + 1
    # https://github.com/huggingface/transformers/blob/e3b70b0d1c15c87ba2010b00830fbd92b2c50252/src/transformers/models/mllama/image_processing_mllama.py#L314
    aspect_ratio_mask = jnp.zeros((4,))
    for idx in range(smallest_area):
        aspect_ratio_mask = aspect_ratio_mask.at[idx].set(1) # leave unused for now. in future optimize 1) for fast loading and 2) so batches have same tile count

    # scale to tiles
    tile_width, tile_height = tile_resolution
    canvas_width, canvas_height = tile_width*aspect_ratio[1], tile_height*aspect_ratio[0]

    new_width, new_height = int(scale*width), int(scale*height)
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L840
    image = image.resize((new_width, new_height), resample=Image.BILINEAR) # PIL.Image.BILINEAR. default is BICUBIC, but thats not whats used here
    
    # create canvas of 0s in the shape of the tiles
    canvas = jnp.zeros((canvas_height, canvas_width, 3), dtype=jnp.float32) # doesnt work with batches rn
    pixel_array = jnp.array(image, dtype=jnp.float32) # go from (w, h) convention (images) to (h, w) convention (linalg arrays)
    print("PIXEL ARRAY SHAPE", pixel_array.shape)
    # get normed pixel values

    # add img to the top left corner
    print("CANVAS SHAPE", canvas.shape)
    pixel_array = pixel_array[..., :3] # RGB only
    canvas = canvas.at[0:new_height, 0:new_width].set(pixel_array)

    # return tiles
    pixel_values = preprocess_image(canvas)
    print("PIXEL VALUES SHAPE", pixel_values.shape)
    tiles = rearrange(pixel_values, "(Th h) (Tw w) C -> (Th Tw) C h w", Th=int(aspect_ratio[0]), Tw=int(aspect_ratio[1]))
    # pad tiles to 4
    tile_count = aspect_ratio[0]*aspect_ratio[1]
    tile_canvas = jnp.zeros((max_tiles, 3, tile_height, tile_width), dtype=jnp.float32)
    tile_canvas = tile_canvas.at[0:tile_count].set(tiles)
    print("ASPECT RATIO", aspect_ratio)
    print("TILE 0", tile_canvas[0])
    print("TILE 1", tile_canvas[1])
    print("TILE 2", tile_canvas[2])
    print("TILE 3", tile_canvas[3])
    
    return tile_canvas.astype("bfloat16"), aspect_ratio_id


def embed_tiles(vision_model_params: VisionModel, image_tiles: jax.Array, aspect_ratio_id: int) -> TensorBTC:
  B, T, RGB_C, H, W,  = image_tiles.shape
  ## Tile pixels -> channel
  print("shapes: ", image_tiles.shape, vision_model_params.patch_embedding_weight.shape)
  # (B*T, 3, 224, 224) convolve (1280, 3, 14, 14) => (B*T, 16, 16, 1280)
  patch_resolution = (14, 14)

  # print("MllamaVisionModel | forward | pixel values:", image_tiles.shape, aspect_ratio_id, image_tiles[0]) # works, no need to print
  image_tiles = rearrange(image_tiles, "B T C H W -> (B T) C H W")
  imgBTPC = jax.lax.conv_general_dilated(
      image_tiles,
      vision_model_params.patch_embedding_weight,
      patch_resolution,
      padding="valid",
      dimension_numbers=("NCHW", "OIHW", "NCHW")
  ) # ((B, T), C, Ph, Pw) == (B*T, 1280, 16, 16)

  imgBTPC = rearrange(imgBTPC, "(B T) C Ph Pw -> B T (Ph Pw) C", B=B)
  B, T, P, C = imgBTPC.shape

  # print("MllamaVisionModel | forward | post conv:", imgBTPC.shape, imgBTPC.dtype, imgBTPC) # DONE

  ## pre tile positional embeddings
  pre_tile_positional_embedding = vision_model_params.pre_tile_positional_embedding_embedding_weight[aspect_ratio_id] # (5120)
  pre_tile_positional_embedding = jnp.reshape(pre_tile_positional_embedding, (1, 4, 1, -1)) # (1, 4, 1, 1280)
  pre_tile_positional_embedding = pre_tile_positional_embedding*jnp.tanh(vision_model_params.pre_tile_positional_embedding_gate)
  pre_tile_positional_embedding = pre_tile_positional_embedding
    
  # the positional embeddings are of shape (9, 5120). this '9' corresponds to the 9 aspect ratio types
  # that tiles can be arranged in. there are 4 max tiles. 1280*4 = 5120.
  # this is a positional embedding for each tile. NOT each patch, but each tile
  imgBTPC = imgBTPC + pre_tile_positional_embedding 

  # print("MllamaVisionModel | forward | post tile embed:", imgBTPC.shape, pre_tile_positional_embedding.dtype, vision_model_params.pre_tile_positional_embedding_gate.dtype, imgBTPC)

  # add cls token before gated positional embedding
  cls_token = vision_model_params.class_embedding
  cls_token = jnp.reshape(cls_token, shape=(1, 1, 1, -1))
  cls_tokens = jnp.repeat(cls_token, T, axis=1) # one cls for each tile. T = max tiles always, since there is tile padding
  imgBTPC = jax.lax.concatenate([cls_tokens, imgBTPC], dimension=2) # add class token as first 'patch' for each tile

  # print("MllamaVisionModel | forward | post cls token:", imgBTPC.shape, imgBTPC)

  ## gated positional embedding
  # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py#L150
  # Patch wise
  gate = vision_model_params.gated_positional_embedding_gate
  patch_embedding = vision_model_params.gated_positional_embedding_embedding 
  gated_patch_embedding = (1 - jnp.tanh(gate)) * patch_embedding # (T*P + 1, C) == (4*256 + 1, 1280)
  gated_patch_embedding = jnp.reshape(gated_patch_embedding, (1, 1, P+1, C)) # P += 1 because of cls token
  # gated_patch_embedding = gated_patch_embedding[:, :T, :, :] # only add embed to  existing patches
  imgBTPC = imgBTPC + gated_patch_embedding
  # Tile wise
  tile_embedding = vision_model_params.gated_positional_embedding_tile_embedding_weight[aspect_ratio_id] # (5248000)
  gated_tile_embedding = jnp.tanh(gate) * tile_embedding
  gated_tile_embedding = jnp.reshape(gated_tile_embedding, (1, T, 1+P, C))  # T = max_tiles = 4
  imgBTPC = imgBTPC + gated_tile_embedding 
  # print("MllamaVisionModel | forward | post pos embed:", imgBTPC.shape, imgBTPC)

  imgBTPC = layer_norm(imgBTPC)*vision_model_params.layernorm_pre_weight + vision_model_params.layernorm_pre_bias

  ## Add padding to make token count a multiple of 8 
  # add padding tokens to the right
  patch_count = imgBTPC.shape[-2]
  padding_patch_count = (8 - (patch_count % 8)) % 8
  padding_configuration = [(0, 0), (0, 0), (0, padding_patch_count), (0, 0)]
  imgBTPC = jnp.pad(imgBTPC, padding_configuration, mode="constant") # mode="constant" means that we are padding using a constant value. namely, 0 (default)

  return imgBTPC



def noncausal_multihead_attention(attention_layer: AttentionLayer, xBTC: TensorBTC, attention_mask: jax.Array) -> TensorBTC:
    # ASSUMPTION: no GQA here
    # ASSUMPTION: not transposed (probably incorrect)
    q_weight = jnp.transpose(attention_layer.self_attn_q_proj_weight)
    k_weight = jnp.transpose(attention_layer.self_attn_k_proj_weight)
    v_weight = jnp.transpose(attention_layer.self_attn_v_proj_weight)

    Q = xBTC @ q_weight # B, T, Q
    K = xBTC @ k_weight # B, T, KV
    V = xBTC @ v_weight # B, T, KV

    # multi head
    attention_heads = 16 # hardcoded from config.json
    Q = rearrange(Q, "B T (H q) -> B H T q", H=attention_heads)
    Kt = rearrange(K, "B T (H k) -> B H k T", H=attention_heads)
    V = rearrange(V, "B T (H v) -> B H T v", H=attention_heads)
    print("SHAPES:", xBTC.shape, Q.shape, Kt.shape, V.shape)

    d = Q.shape[-1] # channel size per attention head
    attention_mask = jnp.where(attention_mask, 0, -jnp.inf).astype("float32") # 1 => identity, 0 => mask out
    attention_scores = jnp.einsum("BHTc,BHct->BHTt", Q, Kt)/jnp.sqrt(d)  # B H T T
    print("SHAPES", attention_scores.shape, attention_mask.shape)
    attention_scores = attention_scores + attention_mask # ignore until batching is implemented and all BTPC have 4 tiles
    attention_table = jax.nn.softmax(attention_scores.astype("float32"), axis=-1).astype("float32")
    attention_table = jnp.einsum("BHTt,BHTv->BHtv", attention_table, V)
    Z = attention_table
    
    # rearrange heads at Z and project back to B T C
    Z = rearrange(Z, "B H T v -> B T (H v)") # B T V

    ## in the official implementation, the outprojection is row parallel linear
    # but I think that has the same numerical result as a normal matmul
    # so for simplicity I am just going to do a normal matmul, since I am running an 11B model on a single GPU
    xBTC = Z @ jnp.transpose(attention_layer.self_attn_o_proj_weight) # B T C 

    print("SHAPES:", xBTC.shape, Z.shape, attention_table.shape, attention_scores.shape)

    return xBTC
    


# page 3 https://arxiv.org/pdf/2010.11929
def local_encoder(vision_model_params: VisionModel, image_tiles: jax.Array, aspect_ratio_id: int) -> jax.Array:
    ## embed tiles
    imgBTPC = embed_tiles(vision_model_params, image_tiles, aspect_ratio_id) # minor thing: move this outside of the local encoder func
    B, T, P, C = imgBTPC.shape
    imgBPC = rearrange(imgBTPC, "B T P C -> B (T P) C")

    print("MllamaVisionModel | forward | start of local encoder:", imgBPC.shape, imgBPC)
    ## LOCAL TRANSFORMER - scan through layers
    # NOT gated
    attention_mask = padded_aspect_ratio_attention_mask(aspect_ratio_id)
    import numpy as np
    np.set_printoptions(threshold=np.inf, linewidth=200) 
    #print("ATTENTION MASK ROW SUM", jnp.sum(attention_mask, axis=0))
    #print("ATTENTION MASK COL SUM", jnp.sum(attention_mask, axis=1))
    stride = 100
    print("ATTENTION MASK: Masked: ", (attention_mask.size - jnp.count_nonzero(attention_mask.astype("int8"))),"/", attention_mask.size, attention_mask.shape, )
    print("ATTENTION MASK:", attention_mask[::stride, ::stride])
    np.set_printoptions(threshold=10, edgeitems=3, linewidth=75)
    def scan_fn(imgBPC, layer_params):
        ### ATTENTION_RESIDUAL
        ## norm -> attn -> reconnect
        attn_residual = imgBPC
        attn_residual = layer_norm(attn_residual)*layer_params.input_layernorm_weight + layer_params.input_layernorm_bias
        #print("MllamaVisionEncoderLayer | post input layernorm state ", attn_residual.shape,layer_params.input_layernorm_weight.shape,layer_params.input_layernorm_bias.shape, attn_residual[0])
        attn_residual = noncausal_multihead_attention(layer_params, attn_residual, attention_mask)
        #print("MllamaVisionEncoderLayer | post attn state ", attn_residual.shape, attn_residual)
        imgBPC = imgBPC + attn_residual
        ### POST ATTN RESIDUAL
        ## norm -> MLP -> reconnect
        post_attn_residual = layer_norm(imgBPC)*layer_params.post_attention_layernorm_weight + layer_params.post_attention_layernorm_bias
        post_attn_residual = vision_model_local_feed_forward(post_attn_residual, layer_params)
        imgBPC = imgBPC + post_attn_residual # NOT gated
        ### send imgBTC to next block and output the layer activation
        layer_activation = imgBPC
        return imgBPC, layer_activation # carry_over, output
    imgBPC, layer_activations = jax.lax.scan(scan_fn, imgBPC, vision_model_params.transformer.layers)
    
    ## return layers 3, 7, 15, 23, 30
    layers = jnp.array([3, 7, 15, 23, 30]) # mlamma uses 0 indexing, so no + 1 
    selected_layer_activations = layer_activations[layers, ...] # (L, BT, P, C)

    selected_layer_activations = rearrange(selected_layer_activations, "L (B T) P C -> B L T P C", B=B, T=T, P=P)
    return imgBPC, selected_layer_activations




def global_encoder(vision_model_params: VisionModel, key_features: jax.Array, aspect_ratio_id: int) -> jax.Array:
  attention_mask = aspect_ratio_attention_mask(aspect_ratio_id)
  def scan_fn(xBTC, layer_params):
      ### ATTENTION_RESIDUAL
      # GATED
      ## norm -> attn -> reconnect
      attn_residual = RMSnorm_bias(xBTC, layer_params.input_layernorm_weight, layer_params.input_layernorm_bias) # ASSUMPTION: this is RMSnorm
      attn_residual = noncausal_multihead_attention(layer_params, attn_residual) # unmasked
      gate = jnp.tanh(layer_params.gate_attn)
      xBTC = xBTC + attn_residual*gate
      ### POST ATTN RESIDUAL
      ## norm -> MLP -> reconnect
      post_attn_residual = RMSnorm_bias(xBTC, layer_params.post_attention_layernorm_weight, layer_params.post_attention_layernorm_bias)
      post_attn_residual = vision_model_global_feed_forward(post_attn_residual, layer_params) # replace with fully connected, i think
      xBTC = xBTC + post_attn_residual*jnp.tanh(layer_params.gate_attn)
      return xBTC, None 
  global_features, _ = jax.lax.scan(scan_fn, key_features, vision_model_params.global_transformer.layers)
  return global_features




# https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf
# page 56 https://arxiv.org/pdf/2407.21783
def vision_processing(vision_model_params: VisionModel, tiles: jax.Array, aspect_ratio_id: int) -> TensorBTC:
  ## process through 32 layer local encoder https://arxiv.org/abs/2010.11929
  ## save key intermediate features (layers 3 7 15 23 30
  imgBPC, intermediate_features = local_encoder(vision_model_params, tiles, aspect_ratio_id) # (B L T P C)
  B, L, T, P, C = intermediate_features.shape
  print("after local encoder", imgBPC.shape, imgBPC)
  
  ## process these through 8 layer global encoder
  imgBPC = RMSnorm_bias(imgBPC, vision_model_params.layernorm_pre_weight, vision_model_params.layernorm_pre_bias)
  print("after RMSnorm", imgBPC.shape)
  
  ## Post tile embeddings
  post_tile_embedding_weight = vision_model_params.post_tile_positional_embedding_embedding_weight[aspect_ratio_id]
  gate = vision_model_params.post_tile_positional_embedding_gate
  post_tile_embedding = post_tile_embedding_weight*gate
  post_tile_embedding = jnp.reshape(post_tile_embedding, (1, 4, 1, C))[:, :T, ...] # 4 = MAX_TILES
  #imgBPC = jnp.reshape(imgBPC, (B, T, P, C))
  # add padding to imgBPC to make it divisible by 4. then split into 4 tiles. then treat every
  # padding_patch_count = (8 - (imgBPC.shape[-2] % 8)) % 8
  imgBTPC = rearrange(imgBPC, "B (T P) C -> B T P C", T=T)
  imgBTPC = imgBTPC + post_tile_embedding # B T P C + 1 MAX_T 1 C => B T P C
  print("post post_tile_embed", imgBTPC.shape)
  

  imgBPC = rearrange(imgBTPC, "B T P C -> B (T P) C")
  print("imgBPC", imgBPC.shape)
  global_features = global_encoder(vision_model_params, imgBPC, aspect_ratio_id) # B TP C
  print("global features", global_features.shape, global_features)
  global_features = RMSnorm_bias(global_features, vision_model_params.layernorm_post_weight, vision_model_params.layernorm_post_bias)
  print("global features post rmsnorm", global_features.shape, global_features)

  ## concatenate layers into 7680 channel size
  global_features = jnp.reshape(global_features, (B, 1, T*P, C)) # single layer. append to the key features layers.
  intermediate_features = rearrange(intermediate_features, "B L T P C -> B L (T P) C", T=T, P=P)
  imgBLPC = jax.lax.concatenate([intermediate_features, global_features], dimension=1) # (B, Layer, TP, C)
  imgBPC = rearrange(imgBLPC, "B L (T P) C -> B (T P) (L C)", L=L+1, T=T, P=P) # L += 1 from concatenation
  print("final imgbpc", imgBPC)

  return imgBPC 




# same as regular attention BUT K anv V are from vision and Q is from text. I assume Q dim in attention_table is masked.
def cross_attention_layer(layer_params: LangModelCrossAttentionLayer, xBTC: TensorBTC, xBTC_image: TensorBTC, padding_mask: jax.Array) -> TensorBTC:
  # vision => k, v
  # text => q
  B, T_self, Cq  = xBTC.shape
  B, T_cross, Ckv = xBTC_image.shape
  H_Q = 32 # TODO replace the hardcoding
  H_KV = 8 # GQA kv-heads
  
  q_weight = jnp.transpose(layer_params.cross_attn_q_proj_weight)
  k_weight = jnp.transpose(layer_params.cross_attn_k_proj_weight)
  v_weight = jnp.transpose(layer_params.cross_attn_v_proj_weight)
  
  Q = xBTC @ q_weight
  K = xBTC_image @ k_weight
  V = xBTC_image @ v_weight

  # rope? not in crossattn
  # Q = rope(Q)
  # K = rope(K)

  ## GQA
  d_Q = Q.shape[-1]//H_Q
  d_KV = K.shape[-1]//H_KV
  Q = jnp.reshape(Q, (B, T_self, H_Q, d_Q)) # (B, T, d_k) => (B, T, H_Q, d_Q)
  Q = jnp.einsum("BTHD->BHTD", Q)
  K = jnp.reshape(K, (B, T_cross, H_KV, d_KV)) # (B, T, d_k) => (B, T, H_KV, d_KV)
  K = jnp.einsum("Bthd->Bhtd", K)
  V = jnp.reshape(V, (B, T_cross, H_KV, d_KV)) # (B, T, d_v) => (B, T, H_KV, d_KV)
  V = jnp.einsum("Bthd->Bhtd", V) # (B, T, H, D) => (B, H, T, D)
  
  # RMS after splitting into GQA heads
  Q = RMSnorm(Q, layer_params.cross_attn_q_norm_weight)
  K = RMSnorm(K, layer_params.cross_attn_k_norm_weight)
  
  G = H_Q // H_KV # groups
  Q = rearrange(Q, "B (h G) T D -> B G h T D", G=G)
  

  # Compute attention scores
  attention_table = jnp.einsum("BGhTD,Bhtd->BGhTt", Q, K) / jnp.sqrt(d_KV) # (B,G,H_Q,T,d_Q) @ (B,H_Q,T,d_kv) => (B, G,H_Q, T, T)
  query_padding_mask = jnp.where(padding_mask, 0, 1) # multiply post-softmax
  query_padding_mask = jnp.reshape(query_padding_mask, (B, 1, 1, T_self, 1)) # do this afterwards.
  attention_scores = jax.nn.softmax(attention_table.astype(jnp.float32), axis=-1).astype("bfloat16") # ASSUMPTION: does not need axis=-1, -2
  attention_scores = attention_scores*query_padding_mask # mask out padding tokens in query. multiplies entire rows by 0.
  Z = jnp.einsum("BGhTt,Bhtd->BGhTd", attention_scores, V)   # (B,H,Tq,Tk) @ (B,H,Tk,d_v//H_KV) => (B, H, Tq, d_v//H_KV)
  # bug: "BHTT,BhTd->BHTd" breaks this. future: don't use the same letter twice. be specific about the dimensions in the operation

  # reshape back into xBTC_residual
  Z = rearrange(Z, "B G h T D -> B (G h) T D") # group the 4x8 heads into 32 heads
  Z = rearrange(Z, "B H T D -> B T H D") # Swap head and time dim
  Z = rearrange(Z, "B T H D -> B T (H D)") # Recombine head and channel dim

  # project BHTd_v back to BHTd
  xBTC_attn_residual = Z @ jnp.transpose(layer_params.cross_attn_o_proj_weight) # (B, T, Z) @ (Z, C) => B, T, C # ASSUMPTION does not need transposition
  return xBTC_attn_residual


