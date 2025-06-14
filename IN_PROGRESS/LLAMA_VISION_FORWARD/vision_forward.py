import jax
import jax.numpy as jnp
from einops import rearrange
from PIL import Image
from forward_utils import (
    RMSnorm, RMSnorm_bias,
    layer_norm,
    feed_forward, rope, rope_channel,
    vision_model_local_feed_forward, vision_model_global_feed_forward
)
from llama_types import (
  Text, Tokens, TensorBTC
)
from text_forward import GQA_attention # OPTIMIZATION move to separate file
from llama_types import *
from typing import Tuple

import tdiff

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



#jax.config.update("jax_default_matmul_precision", "tensorfloat32")


def unfold_convolve(image_tiles, kernel, patch_resolution):
    B, T, C, H, W = image_tiles.shape
    hp, wp = patch_resolution
    # flatten kernel
    flattened_kernel = rearrange(kernel, "O C H W -> (C H W) O") # vllm collapses this down
    tdiff.capture(flattened_kernel.T, name="pre_convolve.patch_embedding_weight", project="jax")
    # collapse image tiles
    image_tiles = image_tiles.astype(jnp.bfloat16)
    image_tiles = rearrange(image_tiles, "B T C (h hp) (w wp) -> (B T) (h w) (C hp wp)", hp=hp, wp=wp, h=H//hp, w=W//wp)
    tdiff.capture(image_tiles, name="pre_convolve.prematmul_tiles", project="jax")
    # matmul
    x = jnp.matmul(
        image_tiles,
        flattened_kernel
    )#.astype(jnp.bfloat16)
    tdiff.capture(x, name="post_convolve_linear.hidden_state", project="jax")
    #x = jnp.einsum("BNP,OP->BNO", x, kernel_flat,
    #               precision=jax.lax.Precision.DEFAULT,
    #               preferred_element_type=jnp.float32,
    #              ) # => (B T) (h w) (1280)
    # uncollapse
    x = rearrange(
        x, "(B T) (h w) O -> B T (h w) O",
        B=B, T=T, h=H//hp, w=W//wp)
    return x
    


def embed_tiles(vision_model_params: VisionModel, image_tiles: jax.Array, aspect_ratio_id: int) -> TensorBTC:
  B, T, RGB_C, H, W,  = image_tiles.shape
  ## Tile pixels -> channel
  print("shapes: ", image_tiles.shape, vision_model_params.patch_embedding_weight.shape)
  # (B*T, 3, 224, 224) convolve (1280, 3, 14, 14) => (B*T, 16, 16, 1280)
  patch_resolution = (14, 14)
  tdiff.capture(image_tiles[0], name="pre_convolve.pixel_values", project="jax")
  imgBTPC = unfold_convolve(image_tiles, vision_model_params.patch_embedding_weight, patch_resolution)
  #imgBTPC = imgBTPC.astype("bfloat16")
  tdiff.capture(imgBTPC[0], name="post_convolve.hidden_state", project="jax")

  # capture these vals in vllm. see if that helps. but then after just keep going
  # and then yeah see abt minimizing error. when it becomes too much, then fix it
  B, T, P, C = imgBTPC.shape

  ## pre tile positional embeddings
  pre_tile_positional_embedding = vision_model_params.pre_tile_positional_embedding_embedding_weight[aspect_ratio_id] # (5120)
  pre_tile_positional_embedding = jnp.reshape(pre_tile_positional_embedding, (1, 4, 1, -1)) # (1, 4, 1, 1280)
  pre_tile_positional_embedding = pre_tile_positional_embedding*jnp.tanh(vision_model_params.pre_tile_positional_embedding_gate)
  pre_tile_positional_embedding = pre_tile_positional_embedding
    
  # the positional embeddings are of shape (9, 5120). this '9' corresponds to the 9 aspect ratio types
  # that tiles can be arranged in. there are 4 max tiles. 1280*4 = 5120.
  # this is a positional embedding for each tile. NOT each patch, but each tile
  imgBTPC = imgBTPC + pre_tile_positional_embedding 
  tdiff.capture(imgBTPC, name="post_tile_pos_embed.hidden_state", project="jax")

  # add cls token before gated positional embedding
  cls_token = vision_model_params.class_embedding
  cls_token = jnp.reshape(cls_token, shape=(1, 1, 1, -1))
  cls_tokens = jnp.repeat(cls_token, T, axis=1) # one cls for each tile. T = max tiles always, since there is tile padding
  imgBTPC = jax.lax.concatenate([cls_tokens, imgBTPC], dimension=2) # add class token as first 'patch' for each tile
  tdiff.capture(imgBTPC[0], name="post_cls_token.hidden_state", project="jax")

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
  tdiff.capture(imgBTPC, name="post_gated_pos_embed.hidden_state", project="jax")

  imgBTPC = layer_norm(jnp.float32(imgBTPC), jnp.float32(vision_model_params.layernorm_pre_weight), jnp.float32(vision_model_params.layernorm_pre_bias))
  imgBTPC = jnp.bfloat16(imgBTPC)
  tdiff.capture(imgBTPC, name="post_layernorm.hidden_state", project="jax")

  ## Add padding to make token count a multiple of 8 
  # add padding tokens to the right
  patch_count = imgBTPC.shape[-2]
  padding_patch_count = (8 - (patch_count % 8)) % 8
  padding_configuration = [(0, 0), (0, 0), (0, padding_patch_count), (0, 0)]
  imgBTPC = jnp.pad(imgBTPC, padding_configuration, mode="constant") # mode="constant" means that we are padding using a constant value. namely, 0 (default)

  return imgBTPC



def noncausal_multihead_attention(attention_layer: AttentionLayer, xBTC: TensorBTC, attention_mask: jax.Array) -> TensorBTC:
    #print("ATTENTION | INPUT | ", xBTC.shape, xBTC.dtype, jnp.sum(xBTC[0]), jnp.sum(xBTC[0], axis=-1), jnp.sum(xBTC[0], axis=-2))
    # ASSUMPTION: no GQA here
    # ASSUMPTION: not transposed (probably incorrect)
    q_weight = jnp.transpose(attention_layer.self_attn_q_proj_weight)
    k_weight = jnp.transpose(attention_layer.self_attn_k_proj_weight)
    v_weight = jnp.transpose(attention_layer.self_attn_v_proj_weight)

    Q = (xBTC @ q_weight) # B, T, Q
    K = (xBTC @ k_weight) # B, T, KV
    V = (xBTC @ v_weight) # B, T, KV

    #print("QKV", jnp.sum(Q), jnp.sum(K), jnp.sum(V))

    # multi head
    attention_heads = 16 # hardcoded from config.json
    Q = rearrange(Q, "B T (H q) -> B H T q", H=attention_heads).astype(jnp.float32)
    Kt = rearrange(K, "B T (H k) -> B H k T", H=attention_heads).astype(jnp.float32)
    V = rearrange(V, "B T (H v) -> B H T v", H=attention_heads).astype(jnp.float32)
    #print("SHAPES:", xBTC.shape, Q.shape, Kt.shape, V.shape, Q.dtype)
    #print("K FIRST HEAD", K.shape, K.dtype, K[0][0][0])

    scale = jnp.sqrt(jnp.float32(Q.shape[-1]))
    # https://github.com/pytorch/pytorch/blob/51c6c5e156c64d84ff0cd06a559fa6786c96f128/aten/src/ATen/native/transformers/attention.cpp#L768
    # in the above func, Q is passed in having already been rearranged into heads. so the channel dim (-1) is the head channel dim
    
    # scale = scale.astype(Q.dtype) # https://github.com/pytorch/pytorch/blob/dcb3edd30dfe8e42aeb33e8551eee1f2c54441bd/torch/onnx/symbolic_opset14.py#L240
    Q = Q / scale
    
    attention_scores = jnp.einsum("BHTc,BHct->BHTt", Q, Kt)  # B H T T
    attention_scores = jnp.where(attention_mask[jnp.newaxis, jnp.newaxis, ...], attention_scores, jnp.finfo(jnp.float64).min) # 1 => identity, 0 => mask out
    #attention_scores = attention_scores + attention_mask[jnp.newaxis, jnp.newaxis, ...] # ignore until batching is implemented and all BTPC have 4 tiles
    #print("SCORES DTYPE", attention_scores.dtype)
    attention_table = jax.nn.softmax(attention_scores, axis=-1) # TODO there is a precision problem here

    # rearrange heads at Z and project back to B T C
    Z = jnp.einsum("BHqk,BHkv->BHqv", attention_table, V)
    Z = Z.astype(jnp.bfloat16)
    #print("attn output first row", Z.dtype, Z[0][0][0])
    Z = rearrange(Z, "B H T v -> B T (H v)") # B T V

    ## in the official implementation, the outprojection is row parallel linear
    # but I think that has the same numerical result as a normal matmul
    # so for simplicity I am just going to do a normal matmul, since I am just running an 11B model on a single GPU
    #print("pre-o", Z.dtype, Z[0][0])

    o_weight = jnp.transpose(attention_layer.self_attn_o_proj_weight)
    xBTC = Z @ o_weight # B T C 
    return xBTC



def _noncausal_multihead_attention(attention_layer: AttentionLayer, xBTC: TensorBTC, attention_mask: jax.Array) -> TensorBTC:
    # ASSUMPTION: no GQA here
    # ASSUMPTION: not transposed (probably incorrect)
    q_weight = jnp.transpose(attention_layer.self_attn_q_proj_weight)
    k_weight = jnp.transpose(attention_layer.self_attn_k_proj_weight)
    v_weight = jnp.transpose(attention_layer.self_attn_v_proj_weight)

    # q_weight, k_weight, v_weight = q_weight.astype(jnp.float64), k_weight.astype(jnp.float64), v_weight.astype(jnp.float64)
    #xBTC = xBTC.astype(jnp.float32)

    Q = (xBTC @ q_weight) # B, T, Q
    K = (xBTC @ k_weight) # B, T, KV
    V = (xBTC @ v_weight) # B, T, KV

    attention_heads = 16 # hardcoded from config.json
    Q = rearrange(Q, "B T (N H) -> B T N H", N=attention_heads)
    K = rearrange(K, "B T (N H) -> B T N H", N=attention_heads)
    V = rearrange(V, "B T (N H) -> B T N H", N=attention_heads)
    #write_array_to_txt(Q, "outputs/jax_q.txt")
    #write_array_to_txt(K, "outputs/jax_k.txt")
    #write_array_to_txt(V, "outputs/jax_v.txt")
    attention_mask = attention_mask[jnp.newaxis, ...] # Turn into batch
    #write_array_to_txt(attention_mask, "outputs/jax_attention_mask.txt")
    print("SHAPES:", xBTC.shape, Q.shape, K.shape, V.shape, Q.dtype, attention_mask.shape, attention_mask.dtype)
    print("K FIRST HEAD", K.shape, K.dtype, K[0][0][0])
    

    Z = jax.nn.dot_product_attention(Q, K, V, mask=attention_mask, is_causal=False, implementation="xla")
    print(Z.shape)
    Z = rearrange(Z, "B T N H -> B T (N H)") # B T V
    
    print("pre-o", Z.dtype, Z.shape, Z[0][0])
    
    xBTC = Z @ jnp.transpose(attention_layer.self_attn_o_proj_weight) # B T C 

    return xBTC


# page 3 https://arxiv.org/pdf/2010.11929
def local_encoder(vision_model_params: VisionModel, imgBTPC: jax.Array, aspect_ratio_id: int) -> jax.Array:
    B, T, P, C = imgBTPC.shape
    imgBPC = rearrange(imgBTPC, "B T P C -> B (T P) C")
    tdiff.capture(imgBPC[0], name="pre_local_encoder.hidden_state", project="jax")
    ## LOCAL TRANSFORMER - scan through layers
    # NOT gated
    attention_mask = padded_aspect_ratio_attention_mask(aspect_ratio_id)
    tdiff.capture(attention_mask, name="pre_local_layers.attention_mask", project="jax")
    def scan_fn(imgBPC, layer_params):
        ### ATTENTION_RESIDUAL
        ## norm -> attn -> reconnect
        attn_residual = imgBPC
        attn_residual = layer_norm(jnp.float32(attn_residual), jnp.float32(layer_params.input_layernorm_weight), jnp.float32(layer_params.input_layernorm_bias))
        attn_residual = jnp.bfloat16(attn_residual)
        #print("MllamaVisionEncoderLayer | post input layernorm state ", attn_residual.shape,layer_params.input_layernorm_weight.shape,layer_params.input_layernorm_bias.shape, attn_residual[0])
        attn_residual = noncausal_multihead_attention(layer_params, attn_residual, attention_mask)
        #print("MllamaVisionEncoderLayer | post attn state ", attn_residual.dtype, attn_residual.shape, attn_residual)
        imgBPC = imgBPC + attn_residual
        ### POST ATTN RESIDUAL
        ## norm -> MLP -> reconnect
        post_attn_residual = layer_norm(jnp.float32(imgBPC), jnp.float32(layer_params.post_attention_layernorm_weight), jnp.float32(layer_params.post_attention_layernorm_bias))
        post_attn_residual = jnp.bfloat16(post_attn_residual)
        #print("MllamaVisionEncoderLayer | postattn residual 1 ", post_attn_residual.dtype, post_attn_residual.shape, post_attn_residual[0][0])
        post_attn_residual = vision_model_local_feed_forward(post_attn_residual, layer_params)
        #print("MllamaVisionEncoderLayer | postattn residual postffw ", post_attn_residual.dtype, post_attn_residual.shape, post_attn_residual[0][0])
        imgBPC = imgBPC + post_attn_residual # NOT gated
        ### send imgBTC to next block and output the layer activation
        layer_activation = imgBPC
        return imgBPC, layer_activation # carry_over, output
    imgBPC, layer_activations = jax.lax.scan(scan_fn, imgBPC, vision_model_params.transformer.layers)
    tdiff.capture(imgBPC, name="post_local_layers.hidden_state", project="jax")
    
    ## return layers 3, 7, 15, 23, 30
    layers = jnp.array([3, 7, 15, 23, 30]) # mlamma uses 0 indexing, so no + 1 
    selected_layer_activations = layer_activations[layers, ...] # (L, BT, P, C)

    #selected_layer_activations = rearrange(selected_layer_activations, "L B (T P) C -> B L T P C", B=B, T=T, P=P)
    return imgBPC, selected_layer_activations




def global_encoder(vision_model_params: VisionModel, key_features: jax.Array, aspect_ratio_id: int) -> jax.Array:
  attention_mask = padded_aspect_ratio_attention_mask(aspect_ratio_id)
  def scan_fn(xBTC, layer_params):
      ### ATTENTION_RESIDUAL
      # GATED
      ## norm -> attn -> reconnect
      attn_residual = layer_norm(
          xBTC,
          layer_params.input_layernorm_weight,
          layer_params.input_layernorm_bias
      ) # ASSUMPTION: this is RMSnorm
      attn_residual = noncausal_multihead_attention(layer_params, attn_residual, attention_mask) # not causal masked, but it is masked tile and padding-wise
      gate = jnp.tanh(layer_params.gate_attn)
      xBTC = xBTC + attn_residual*gate
      ### POST ATTN RESIDUAL
      ## norm -> MLP -> reconnect
      post_attn_residual = layer_norm(
          xBTC,
          layer_params.post_attention_layernorm_weight,
          layer_params.post_attention_layernorm_bias
      )
      post_attn_residual = vision_model_global_feed_forward(post_attn_residual, layer_params) # replace with fully connected, i think
      xBTC = xBTC + post_attn_residual*jnp.tanh(layer_params.gate_attn)
      return xBTC, None 
  global_features, _ = jax.lax.scan(scan_fn, key_features, vision_model_params.global_transformer.layers)
  return global_features




# https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf
# page 56 https://arxiv.org/pdf/2407.21783
def vision_processing(vision_model_params: VisionModel, image_tiles: jax.Array, aspect_ratio_id: int) -> TensorBTC:
  imgBTPC = embed_tiles(vision_model_params, image_tiles, aspect_ratio_id) # minor thing: move this outside of the local encoder func
  B, T, P, C = imgBTPC.shape
  ## process through 32 layer local encoder https://arxiv.org/abs/2010.11929
  ## save key intermediate features (layers 3 7 15 23 30
  imgBPC, intermediate_features = local_encoder(vision_model_params, imgBTPC, aspect_ratio_id) # (B L T P C)
  tdiff.capture(imgBPC, name="post_local_encoder.hidden_state", project="jax")
  intermediate_features = rearrange(intermediate_features, "L B P C -> B L P C")
  B, L, P, C = intermediate_features.shape
  
  ## process these through 8 layer global encoder
  imgBPC = RMSnorm_bias(imgBPC, vision_model_params.layernorm_pre_weight, vision_model_params.layernorm_pre_bias)
  print("after RMSnorm", imgBPC.shape)
  
  ## Post tile embeddings
  post_tile_embedding_weight = vision_model_params.post_tile_positional_embedding_embedding_weight[aspect_ratio_id]
  gate = vision_model_params.post_tile_positional_embedding_gate
  post_tile_embedding = post_tile_embedding_weight*gate
  post_tile_embedding = jnp.reshape(post_tile_embedding, (1, 4, 1, C)) # 4 = MAX_TILES
  #imgBPC = jnp.reshape(imgBPC, (B, T, P, C))
  # add padding to imgBPC to make it divisible by 4. then split into 4 tiles. then treat every
  # padding_patch_count = (8 - (imgBPC.shape[-2] % 8)) % 8
  max_tiles = 4
  imgBTPC = rearrange(imgBPC, "B (T P) C -> B T P C", T=max_tiles) # always 4.
  imgBTPC = imgBTPC + post_tile_embedding # B T P C + 1 MAX_T 1 C => B T P C
  tdiff.capture(imgBTPC, name="post_tile_embed.hidden_state", project="jax")
  

  imgBPC = rearrange(imgBTPC, "B T P C -> B (T P) C")
  tdiff.capture(imgBPC, name="pre_global_encoder.hidden_state", project="jax")
  global_features = global_encoder(vision_model_params, imgBPC, aspect_ratio_id) # B TP C
  tdiff.capture(global_features, name="post_global_encoder.hidden_state", project="jax")
  global_features = RMSnorm_bias(global_features, vision_model_params.layernorm_post_weight, vision_model_params.layernorm_post_bias)
  tdiff.capture(global_features, name="post_global_encoder_and_rms.hidden_state", project="jax")

  ## concatenate layers into 7680 channel size
  B, TP, C = global_features.shape
  global_features = jnp.reshape(global_features, (B, 1, TP, C)) # single layer. append to the key features layers.
  #intermediate_features = rearrange(intermediate_features, "B L T P C -> B L (T P) C", T=T, P=P)
  imgBLPC = jax.lax.concatenate([intermediate_features, global_features], dimension=1) # (B, Layer, TP, C)
  imgBTC = rearrange(imgBLPC, "B L P C -> B P (L C)")
  tdiff.capture(imgBTC, name="vision_processing_output.hidden_state", project="jax")

  return imgBTC 




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

