import jax
import jax.numpy as jnp
from einops import rearrange
from PIL import Image
from forward_utils import RMSnorm, feed_forward, rope, rope_channel 
from llama_types import (
  Text, Tokens, TensorBTC
)
from text_forward import GQA_attention # OPTIMIZATION move to separate file
from llama_types import *

# img -> (h, w) -> (p, p) -> 2D array of patches(patch = 2D array of pixels)
def image_to_patches(image: Image, target_resolution, patches) -> jax.Array:
    ## PIL resize image to target_resolution
    image = image.resize(target_resolution)
    ## break into patches using jnp.reshape
    # input shape: (H, W, 3) = (h*p, w*p, 3) # no batching for now. just vmap this if needed
    # target shape: (h, w, p, p, 3)
    image = jnp.array(image)
    patches = rearrange(image, "(h P) (w p) C -> h w P p C", P=patches[0], p=patches[1])
    return patches



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
    attention_scores = jnp.dot(Q, Kt)/jnp.sqrt(d) # B H T T
    attention_table = jnp.einsum("BHTt,BHTv->BHtv", attention_scores, V)
    Z = attention_table
    
    # rearrange heads at Z and project back to B T C
    Z = rearrange(Z, "B H T v -> B T (H v)") # B T V
    xBTC = Z @ attention_layer.self_attn_o_proj_weight # B T C

    return xBTC



def embed_patches(vision_model_params: VisionModel, patches: jax.Array) -> TensorBTC:
  # TODO I believe this should be conv based. inspect the shapes of patch_embedding_weight.
  patches_flattened = rearrange(patches, "B P p h w c -> B (P p) h w c") # (1, 32, 32, 7, 7, 3) => (1, 1024, 147)
  # just do the convolusion
  jax.scipy.signal.convolve2d
  imgBTC = jnp.einsum(
     "BPHWC,OHWC->BPO",
     patches_flattened, vision_model_params.patch_embedding_weight) # (B P C) => B T C = (1, 1024, 1280)
  # TODO convert this to use conv2D 
  
  ### gated positional embedding w cls token
  B, T, C = imgBTC.shape
  # add CLS
  cls_token = jnp.zeros(shape=(B, 1, C), dtype="bfloat16")
  imgBTC = jax.lax.concatenate([[cls_token], imgBTC], dimension=1)
  # add positional embeddings
  gated_positional_embeddings = vision_model_params.post_tile_positional_embedding_embedding_weight * vision_model_params.post_tile_positional_embedding_gate
  imgBTC = imgBTC + gated_positional_embeddings 
  # gate
  return imgBTC



# page 3 https://arxiv.org/pdf/2010.11929
def local_encoder(vision_model_params: VisionModel, patches: jax.Array) -> jax.Array:
    ## embed patches
    imgBTC = embed_patches(vision_model_params.transformer, patches)

    ## LOCAL TRANSFORMER - scan through layers
    def scan_fn(imgBTC, layer_params):
        ### ATTENTION_RESIDUAL
        ## norm -> attn -> reconnect
        attn_residual = RMSnorm(imgBTC, layer_params.input_layernorm_weight, layer_params.input_layernorm_bias) # ASSUMPTION: this is RMSnorm
        attn_residual = unmasked_multihead_attention(vision_model_params.transformer, attn_residual) # unmasked
        imgBTC = imgBTC + attn_residual
        ### POST ATTN RESIDUAL
        ## norm -> MLP -> reconnect
        post_attn_residual = RMSnorm(imgBTC, layer_params.post_attn_layernorm_weight, layer_params.post_attn_layernorm_bias)
        post_attn_residual = feed_forward(post_attn_residual, layer_params)
        imgBTC = imgBTC + post_attn_residual
        ### send imgBTC to next block and output the layer activation
        layer_activation = imgBTC
        return imgBTC, layer_activation # carry_over, output
    _, layer_activations = jax.lax.scan(scan_fn, imgBTC, vision_model_params.transformer.layers)
    
    ## return layers 3, 7, 15, 23, 30
    layers = jnp.array([3, 7, 15, 23, 30]) + 1 # wait should this be +1 for indexes?
    selected_layer_activations = layer_activations[layers, ...]
    
    # not gated..?
    return selected_layer_activations




def global_encoder(vision_model_params: VisionModel, key_features) -> jax.Array:
  featureBTC = rearrange(key_features, "L B T C -> B (L T) C") # ASSUMPTION make sure its not B (T L) C 

  def scan_fn(xBTC, layer_params):
      ### ATTENTION_RESIDUAL
      ## norm -> attn -> reconnect
      attn_residual = RMSnorm(xBTC, layer_params.input_layernorm_weight, layer_params.input_layernorm_bias) # ASSUMPTION: this is RMSnorm
      attn_residual = unmasked_multihead_attention(vision_model_params.global_transformer, attn_residual) # unmasked
      xBTC = xBTC + attn_residual
      ### POST ATTN RESIDUAL
      ## norm -> MLP -> reconnect
      post_attn_residual = RMSnorm(xBTC, layer_params.post_attn_layernorm_weight, layer_params.post_attn_layernorm_bias)
      post_attn_residual = feed_forward(post_attn_residual, layer_params) # replace with fully connected, i think
      xBTC = xBTC + post_attn_residual
      return xBTC, None 
  global_features, _ = jax.lax.scan(scan_fn, featureBTC, vision_model_params.global_transformer.layers)

  # take CLS?
  # return global_features[:, 0]
  return global_features




# https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf
# page 56 https://arxiv.org/pdf/2407.21783
def vision_processing(vision_model_params: VisionModel, image: jax.Array) -> TensorBTC:
    ## split img into patches
    patches = image_to_patches(image, (224, 224), (16, 16)) # outputs 14x14 patches
    
    ## process through 32 layer local encoder https://arxiv.org/abs/2010.11929
    ## save key intermediate features (layers 3 7 15 23 30
    key_features = local_encoder(vision_model_params.transformer, patches)
    
    ## process these through 8 layer global encoder
    # pre global norm?
    key_features = RMSnorm(key_features, vision_model_params.layernorm_pre_weight, vision_model_params.layernorm_pre_bias)
    global_features = global_encoder(vision_model_params.global_transformer, key_features)
    global_features = RMSnorm(global_features, vision_model_params.layernorm_post_weight, vision_model_params.layernorm_post_bias)
    # post global layernorm?

    ## concatenate features into 7680 dimensional representation
    imgBTC = jax.lax.concatenate([key_features, [global_features]], dimension=1)

    ## project image features into the model's semantic space (7198 -> 4096, or whatever)
    imgBTC = imgBTC @ vision_model_params.multi_modal_projector.weight
    imgBTC = imgBTC + vision_model_params.multi_modal_projector.bias

    return imgBTC




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


