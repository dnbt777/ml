import jax
import jax.numpy as jnp
import jax.random as jrand
import functools
from einops import rearrange
from PIL import Image

from forward_utils import layer_norm, RMSnorm, feed_forward, rope, rope_channel, embed_tokens

from load_params import LlamaParams
from typing import TypeAlias
from llama_types import (
  LlamaParams, 
  Text, Tokens, TensorBC, TensorBT, TensorBTC, LogProbsBT, 
)
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



def local_vision_attention(local_encoder_params: VisionModelLocalLayer, xBTC: TensorBTC) -> TensorBTC:
    # unmasked multi-head attention
    # ASSUMPTION: no GQA here
    # ASSUMPTION: not transposed (probably incorrect)
    q_weight = local_encoder_params.q
    k_weight = local_encoder_params.k
    v_weight = local_encoder_params.v

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
    xBTC = Z @ local_encoder_params.o # B T C

    return xBTC




# page 3 https://arxiv.org/pdf/2010.11929
def vision_encode(local_encoder_params: VisionModelTransformer, patches) -> jax.Array:
    ## embed patches
    imgBTC = f(patches)

    ## scan through transformer encoder
    def scan_fn(imgBTC, layer_params):
        ### ATTENTION_RESIDUAL
        ## norm -> attn -> reconnect
        attn_residual = RMSnorm(imgBTC) # ASSUMPTION: this is RMSnorm
        attn_residual = local_vision_attention(local_encoder_params, attn_residual) # unmasked
        imgBTC = imgBTC + attn_residual
        ### POST ATTN RESIDUAL
        ## norm -> MLP -> reconnect
        post_attn_residual = RMSnorm(imgBTC)
        post_attn_residual = feed_forward(post_attn_residual, layer_params)
        imgBTC = imgBTC + post_attn_residual
        ### send imgBTC to next block and output the layer activation
        layer_activation = imgBTC
        return imgBTC, layer_activation # carry_over, output
    layer_activations = jax.lax.scan(scan_fn, imgBTC, local_encoder_params.layers)
    
    ## return layers 3, 7, 15, 23, 30
    layers = jnp.array([3, 7, 15, 23, 30])
    selected_layer_activations = layer_activations[layers, ...]
    return selected_layer_activations




def global_vision_encode(global_encoder_params: VisionModelGlobalTransformer, key_features) -> jax.Array:
  ## idk
  return global_features    




# https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf
# page 56 https://arxiv.org/pdf/2407.21783
def vision_processing(vision_model_params: VisionModel, image: jax.Array) -> TensorBTC:
    ## split img into patches
    patches = image_to_patches(image, (224, 224), (32, 32))
    
    ## process through 32 layer local encoder https://arxiv.org/abs/2010.11929
    ## save key intermediate features (layers 3 7 15 23 30
    key_features = vision_encode(vision_model_params.transformer, patches)
    
    ## process these through 8 layer global encoder
    global_features = global_vision_encode(vision_model_params.global_transformer, key_features)

    ## concatenate features into 7680 dimensional representation
    imgBTC = jax.lax.concatenate([key_features, [global_features]], dimension=0)

    ## project image features into the model's semantic space
    imgBTC = imgBTC @ vision_model_params.multi_modal_projector.weight
    imgBTC = imgBTC + vision_model_params.multi_modal_projector.bias

    return imgBTC




def cross_attention_layer(layer_params, xBTC, xBTC_image):
    # vision => k, v
    # text => q
    pass


