import jax
import jax.numpy as jnp
import functools
from forward_utils import RMSnorm, feed_forward, embed_tokens
from setup_utils import LlamaParams
from llama_types import (
  LlamaParams,
  LogProbsBT, 
)

from text_forward import self_attention_layer
from vision_forward import vision_processing
from vision_forward import cross_attention_layer


@functools.partial(jax.jit, static_argnames=["temp"])
def llama_forward(model_params: LlamaParams, context_tokens, image_tiles, aspect_ratio_id, temp: float) -> LogProbsBT:
  ### PADDING MASK
  non_padding_tokens = (context_tokens != 128004)
  padding_mask = ~non_padding_tokens
  
  ### TEXT EMBEDDINGS
  xBTC = embed_tokens(model_params.language_model, context_tokens)
  #print("starting xBTC: ", xBTC)

  ### VISION ENCODING
  #print("starting image tiles:", image_tiles)
  xBTC_image = vision_processing(model_params.vision_model, image_tiles, aspect_ratio_id)
  print("starting xBTC_image: ", xBTC_image)
    
  ## MULTI MODAL PROJECTOR
  # project image features into the model's semantic space (7198 -> 4096, or whatever)
  xBTC_image = xBTC_image @ jnp.transpose(model_params.multi_modal_projector.weight)
  xBTC_image = xBTC_image + jnp.transpose(model_params.multi_modal_projector.bias)
  print("img embed post multi modal projector:", xBTC_image.shape, xBTC_image)

  ### TRANSFORMER LAYERS
  ## SETUP
  layer_count = 39
  layers = list(range(0, 39+1))
  cross_attn_layers = [3, 8, 13, 18, 23, 28, 33, 38] # ASSUMPTION double check these layers
  self_attn_layers = [n for n in layers if n not in cross_attn_layers]
  print("self attn layers", self_attn_layers)
  ## RUN LAYERS (manually pick layers for now ig)
  cross_attention_layer_idx = 0
  self_attention_layer_idx = 0
  for layer in layers:
    if layer in cross_attn_layers:
        #print(f"doing cross attn layer {cross_attention_layer_idx} (layer {layer})")
        cross_attn_layer_params = jax.tree_util.tree_map(lambda attr: attr[cross_attention_layer_idx], model_params.language_model.model.cross_attention_layers)
        xBTC = cross_attention_layer(cross_attn_layer_params, xBTC, xBTC_image, padding_mask)
        #print(f"layer {layer} xBTC:", xBTC)
        cross_attention_layer_idx += 1
    else:
        #print(f"doing self attn layer {self_attention_layer_idx} (layer {layer})")
        self_attn_layer_params = jax.tree_util.tree_map(lambda attr: attr[self_attention_layer_idx], model_params.language_model.model.self_attention_layers)
        xBTC_attn_residual = self_attention_layer(self_attn_layer_params, xBTC, padding_mask)
        #print(f"residual {layer}:", xBTC_attn_residual)
        xBTC = xBTC + xBTC_attn_residual
        # layer norm
        xBTC_postattn_residual = RMSnorm(xBTC, self_attn_layer_params.post_attention_layernorm_weight)
        #print(f"postattn residual {layer} (postnorm):", xBTC_attn_residual)
        # swiglu
        xBTC_postattn_residual = feed_forward(xBTC_postattn_residual, self_attn_layer_params)
        #print(f"postattn residual {layer} (post ffn):", xBTC_attn_residual)
        #print("xBTC:", xBTC)
        #print("residual:", xBTC_postattn_residual)
        xBTC = xBTC + xBTC_postattn_residual
        #print(f"layer {layer} xBTC:", xBTC)
        self_attention_layer_idx += 1
  # layer norm
  xBTC_postattn_residual = RMSnorm(xBTC, self_attn_layer_params.post_attention_layernorm_weight)
  # swiglu
  xBTC_postattn_residual = feed_forward(xBTC_postattn_residual, self_attn_layer_params)
  xBTC = xBTC + xBTC_postattn_residual
  ## OUTPUT
  #layernorm and project to logits
  xBTC = RMSnorm(xBTC, model_params.language_model.model.norm_weight)
  # ffw
  yBTC_logits = xBTC @ jnp.transpose(model_params.language_model.lm_head_weight)
  # logits => logprobs
  yBTC_logprobs = jax.nn.log_softmax(yBTC_logits/(temp + 1e-5), axis=-1)
  return yBTC_logprobs, yBTC_logits




@functools.partial(jax.jit, static_argnames=["temp"])
def llama_forward_scan(model_params: LlamaParams, context_tokens, image_tiles, aspect_ratio_id, temp: float) -> LogProbsBT:
  ### PADDING MASK
  non_padding_tokens = (context_tokens != 128004)
  padding_mask = ~non_padding_tokens
  
  ### TEXT EMBEDDINGS
  xBTC = embed_tokens(model_params.language_model, context_tokens)

  ### VISION ENCODING
  xBTC_image = vision_processing(model_params.vision_model, image_tiles, aspect_ratio_id) 
    
  ## MULTI MODAL PROJECTOR
  # project image features into the model's semantic space (7198 -> 4096, or whatever)
  xBTC_image = xBTC_image @ jnp.transpose(model_params.multi_modal_projector.weight)
  xBTC_image = xBTC_image + jnp.transpose(model_params.multi_modal_projector.bias)

  ### TRANSFORMER LAYERS
  ## SETUP
  def scan_self_attn_layers(xBTC_, layer_params):
    xBTC_attn_residual = self_attention_layer(layer_params, xBTC_, padding_mask)
    xBTC_ = xBTC_ + xBTC_attn_residual 
    # layer norm
    xBTC_postattn_residual = RMSnorm(xBTC_, layer_params.post_attention_layernorm_weight)
    # swiglu
    xBTC_postattn_residual = feed_forward(xBTC_postattn_residual, layer_params)
    #print("xBTC:", xBTC)
    #print("residual:", xBTC_postattn_residual)
    xBTC_ = xBTC_ + xBTC_postattn_residual
    return xBTC_, None
  layer_count = 39
  cross_attn_layers = [3, 8, 13, 18, 23, 28, 33, 38] # ASSUMPTION double check these layers
  self_attn_layers = [2, 4, 4, 4, 4, 4, 4, 4]
  layers = list(range(1, 39+1))
  ## RUN SCANS (manually pick layers for now ig)
  last_self_attn_layer = 0
  for idx, self_attn_layers in enumerate(self_attn_layers):
    self_attn_layers_params = jax.tree_util.tree_map(lambda attr: attr[last_self_attn_layer:last_self_attn_layer+self_attn_layers+1], model_params.language_model.model.self_attention_layers)
    print("Scanning through", (last_self_attn_layer, last_self_attn_layer+self_attn_layers), "out of", model_params.language_model.model.self_attention_layers.input_layernorm_weight.shape)
    xBTC, _ = jax.lax.scan(scan_self_attn_layers, xBTC, self_attn_layers_params)
    cross_attn_layer_params = jax.tree_util.tree_map(lambda attr: attr[idx], model_params.language_model.model.cross_attention_layers)
    xBTC = cross_attention_layer(cross_attn_layer_params, xBTC, xBTC_image, padding_mask)
    last_self_attn_layer = last_self_attn_layer+self_attn_layers+1 # +1 for the crossattn layer
  ## do final 39th layer
  self_attn_layer_params = jax.tree_util.tree_map(lambda attr: attr[last_self_attn_layer+1], model_params.language_model.model.self_attention_layers)
  xBTC_attn_residual = self_attention_layer(self_attn_layer_params, xBTC, padding_mask)
  xBTC = xBTC + xBTC_attn_residual 
  # layer norm
  xBTC_postattn_residual = RMSnorm(xBTC, self_attn_layer_params.post_attention_layernorm_weight)
  # swiglu
  xBTC_postattn_residual = feed_forward(xBTC_postattn_residual, self_attn_layer_params)
  xBTC = xBTC + xBTC_postattn_residual

  ## OUTPUT
  #layernorm and project to logits
  xBTC = RMSnorm(xBTC, model_params.language_model.model.norm_weight)
  # ffw
  yBTC_logits = xBTC @ jnp.transpose(model_params.language_model.lm_head_weight)
  # logits => logprobs
  yBTC_logprobs = jax.nn.log_softmax(yBTC_logits/(temp + 1e-5), axis=-1)
  return yBTC_logprobs, yBTC_logits
