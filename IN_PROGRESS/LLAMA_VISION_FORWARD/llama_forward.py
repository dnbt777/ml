import jax
import jax.numpy as jnp
import functools
from forward_utils import RMSnorm, feed_forward, embed_tokens
from load_params import LlamaParams
from llama_types import (
  LlamaParams,
  LogProbsBT, 
)

from text_forward import self_attention_layer
from vision_forward import vision_processing
from vision_forward import cross_attention_layer



@functools.partial(jax.jit, static_argnames=["temp"])
def llama_forward(model_params: LlamaParams, context_tokens, image_patches, temp: float) -> LogProbsBT:
  ### PADDING MASK
  non_padding_tokens = (context_tokens != 128004)
  padding_mask = ~non_padding_tokens
  
  ### TEXT EMBEDDINGS
  xBTC = embed_tokens(model_params.language_model, context_tokens)

  ### VISION ENCODING
  xBTC_image = vision_processing(model_params.vision_model, image_patches) 
    
  ### TRANSFORMER LAYERS
  ## SETUP: TEXT
  def scan_self_attn_layers(xBTC_, layer_params):
    xBTC_attn_residual = self_attention_layer(layer_params, xBTC_, padding_mask)
    xBTC_ = xBTC_ + xBTC_attn_residual 
    # layer norm
    xBTC_postattn_residual = RMSnorm(xBTC_, layer_params.post_attention_layernorm_weight)
    # swiglu
    xBTC_postattn_residual = feed_forward(xBTC_postattn_residual, layer_params)
    xBTC_ = xBTC_ + xBTC_postattn_residual
    return xBTC_, None
  cross_attn_layers = [3, 8, 13, 18, 23, 28, 33, 38] # ASSUMPTION double check these layers
  layers_between_cross_attn_layers = [b - a for a, b in zip(cross_attn_layers, cross_attn_layers[:1])]
  ## RUN SCANS (manually pick layers for now ig)
  self_attn_layer = 0
  for cross_attn_layer_index, layers_between_cross_attn_layers in enumerate(layers_between_cross_attn_layers):
    next_self_attn_layer = self_attn_layer + layers_between_cross_attn_layers
    cross_attn_layer = cross_attn_layers[cross_attn_layer_index]
    xBTC, _ = jax.lax.scan(scan_self_attn_layers, xBTC, model_params.language_model.model.self_attention_layers[self_attn_layer:next_self_attn_layer])
    xBTC = cross_attention_layer(model_params.language_model.model.cross_attention_layers[cross_attn_layer], xBTC, xBTC_image, padding_mask)
    self_attn_layer = next_self_attn_layer
  
  ## OUTPUT
  #layernorm and project to logits
  xBTC = RMSnorm(xBTC, model_params.language_model.model.norm_weight)
  # ffw
  yBTC_logits = xBTC @ jnp.transpose(model_params.language_model.lm_head_weight)
  # logits => logprobs
  yBTC_logprobs = jax.nn.log_softmax(yBTC_logits/(temp + 1e-5), axis=-1)
  return yBTC_logprobs, yBTC_logits


