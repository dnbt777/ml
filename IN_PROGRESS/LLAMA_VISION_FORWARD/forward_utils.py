import jax.numpy as jnp
import jax
from typing import Union
from llama_types import (
  LangModelSelfAttentionLayer, LangModel, VisionModelLocalLayer,
  VisionModelGlobalLayer,
  TensorBT, TensorBTC,
  AttentionLayer,
)


def layer_norm(x, weight, bias):
  # ASSUMPTION: layer norm is axis=-1
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.mean((x - mean)**2, axis=-1, keepdims=True)
  x = (x - mean) / jnp.sqrt(var + 0.00001) # config.json > norm_eps : 0.00001
  return weight*x + bias

def RMSnorm_bias(x: TensorBTC, weight: jax.Array, bias: jax.Array) -> TensorBTC:
  rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True, dtype="bfloat16") + 1e-6)
  return bias + weight*(x/rms)


def RMSnorm(x: TensorBTC, weight: jax.Array) -> TensorBTC:
  rms = jnp.sqrt(jnp.mean(x*x, axis=-1, keepdims=True, dtype="bfloat16") + 1e-6)
  return weight*(x/rms)

def vision_model_global_feed_forward(
        x: TensorBTC,
        layer_params: VisionModelGlobalLayer, 
        ) -> TensorBTC:
    gate = layer_params.gate_ffn
    up = x @ jnp.transpose(layer_params.mlp_fc1_weight) + layer_params.mlp_fc1_bias
    out = up * jax.nn.silu(gate)
    out = out @ jnp.transpose(layer_params.mlp_fc2_weight) + layer_params.mlp_fc2_bias
    return out


def vision_model_local_feed_forward(
        x: TensorBTC,
        layer_params: VisionModelLocalLayer,
        ) -> TensorBTC:
    x = x @ jnp.transpose(layer_params.mlp_fc1_weight) + layer_params.mlp_fc1_bias
    x = jax.nn.gelu(x) # see config.json for hidden_act
    x = x @ jnp.transpose(layer_params.mlp_fc2_weight) + layer_params.mlp_fc2_bias
    return x


def feed_forward(
        x: TensorBTC,
        layer_params: AttentionLayer 
        ) -> TensorBTC:
    gate = x @ jnp.transpose(layer_params.mlp_gate_proj_weight)
    up = x @ jnp.transpose(layer_params.mlp_up_proj_weight)
    out = up * jax.nn.silu(gate)
    out = out @ jnp.transpose(layer_params.mlp_down_proj_weight)
    return out





def embed_tokens(lang_model_params: LangModel, batch_tokens: TensorBT) -> TensorBTC:
  #vocab_size = lang_model_params.lm_head_weight.shape[0]
  #one_hot_tokens = jax.nn.one_hot(batch_tokens, vocab_size, axis=-1)
  #token_embeddings = one_hot_tokens @ lang_model_params.lm_head_weight
  token_embeddings = jnp.take(lang_model_params.model.embed_tokens, batch_tokens, axis=0)
  #token_embeddings = lang_model_params.model.embed_tokens[batch_tokens]
  return token_embeddings




def rope(channels: TensorBTC) -> TensorBTC:
  """
  channels: tensor of shape (B, T, Q)
  rope(channels) applies rope_channel across each channel at each position T
  example: rotated_channels = rope(Q)
  """
  B, T, Q = channels.shape
  rope_each_channel = jax.vmap(rope_channel, in_axes=(0, 0))
  rope_each_batch = jax.vmap(rope_each_channel, in_axes=(0, None))
  positions = jnp.arange(T)
  rotated_channels = rope_each_batch(channels, positions)
  return rotated_channels




def rope_channel(channel: jax.Array, position: int) -> jax.Array:
  """
  channel: the C in B, T, C.
  frequencies = [th1, th1, th2, th2, ...]
  output: channel, but rotated.
  """
  d = channel.shape[-1]
  i = jnp.arange(d, dtype="bfloat16")
  N = 50_000 # value from llama 3.1 #jnp.repeat(i, 2)
  frequencies = jnp.float_power(N, -2*i/d).astype("bfloat16") # can and should be precomputed. probably will be stored when compiled
  cos = jnp.cos(position*frequencies).astype("bfloat16")
  sin = jnp.sin(position*frequencies).astype("bfloat16")

  even_indices = (jnp.arange(d) % 2 == 0).astype(int)
  signs = jnp.where(even_indices.astype(bool), -jnp.ones(d, dtype="bfloat16"), jnp.ones(d, dtype="bfloat16")).astype("bfloat16")
  indices = jnp.arange(d, dtype=int) + 2*(even_indices)
  channel_rotated = cos*channel + sin*channel[indices]*signs
  
  return channel_rotated

