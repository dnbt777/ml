import jax
import jax.numpy as jnp
import jax.random as jrand
from llama_forward import llama_forward 
from vision_forward import image_to_tiles
from llama_types import (
  Text, Tokens, Token, Tokenizer,
  LlamaParams,
)

# TODO make sure all ops are bfloat16 or float16
# TODO for training DEFINITELY make sure all ops are bfloat16
@jax.jit
def inference(
    model_params: LlamaParams,
    context_tokens: Tokens,
    image: jax.Array,
    temp: float,
    key: jax.Array) -> Token:
  # llama_forward takes batches of inputs
  # purpose: faster fine tuning
  # set up inputs 
  image_batch = jnp.array(image, dtype="bfloat16")[jnp.newaxis, ...]
  tile_resolution = (224, 224)
  patch_count = (16, 16)
  image_patches = image_to_tiles(image_batch, tile_resolution, patch_count)
  context_tokens_batch = context_tokens[jnp.newaxis, ...]
  
  # get output
  output_logprobs, output_logits = llama_forward(model_params, context_tokens_batch, image_patches, temp)
  
  # sample last logprob (jax.nn.categorical or whatever it was)
  padding_token = 128004
  final_token_index = 1 + jnp.sum(context_tokens != 128000) - 1
  final_token_logprobs = output_logprobs[0][final_token_index] # (B, T, logprob) => (logprob,)
  final_token_probs = jnp.exp(final_token_logprobs)
  output_token = jrand.categorical(key, final_token_probs)
  
  predicted_tokens = jrand.categorical(key, output_logits/(temp + 1e-7), axis=-1)[0]
  # convert tokens to text
  return output_token, predicted_tokens
