import jax
import jax.numpy as jnp
import jax.random as jrand
import functools
from llama_forward import text_forward 
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
    temp: float,
    key: jax.Array) -> Token:
  # text -> tokens
  # batch[tokens] -> batch[tokens]  # fake batch of only 1
  print("compiling inference")
  output_logprobs = text_forward(model_params, context_tokens[jnp.newaxis, ...], temp)
  
  # sample last logprob (jax.nn.categorical or whatever it was)
  final_token_logprobs = output_logprobs[0][-1] # (B, T, logprob) => (logprob,)
  final_token_probs = jnp.exp(final_token_logprobs)
  output_token = jrand.categorical(key, final_token_probs)

  # convert tokens to text
  return output_token
