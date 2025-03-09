import json
import jax
import jax.numpy as jnp

from llama_types import Text, Tokens


tokenizer_config = json.load('Llama/tokenizer.json')
vocab = tokenizer_config["vocab"]


# text -> tokens
def encode(text: str) -> Tokens:
  pass





# tokens -> text
def decode(tokens: Tokens) -> Text:
  return [vocab[token] for token in tokens]