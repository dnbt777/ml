import json
import jax
import jax.numpy as jnp

from llama_types import Text, Tokens

BPE_SPACE_CHAR = 'Ġ'

# load config
path = 'Llama/tokenizer.json'
with open(path, 'r') as config:
  tokenizer_config = json.load(config)

vocab = tokenizer_config["model"]["vocab"]
merges = tokenizer_config["model"]["merges"]


# text -> tokens
def encode(text: str) -> Tokens:
  # prep subwords. replace space with Gspace
  subwords = text
  subwords = subwords.replace(" ", BPE_SPACE_CHAR)
  subwords = list(subwords)
  # do merge loop until no more merges are possible
  i = 0
  while 1:
    if i == len(subwords) - 1:
        break # done
    attempted_merge = f"{subwords[i]} {subwords[i+1]}"
    if attempted_merge in merges:
      subwords[i] = attempted_merge.replace(" ", "")
      subwords.pop(i+1)
      # if merge found, restart
      i = 0
      continue
    else:
      i += 1
  # convert to tokens and return
  encoded = [vocab[subword] for subword in subwords]
  return encoded


# tokens -> text
def decode(tokens: Tokens) -> Text:
  vocab_list = list(vocab.keys())
  return ("".join([vocab_list[token] for token in tokens])).replace(BPE_SPACE_CHAR, " ")
