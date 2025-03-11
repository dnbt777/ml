import json
from llama_types import Text, Tokens, Tokenizer

BPE_SPACE_CHAR = 'Ġ'


def load_tokenizer(path):
  with open(path, 'r') as config:
    tokenizer_config = json.load(config)
  vocab = tokenizer_config["model"]["vocab"]
  merges = tokenizer_config["model"]["merges"]
  return Tokenizer(
    vocab=vocab,
    merges=merges
  )


# text -> tokens
def encode(tokenizer: Tokenizer, text: str) -> Tokens:
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
    if attempted_merge in tokenizer.merges:
      subwords[i] = attempted_merge.replace(" ", "")
      subwords.pop(i+1)
      # if merge found, restart
      i = 0
      continue
    else:
      i += 1
  # convert to tokens and return
  encoded = [tokenizer.vocab[subword] for subword in subwords]
  return encoded


# tokens -> text
def decode(tokenizer: Tokenizer, tokens: Tokens) -> Text:
  vocab_list = list(tokenizer.vocab.keys())
  if len(tokens) == 1:
      return vocab_list[int(token)].replace(BPE_SPACE_CHAR, " ")
  return ("".join([vocab_list[int(token)] for token in tokens])).replace(BPE_SPACE_CHAR, " ")
