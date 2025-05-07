import json
from llama_types import Text, Tokens, Tokenizer
from typing import List
import regex as re # not re - regex. this is needed for the llama3 regex

BPE_SPACE_CHAR = 'Ġ'


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
  with open(tokenizer_path, 'r') as config:
    tokenizer_config = json.load(config)
  vocab = tokenizer_config["model"]["vocab"]
  merges = tokenizer_config["model"]["merges"]
  additional_vocab = tokenizer_config["added_tokens"]
  additional_vocab = {entry["content"] : entry["id"] for entry in additional_vocab}
  vocab.update(additional_vocab) # smh python
  return Tokenizer(
    vocab=vocab,
    additional_vocab=additional_vocab,
    merges=merges
  )


# used ai to help w this one 
def split_string_with_substrings(string: str, substrs: List[str]) -> List[str]:
    # Longest‑first avoids sneaky overlaps like "<|image|>" vs "<|"
    substrs_sorted = sorted(substrs, key=len, reverse=True)
    pattern = "|".join(re.escape(s) for s in substrs_sorted)
    parts = []
    last_end = 0
    for m in re.finditer(pattern, string):
        # text before this special token
        if m.start() > last_end:
            parts.append(string[last_end:m.start()])
        # the special token itself
        parts.append(m.group(0))
        last_end = m.end()

    # trailing text after the final special token
    if last_end < len(string):
        parts.append(string[last_end:])

    return parts


# text -> tokens
def encode(tokenizer: Tokenizer, text: str) -> Tokens:

  # first split into special tokens
  subwords = text
  # subwords: "<|image|> Describe the image."
  subwords = split_string_with_substrings(subwords, list(tokenizer.additional_vocab.keys())) # preserves special tokens
  print("post split: ", subwords)
  # subwords: ["<|image|>", " Describe the image."]

  # then split every substring using the regex
  preprocess_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  pattern = re.compile(preprocess_regex)
  subwords = [
      [match.group(0) for match in pattern.finditer(subword)]
      if not (subword.startswith("<|") and subword.endswith("|>"))
      else [subword]
      for subword in subwords
  ]
  subwords = [item for sublist in subwords for item in sublist] # flatten
  print("post regex:", subwords)
  # subwords: ["<|image|>", " Describe", " the", " image", "."]

  # finally, replace space with Gspace
  subwords = [subword.replace(" ", BPE_SPACE_CHAR) for subword in subwords]

  # subwords: ["<|image|>", "ĠDescribe", "Ġthe", "Ġimage", "."]
  
  # do merge loop until no more merges are possible
  i = 0
  while 1:
    if i == len(subwords) - 1:
      break # done
    if (subwords[i] in tokenizer.additional_vocab) or (subwords[i+1] in tokenizer.additional_vocab):
      i += 1
      continue # skip special tokens
    attempted_merge = f"{subwords[i]} {subwords[i+1]}"
    potential_vocab = f"{subwords[i]}{subwords[i+1]}"
    # test if combined tokens are in vocab first, according to the llama3 docs https://huggingface.co/docs/transformers/en/model_doc/llama3
    print("attempted merge: ", attempted_merge)
    print("potential vocab: ", potential_vocab)
    if potential_vocab in tokenizer.vocab or potential_vocab in tokenizer.additional_vocab:
      subwords[i] = potential_vocab
      subwords.pop(i+1)
      i=0
      print("potential word not in vocab")
      continue
    # then test for merge rule
    elif attempted_merge in tokenizer.merges:
      subwords[i] = attempted_merge.replace(" ", "")
      subwords.pop(i+1)
      # if merge found, restart
      i = 0
      print("merge found")
      continue
    else:
      i += 1
      print("no merge found")
    print()
  # convert to tokens and return
  encoded = [tokenizer.vocab[subword] for subword in subwords]
  return encoded



# tokens -> text
def decode(tokenizer: Tokenizer, tokens: Tokens) -> Text:
  vocab_list = list(tokenizer.vocab.keys())
  if len(tokens) == 1: # use len not shape in case tokens is a list
      return vocab_list[int(tokens[0])].replace(BPE_SPACE_CHAR, " ")
  return ("".join([vocab_list[int(token)] for token in tokens])).replace(BPE_SPACE_CHAR, " ")

