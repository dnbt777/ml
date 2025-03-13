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
  if tokens.shape[0] == 1:
      return vocab_list[int(tokens[0])].replace(BPE_SPACE_CHAR, " ")
  return ("".join([vocab_list[int(token)] for token in tokens])).replace(BPE_SPACE_CHAR, " ")



def pretokenizer(preprocessor: Tokenizer, text: Text) -> Text:
  """
    "pre_tokenizer": {
      "type": "Sequence",
      "pretokenizers": [
        {
          "type": "Split",
          "pattern": {
            "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
          },
          "behavior": "Isolated",
          "invert": false
        },
        {
          "type": "ByteLevel",
          "add_prefix_space": false,
          "trim_offsets": true,
          "use_regex": false
        }
      ]
    },"""
  pass


def post_processor(postprocessor: Tokenizer, text: Text) -> Text:
  """
    "post_processor": {
      "type": "Sequence",
      "processors": [
        {
          "type": "ByteLevel",
          "add_prefix_space": true,
          "trim_offsets": false,
          "use_regex": true
        },
        {
          "type": "TemplateProcessing",
          "single": [
            {
              "SpecialToken": {
                "id": "<|begin_of_text|>",
                "type_id": 0
              }
            },
            {
              "Sequence": {
                "id": "A",
                "type_id": 0
              }
            }
          ],
          "pair": [
            {
              "SpecialToken": {
                "id": "<|begin_of_text|>",
                "type_id": 0
              }
            },
            {
              "Sequence": {
                "id": "A",
                "type_id": 0
              }
            },
            {
              "SpecialToken": {
                "id": "<|begin_of_text|>",
                "type_id": 1
              }
            },
            {
              "Sequence": {
                "id": "B",
                "type_id": 1
              }
            }
          ],
          "special_tokens": {
            "<|begin_of_text|>": {
              "id": "<|begin_of_text|>",
              "ids": [
                128000
              ],
              "tokens": [
                "<|begin_of_text|>"
              ]
            }
          }
        }
      ]
    },"""
  pass