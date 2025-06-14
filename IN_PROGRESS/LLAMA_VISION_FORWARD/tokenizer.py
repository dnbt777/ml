# NGL I used heavy AI to write the tokenizer/decoder

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
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

def bpe_merge(byte_tokens, merge_ranks):
    while True:
        pairs = [(byte_tokens[i], byte_tokens[i+1]) for i in range(len(byte_tokens)-1)]
        if not pairs:
            break
        pair_ranks = [(pair, merge_ranks.get(pair, float('inf'))) for pair in pairs]
        best_pair, best_rank = min(pair_ranks, key=lambda x: x[1])
        if best_rank == float('inf'):
            break

        # Merge the best pair
        new_tokens = []
        i = 0
        while i < len(byte_tokens):
            if i < len(byte_tokens) - 1 and (byte_tokens[i], byte_tokens[i+1]) == best_pair:
                new_tokens.append(byte_tokens[i] + byte_tokens[i+1])
                i += 2
            else:
                new_tokens.append(byte_tokens[i])
                i += 1
        byte_tokens = new_tokens

    return byte_tokens

def encode(tokenizer: Tokenizer, text: str) -> Tokens:
    subwords = split_string_with_substrings(text, list(tokenizer.additional_vocab.keys()))
    #print("post split: ", subwords)

    preprocess_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    pattern = re.compile(preprocess_regex)
    subwords = [
        [match.group(0) for match in pattern.finditer(subword)]
        if not (subword.startswith("<|") and subword.endswith("|>"))
        else [subword]
        for subword in subwords
    ]
    subwords = [item for sublist in subwords for item in sublist]
    #print("post regex:", subwords)

    subwords = [subword.replace(" ", BPE_SPACE_CHAR) for subword in subwords]

    merge_ranks = {tuple(m.split()): i for i, m in enumerate(tokenizer.merges)}
    final_tokens = []

    for subword in subwords:
        if subword in tokenizer.vocab:
            final_tokens.append(tokenizer.vocab[subword])
            continue
        if subword in tokenizer.additional_vocab:
            final_tokens.append(tokenizer.additional_vocab[subword])
            continue

        # Convert to UTF-8 bytes, then byte-to-unicode tokens
        byte_string = subword.encode('utf-8')
        byte_tokens = [BYTE_ENCODER[b] for b in byte_string]

        # Merge using ranked BPE
        merged_tokens = bpe_merge(byte_tokens, merge_ranks)

        for token_str in merged_tokens:
            if token_str in tokenizer.vocab:
                final_tokens.append(tokenizer.vocab[token_str])
            else:
                raise KeyError(f"Sub-token '{token_str}' not in vocab — cannot encode '{subword}'")

    return final_tokens


# tokens -> text
def decode(tokenizer: Tokenizer, tokens: Tokens) -> Text:
    vocab_items = list(tokenizer.vocab.items())
    id_to_token = {v: k for k, v in vocab_items}

    pieces = []
    for token in tokens:
        token = int(token)
        if token in id_to_token:
            pieces.append(id_to_token[token])
        else:
            raise ValueError(f"Token ID {token} not in vocab.")

    # Decode merged tokens using byte decoder
    text = "".join(pieces)
    byte_sequence = bytearray()

    for char in text:
        if char in BYTE_DECODER:
            byte_sequence.append(BYTE_DECODER[char])
        else:
            # assume this is a special token, not part of byte encoding
            byte_sequence += char.encode('utf-8')

    return byte_sequence.decode('utf-8', errors='replace').replace(BPE_SPACE_CHAR, " ")

