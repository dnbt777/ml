#!/usr/bin/env python
"""farthest_dictionary_words.py
────────────────────────────────────────────────────
Given a **newline‑separated `dictionary.txt`** (words or phrases that may span
*multiple* tokenizer tokens), this script finds the **two entries whose
embedding representations are farthest apart** — reported for both **cosine**
and **Euclidean** distance.

It needs only:
* the Llama **`embed_tokens.weight`** (loaded straight from your
  `*.safetensors` shards), and
* the custom tokenizer in **`tokenizer.py`** (provided).

Everything is hard‑wired, so just run:
```bash
python farthest_dictionary_words.py
```
with these files in the same directory:
```
./Llama/   # safetensor shards + tokenizer.json
./dictionary.txt
./tokenizer.py
```
The script streams through the vocab in batches, so it can scale to tens of
thousands of dictionary entries on a single GPU.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import safe_open
from tqdm import tqdm

from tokenizer import load_tokenizer, encode  # your custom tokenizer

# ───────────────────────────────────────────
# CONFIG (adjust if needed)
# ───────────────────────────────────────────
MODEL_DIR = Path("./Llama")          # shards + tokenizer.json
DICT_PATH = Path("./dictionary.txt") # newline‑separated words/phrases
BATCH_SIZE = 1_024                   # rows processed per chunk in distance search
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16               # keep memory modest, matches shards

# ───────────────────────────────────────────
# LOAD EMBEDDING MATRIX + TOKENIZER
# ───────────────────────────────────────────

def _locate_embed_tensor(model_dir: Path) -> torch.Tensor:
    """Return the `embed_tokens.weight` matrix as a **torch.Tensor[DTYPE]**."""
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight") or "tok_embeddings.weight" in k:
                    t = f.get_tensor(k)
                    return t.to(dtype=DTYPE)
    raise RuntimeError("embed_tokens.weight not found in shards")


# ───────────────────────────────────────────
# BUILD WORD‑LEVEL EMBEDDINGS
# ───────────────────────────────────────────

def build_dictionary_embeddings(embed: torch.Tensor, tok) -> Tuple[torch.Tensor, List[str]]:
    """Return (matrix of shape [N, D], words list) where each row is the *mean* of
    token embeddings for that dictionary entry. Entries containing OOV token
    IDs are skipped (rare).
    """
    words, vecs = [], []
    D = embed.size(1)
    max_idx = embed.size(0)

    with open(DICT_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if not word:
                continue
            try:
                token_ids = encode(tok, word)
            except Exception:
                # encoding failed – skip
                continue
            # filter out any ids outside embedding matrix
            token_ids = [tid for tid in token_ids if tid < max_idx]
            if not token_ids:
                continue
            vec = embed[token_ids].mean(dim=0)
            words.append(word)
            vecs.append(vec)

    if not vecs:
        raise RuntimeError("No dictionary entries could be encoded with available embeddings.")

    matrix = torch.stack(vecs)  # (N, D)
    return matrix, words


# ───────────────────────────────────────────
# DISTANCE SEARCH (batched, GPU‑friendly)
# ───────────────────────────────────────────

def farthest_pair_cosine(X: torch.Tensor) -> Tuple[float, Tuple[int, int]]:
    N = X.size(0)
    Xn = torch.nn.functional.normalize(X, dim=1)  # unit‑norm rows
    best = -1.0
    pair = (-1, -1)

    for start in tqdm(range(0, N, BATCH_SIZE), desc="cosine"):
        batch = Xn[start : start + BATCH_SIZE]  # (B, D)
        sims = batch @ Xn.T                     # (B, N)
        dists = 1.0 - sims                     # cosine distance ∈ [0, 2]
        b = batch.size(0)
        idx = torch.arange(b, device=X.device)
        dists[idx, start + idx] = -math.inf     # mask self
        val, flat = torch.max(dists.view(-1), 0)
        if val.item() > best:
            best = val.item()
            row = flat // N
            col = flat % N
            pair = (start + row.item(), col.item())
    return best, pair


def farthest_pair_euclid(X: torch.Tensor) -> Tuple[float, Tuple[int, int]]:
    N = X.size(0)
    row_norms = (X ** 2).sum(dim=1)  # (N,)
    best2 = -1.0
    pair = (-1, -1)

    for start in tqdm(range(0, N, BATCH_SIZE), desc="euclid"):
        batch = X[start : start + BATCH_SIZE]           # (B, D)
        x2 = (batch ** 2).sum(dim=1, keepdim=True)      # (B, 1)
        dists2 = x2 + row_norms - 2.0 * (batch @ X.T)   # ||x - y||^2
        b = batch.size(0)
        idx = torch.arange(b, device=X.device)
        dists2[idx, start + idx] = -math.inf
        val2, flat = torch.max(dists2.view(-1), 0)
        if val2.item() > best2:
            best2 = val2.item()
            row = flat // N
            col = flat % N
            pair = (start + row.item(), col.item())
    return math.sqrt(best2), pair


# ───────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────

def main() -> None:
    print("Loading tokenizer …")
    tokenizer = load_tokenizer(str(MODEL_DIR / "tokenizer.json"))

    print("Loading embedding matrix … (this may take a moment)")
    embed = _locate_embed_tensor(MODEL_DIR).to(DEVICE)

    print("Encoding dictionary words → vectors …")
    W, words = build_dictionary_embeddings(embed, tokenizer)
    W = W.to(DEVICE)

    print(f"Dictionary entries encoded: {len(words)}; vector dim = {W.size(1)}")

    cos_d, (ci, cj) = farthest_pair_cosine(W)
    euc_d, (ei, ej) = farthest_pair_euclid(W)

    print("\nFarthest‑apart **dictionary words**")
    print(f"Cosine  : {cos_d:.6f}  →  '{words[ci]}'  ↔  '{words[cj]}'  (idx {ci}, {cj})")
    print(f"Euclid  : {euc_d:.6f}  →  '{words[ei]}'  ↔  '{words[ej]}'  (idx {ei}, {ej})")

main()
