#!/usr/bin/env python
"""farthest_tokens_jax.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ultra‑minimal **JAX** version that:
1.  Finds the single `embed_tokens.weight` shard (kept in **bfloat16**).
2.  Computes *both* cosine & Euclidean farthest‑pair on the CPU/GPU/TPU JAX is using.
3.  Ignores any stray tokens whose ID ≥ rows in the embedding matrix so the size
   mismatch disappears.

Drop it next to your `./Llama` folder and run:

```bash
python farthest_tokens_jax.py
```
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
from safetensors.numpy import safe_open  # numpy tensors, we’ll cast to JAX
from tqdm import tqdm

# ───────────────────────────────────────────
# CONFIG – tweak here if you moved things
# ───────────────────────────────────────────
MODEL_DIR = Path("./Llama")          # shards + tokenizer.json live here
BATCH_SIZE = 1_024                   # rows processed at once
DTYPE = jnp.bfloat16                 # keep memory tiny (same as file)

# ───────────────────────────────────────────
# LOAD EMBED + TOKENS (bfloat16 JAX array)
# ───────────────────────────────────────────

def _locate_embed_tensor(shards: List[Path]) -> jnp.ndarray:
    key = None
    for shard in shards:
        with safe_open(shard, framework="numpy") as f:
            if key is None:
                for k in f.keys():
                    if k.endswith("embed_tokens.weight") or "tok_embeddings.weight" in k:
                        key = k
                        break
            if key and key in f.keys():
                np_arr = f.get_tensor(key)           # numpy bfloat16
                return jnp.asarray(np_arr, dtype=DTYPE)
    raise RuntimeError("Could not find embed_tokens.weight in shards")


def load_embed_and_tokens(model_dir: Path) -> Tuple[jnp.ndarray, List[str]]:
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("No .safetensors files found – check MODEL_DIR")

    E = _locate_embed_tensor(shards)  # (V, D) bfloat16 JAX array
    V = E.shape[0]

    # tokens – may be longer than V (e.g. extra special tokens). Trim / pad.
    tok_path = model_dir / "tokenizer.json"
    vocab = json.loads(tok_path.read_text("utf-8"))["model"]["vocab"]

    tokens = [None] * V
    for tok, idx in vocab.items():
        if idx < V and tokens[idx] is None:
            tokens[idx] = tok
    # Fill any holes so we never index None
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"<unk_{i}>"

    return E, tokens

# ───────────────────────────────────────────
# DISTANCE SEARCHERS (python‑side loop, batched)
# ───────────────────────────────────────────

def farthest_cosine(E: jnp.ndarray) -> Tuple[float, Tuple[int, int]]:
    E32 = E.astype(jnp.float32)
    norms = jnp.linalg.norm(E32, axis=1)
    V = E.shape[0]
    best, best_pair = -1.0, (-1, -1)

    for start in tqdm(range(0, V, BATCH_SIZE), desc="cosine"):
        batch = E32[start : start + BATCH_SIZE]
        sims = batch @ E32.T  # (B, V)
        sims /= (jnp.linalg.norm(batch, axis=1, keepdims=True) * norms)
        dists = 1.0 - sims

        # mask self comps if batch overlaps diag
        n = min(batch.shape[0], V - start)
        dists = dists.at[jnp.arange(n), start + jnp.arange(n)].set(-jnp.inf)

        flat_idx = int(jnp.argmax(dists))
        local_best = float(dists.reshape(-1)[flat_idx])
        if local_best > best:
            best = local_best
            row, col = divmod(flat_idx, V)
            best_pair = (start + row, col)
    return best, best_pair


def farthest_euclid(E: jnp.ndarray) -> Tuple[float, Tuple[int, int]]:
    E32 = E.astype(jnp.float32)
    row_norms = jnp.sum(E32 ** 2, axis=1)
    V = E.shape[0]
    best2, best_pair = -1.0, (-1, -1)

    for start in tqdm(range(0, V, BATCH_SIZE), desc="euclid"):
        batch = E32[start : start + BATCH_SIZE]
        x2 = jnp.sum(batch ** 2, axis=1, keepdims=True)
        dists2 = x2 + row_norms - 2.0 * (batch @ E32.T)
        dists2 = jnp.maximum(dists2, 0.0)  # numerical floor

        n = min(batch.shape[0], V - start)
        dists2 = dists2.at[jnp.arange(n), start + jnp.arange(n)].set(-jnp.inf)

        flat_idx = int(jnp.argmax(dists2))
        local_best2 = float(dists2.reshape(-1)[flat_idx])
        if local_best2 > best2:
            best2 = local_best2
            row, col = divmod(flat_idx, V)
            best_pair = (start + row, col)
    return math.sqrt(best2), best_pair

# ───────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────

def main() -> None:
    E, tokens = load_embed_and_tokens(MODEL_DIR)

    cos_d, (ci, cj) = farthest_cosine(E)
    euc_d, (ei, ej) = farthest_euclid(E)

    print("\nFarthest‑apart tokens (JAX)")
    print(f"Cosine  : {cos_d:.6f}  →  '{tokens[ci]}'  ↔  '{tokens[cj]}'  (ids {ci}, {cj})")
    print(f"Euclid  : {euc_d:.6f}  →  '{tokens[ei]}'  ↔  '{tokens[ej]}'  (ids {ei}, {ej})")


if __name__ == "__main__":
    main()
