import os
import pickle
from collections import Counter
import jax.numpy as jnp
import random
from tqdm import tqdm

# I had chatgpt do the dataset loading stuff
# reason being i wanted to focus on doing the model from scratch

# OPTIMIZATION implement custom bpe
# BPE Merge Operation
def bpe_merge(data, merge_pairs):
    for old, new in merge_pairs:
        data = data.replace(old, new)
    return data


# Learn BPE Merge Rules
def learn_bpe(data, vocab_size):
    vocab = list(set(data))
    vocab = [f"{char}" for char in vocab]

    data = list(data)
    pair_freq = Counter()

    with tqdm(total=vocab_size) as pb:
        pb.update(len(vocab))
        while len(vocab) < vocab_size:
            pair_freq.clear()
            for i in range(len(data) - 1):
                pair = (data[i], data[i + 1])
                pair_freq[pair] += 1

            if not pair_freq:
                break

            best_pair = max(pair_freq, key=pair_freq.get)
            new_token = ''.join(best_pair)

            new_data = []
            skip = False
            for i in range(len(data)):
                if skip:
                    skip = False
                    continue
                if i < len(data) - 1 and data[i] == best_pair[0] and data[i + 1] == best_pair[1]:
                    new_data.append(new_token)
                    skip = True
                else:
                    new_data.append(data[i])
            data = new_data

            vocab.append(new_token)
            pb.update(len(vocab) - pb.n)

    return vocab, data


# Encode data using the learned BPE rules
def bpe_encode(data, vocab, token_to_id):
    tokenized = []
    idx = 0

    while idx < len(data):
        matched = False

        for token in sorted(vocab, key=len, reverse=True):
            if data.startswith(token, idx):
                tokenized.append(token_to_id[token])
                idx += len(token)
                matched = True
                break

        if not matched:
            tokenized.append(token_to_id[data[idx]])
            idx += 1

    return tokenized


# Decode tokens back to the original text
def bpe_decode(tokens, id_to_token):
    return ''.join([id_to_token[int(token)] for token in tokens])


# Load or Generate BPE Dataset
def load_dataset(dataset_path, split=0.9, keep_ratio=0.8, vocab_size=1000, prefix=''):
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    bpe_file = f"{prefix}data/preprocessed{base_name}_vocab_{vocab_size}.bpe"

    if os.path.exists(bpe_file):
        print(f"Loading BPE vocab and data from {bpe_file}...")
        with open(bpe_file, 'rb') as f:
            vocab, processed_data = pickle.load(f)
    else:
        print(f"Generating BPE vocab and data, saving to {bpe_file}...")
        with open(dataset_path, 'r') as f:
            data = f.read()

        def clean_dataset(data, keep_ratio):
            counter = Counter(data)
            most_common = counter.most_common(int(len(counter) * keep_ratio))
            allowed_chars = set(char for char, _ in most_common)

            def clean_char(c):
                return c if c in allowed_chars else '<UNK>'

            return ''.join(clean_char(c) for c in data)

        cleaned_data = clean_dataset(data, keep_ratio=keep_ratio)
        vocab, processed_data = learn_bpe(cleaned_data, vocab_size=vocab_size)

        with open(bpe_file, 'wb') as f:
            pickle.dump((vocab, processed_data), f)

    # Create mappings
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    tokenized_data = bpe_encode(''.join(processed_data), vocab, token_to_id)
    dataset = jnp.array(tokenized_data)

    split_idx = int(len(dataset) * split)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    sample_idx = random.randint(0, len(processed_data) - 50)
    sample_input = ''.join(processed_data[sample_idx:sample_idx + 50])
    sample_tokens = bpe_encode(sample_input, vocab, token_to_id)
    sample_decoded = bpe_decode(sample_tokens, id_to_token)

    print("Sample Input:", sample_input)
    print("Tokenized:", sample_tokens)
    print("Decoded:", sample_decoded)

    encode = (lambda x: bpe_encode(x, vocab, token_to_id))
    decode = (lambda x: bpe_decode(x, id_to_token))

    return vocab, train_data, test_data, encode, decode
