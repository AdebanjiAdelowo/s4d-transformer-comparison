"""
Character-level Tiny Shakespeare data pipeline, adapted directly from this
candidate's own nano-gpt/train.py -- reused for engineering consistency (so
both architectures under comparison see byte-identical batches), not as a
source for any S4D claim.
"""

import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_tinyshakespeare(data_dir: str = "data"):
    data_path = os.path.join(data_dir, "input.txt")
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(data_path):
        print("Downloading Tiny Shakespeare ...")
        urllib.request.urlretrieve(DATA_URL, data_path)

    with open(data_path, encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return {
        "train": data[:n],
        "val": data[n:],
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "encode": encode,
        "decode": decode,
        "n_chars": len(text),
    }


def get_batch(split_data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
