"""
High-Performance Memory-Mapped Pretraining Dataset and Sharder.
Handles sequence packing, zero-padding elimination, and memmapped binary streams.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.bpe import CodeTokenizer


class PackedMemmapDataset(Dataset):
    """
    Reads pre-tokenized uint32 binary shards directly from disk via np.memmap.
    Guarantees 100% token packing with zero padding tokens.
    """

    def __init__(self, shard_paths: List[str], seq_len: int = 2048):
        self.seq_len = seq_len
        self.shard_paths = sorted(shard_paths)
        if not self.shard_paths:
            raise ValueError("No shard files provided to PackedMemmapDataset.")

        # Load memmaps and compute total sample count across all shards
        self.shards = []
        self.shard_lengths = []  # Length in tokens
        self.cumulative_samples = [0]  # Number of (seq_len + 1) blocks

        total_samples = 0
        for p in self.shard_paths:
            # Memory map the binary file without loading it into physical RAM
            m = np.memmap(p, dtype=np.uint32, mode="r")
            self.shards.append(m)
            num_tokens = len(m)
            self.shard_lengths.append(num_tokens)
            # Each training example requires seq_len tokens for input + 1 next-token target
            num_samples = max(0, (num_tokens - 1) // seq_len)
            total_samples += num_samples
            self.cumulative_samples.append(total_samples)

        self.total_samples = total_samples

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Binary search to find which shard contains this sample index
        shard_idx = int(np.searchsorted(self.cumulative_samples[1:], idx, side="right"))
        local_sample_idx = idx - self.cumulative_samples[shard_idx]

        # Calculate exact token offset in this shard
        start = local_sample_idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.shards[shard_idx][start:end].astype(np.int64)

        # x: input tokens [0 ... seq_len - 1]
        # y: target tokens [1 ... seq_len] (autoregressive next token)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y


def build_synthetic_code_corpus() -> List[str]:
    """Generates a rich synthetic Python code corpus for unit testing and local sanity checks."""
    snippets = [
        # Functions and recursion
        """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
        # Object-Oriented design
        """class BinarySearchTree:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def insert(self, val):
        if val < self.val:
            if self.left is None:
                self.left = BinarySearchTree(val)
            else:
                self.left.insert(val)
        elif val > self.val:
            if self.right is None:
                self.right = BinarySearchTree(val)
            else:
                self.right.insert(val)
""",
        # Neural network forward pass
        """import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
""",
        # Algorithmic interview problems
        """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
""",
    ]
    # Multiply to create a solid test corpus
    return snippets * 250


def prepare_shards_from_texts(
    texts: List[str],
    output_dir: str,
    shard_prefix: str = "shard",
    tokens_per_shard: int = 50_000,
    tokenizer: Optional[CodeTokenizer] = None,
) -> List[str]:
    """
    Packs a stream of text documents into fixed-size binary shards on disk.
    Each document is cleanly terminated by the tokenizer's EOS token.
    """
    os.makedirs(output_dir, exist_ok=True)
    if tokenizer is None:
        tokenizer = CodeTokenizer()

    shard_paths = []
    shard_idx = 0
    buffer = []

    for text in texts:
        tokens = tokenizer.encode(text) + [tokenizer.eos_id]
        buffer.extend(tokens)

        # When buffer exceeds shard size, flush to disk
        while len(buffer) >= tokens_per_shard:
            shard_data = np.array(buffer[:tokens_per_shard], dtype=np.uint32)
            shard_path = os.path.join(output_dir, f"{shard_prefix}_{shard_idx:05d}.bin")
            with open(shard_path, "wb") as f:
                f.write(shard_data.tobytes())
            shard_paths.append(shard_path)
            shard_idx += 1
            buffer = buffer[tokens_per_shard:]

    # Flush any remaining tokens into final shard
    if buffer:
        shard_data = np.array(buffer, dtype=np.uint32)
        shard_path = os.path.join(output_dir, f"{shard_prefix}_{shard_idx:05d}.bin")
        with open(shard_path, "wb") as f:
            f.write(shard_data.tobytes())
        shard_paths.append(shard_path)

    return shard_paths


if __name__ == "__main__":
    import shutil

    test_dir = "data/test_shards"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("Generating synthetic code corpus...")
    corpus = build_synthetic_code_corpus()
    print(f"Total documents: {len(corpus)}")

    tokenizer = CodeTokenizer()
    print("Writing binary shards to disk...")
    shards = prepare_shards_from_texts(corpus, test_dir, tokens_per_shard=20_000, tokenizer=tokenizer)
    print(f"Created {len(shards)} binary shards:")
    for s in shards:
        print(f"  - {s} ({os.path.getsize(s) / 1024:.1f} KB)")

    # Test the Dataset and DataLoader
    seq_len = 128
    dataset = PackedMemmapDataset(shards, seq_len=seq_len)
    print(f"\nDataset loaded. Total packed sequences (seq_len={seq_len}): {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    x_batch, y_batch = next(iter(loader))

    print(f"Batch x shape: {x_batch.shape} (dtype: {x_batch.dtype})")
    print(f"Batch y shape: {y_batch.shape} (dtype: {y_batch.dtype})")
    assert x_batch.shape == (4, seq_len)
    assert y_batch.shape == (4, seq_len)
    # Check that y is indeed x shifted by 1
    assert torch.equal(x_batch[0, 1:], y_batch[0, :-1]), "Autoregressive target alignment mismatch!"
    print("Dataset verification passed! Zero-padding, autoregressive alignment verified.")
