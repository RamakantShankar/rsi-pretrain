"""
Production Dataset Sharder for DGX:
Downloads real Python code from Hugging Face and packs it into
high-throughput uint32 memory-mapped binary shards for pretraining.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from src.tokenizer.bpe import CodeTokenizer


def shard_code_dataset(
    dataset_name: str = "bigcode/the-stack-smol",
    subset: str = "data/python",
    split: str = "train",
    output_dir: str = "data/shards",
    tokens_per_shard: int = 10_000_000,
    max_tokens_total: int = 50_000_000,  # 50M tokens for initial run
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = CodeTokenizer()

    print(f"Loading dataset '{dataset_name}' (subset: {subset}) in streaming mode...")
    ds = load_dataset(dataset_name, data_dir=subset, split=split, streaming=True)

    buffer = []
    shard_idx = 0
    total_tokens_written = 0

    pbar = tqdm(total=max_tokens_total, unit="tokens", desc="Token Packing Shards")

    for item in ds:
        # The-Stack stores code in 'content'
        code_text = item.get("content") or item.get("code") or ""
        if not code_text or len(code_text.strip()) < 20:
            continue

        # Encode and append EOS token
        tokens = tokenizer.encode(code_text) + [tokenizer.eos_id]
        buffer.extend(tokens)

        # Flush full shards
        while len(buffer) >= tokens_per_shard:
            shard_data = np.array(buffer[:tokens_per_shard], dtype=np.uint32)
            shard_path = os.path.join(output_dir, f"train_shard_{shard_idx:05d}.bin")
            with open(shard_path, "wb") as f:
                f.write(shard_data.tobytes())

            shard_idx += 1
            total_tokens_written += tokens_per_shard
            pbar.update(tokens_per_shard)
            buffer = buffer[tokens_per_shard:]

            if total_tokens_written >= max_tokens_total:
                break

        if total_tokens_written >= max_tokens_total:
            break

    # Write remaining tokens
    if buffer and total_tokens_written < max_tokens_total:
        shard_data = np.array(buffer, dtype=np.uint32)
        shard_path = os.path.join(output_dir, f"train_shard_{shard_idx:05d}.bin")
        with open(shard_path, "wb") as f:
            f.write(shard_data.tobytes())
        total_tokens_written += len(buffer)
        pbar.update(len(buffer))

    pbar.close()
    print(f"\nCompleted! Written {total_tokens_written:,} tokens across {shard_idx + 1} binary shards in '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50_000_000, help="Total tokens to pack")
    parser.add_argument("--shard-size", type=int, default=10_000_000, help="Tokens per shard")
    args = parser.parse_args()

    shard_code_dataset(
        max_tokens_total=args.tokens,
        tokens_per_shard=args.shard_size,
    )
