"""
Production Code Dataset Sharder for DGX:
Downloads real Python code from Hugging Face and packs it into
high-throughput uint32 memory-mapped binary shards for pretraining.
Uses ungated datasets (codeparrot-clean-valid) by default.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from src.tokenizer.bpe import CodeTokenizer


def shard_code_dataset(
    dataset_name: str = "codeparrot/codeparrot-clean-valid",
    subset: str = None,
    split: str = "train",
    output_dir: str = "data/shards",
    tokens_per_shard: int = 5_000_000,
    max_tokens_total: int = 10_000_000,
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = CodeTokenizer()

    print(f"Loading dataset '{dataset_name}' in streaming mode...")
    kwargs = {"split": split, "streaming": True}
    if subset:
        kwargs["data_dir"] = subset
    ds = load_dataset(dataset_name, **kwargs)

    buffer = []
    shard_idx = 0
    total_tokens_written = 0

    pbar = tqdm(total=max_tokens_total, unit="tokens", desc="Packing Binary Shards")

    for item in ds:
        # Code text can be in 'content', 'code', or 'text'
        code_text = item.get("content") or item.get("code") or item.get("text") or ""
        if not code_text or len(code_text.strip()) < 20:
            continue

        # Encode and append EOS token (<|endoftext|>)
        tokens = tokenizer.encode(code_text) + [tokenizer.eos_id]
        buffer.extend(tokens)

        # Flush full shards
        while len(buffer) >= tokens_per_shard and total_tokens_written < max_tokens_total:
            tokens_to_write = min(tokens_per_shard, max_tokens_total - total_tokens_written)
            shard_data = np.array(buffer[:tokens_to_write], dtype=np.uint32)
            shard_path = os.path.join(output_dir, f"train_shard_{shard_idx:05d}.bin")
            with open(shard_path, "wb") as f:
                f.write(shard_data.tobytes())

            shard_idx += 1
            total_tokens_written += tokens_to_write
            pbar.update(tokens_to_write)
            buffer = buffer[tokens_to_write:]

            if total_tokens_written >= max_tokens_total:
                break

        if total_tokens_written >= max_tokens_total:
            break

    # Write remaining buffer tokens if needed
    if buffer and total_tokens_written < max_tokens_total:
        tokens_to_write = min(len(buffer), max_tokens_total - total_tokens_written)
        shard_data = np.array(buffer[:tokens_to_write], dtype=np.uint32)
        shard_path = os.path.join(output_dir, f"train_shard_{shard_idx:05d}.bin")
        with open(shard_path, "wb") as f:
            f.write(shard_data.tobytes())
        total_tokens_written += tokens_to_write
        pbar.update(tokens_to_write)
        shard_idx += 1

    pbar.close()
    print(f"\nCompleted! Written {total_tokens_written:,} tokens across {shard_idx} binary shards in '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack code corpus into binary shards")
    parser.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean-valid", help="Hugging Face dataset")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/data_dir")
    parser.add_argument("--tokens", type=int, default=10_000_000, help="Total tokens to pack")
    parser.add_argument("--shard-size", type=int, default=5_000_000, help="Tokens per shard")
    parser.add_argument("--output-dir", type=str, default="data/shards", help="Output directory")
    args = parser.parse_args()

    shard_code_dataset(
        dataset_name=args.dataset,
        subset=args.subset,
        output_dir=args.output_dir,
        max_tokens_total=args.tokens,
        tokens_per_shard=args.shard_size,
    )
