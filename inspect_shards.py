"""
Interactive Shard Inspector:
See how binary uint32 tokens are stored on disk and read via zero-copy np.memmap.
Supports inspecting any shard path via command-line arguments.
"""

import os
import glob
import argparse
import numpy as np
from src.tokenizer.bpe import CodeTokenizer


def inspect_binary_shards(pattern: str = None, count: int = 15):
    # Search paths in priority order
    candidates = []
    if pattern:
        candidates = sorted(glob.glob(pattern))
    if not candidates:
        candidates = sorted(glob.glob("data/shards/train_shard_*.bin"))
    if not candidates:
        candidates = sorted(glob.glob("data/test_shards/shard_*.bin"))

    if not candidates:
        print(f"No binary shards found matching pattern: {pattern or 'data/shards/train_shard_*.bin'}")
        return

    first_shard = candidates[0]
    file_size_bytes = os.path.getsize(first_shard)
    num_tokens = file_size_bytes // 4  # 4 bytes per uint32 integer

    print("=" * 65)
    print("1. DISK INSPECTION")
    print("=" * 65)
    print(f"File path:            {first_shard}")
    print(f"File size on disk:    {file_size_bytes:,} bytes ({file_size_bytes / (1024 * 1024):.2f} MB)")
    print(f"Total uint32 tokens:  {num_tokens:,}")
    print(f"Bytes per token:      4 bytes (np.uint32)")

    # Read the raw first 16 bytes directly from disk (hexadecimal)
    with open(first_shard, "rb") as f:
        raw_bytes = f.read(16)
    print(f"Raw first 16 bytes:   {raw_bytes.hex(' ')}")

    print("\n" + "=" * 65)
    print("2. ZERO-COPY MEMMAP READ (Without loading file into RAM)")
    print("=" * 65)
    mmap_data = np.memmap(first_shard, dtype=np.uint32, mode="r")
    
    first_tokens = mmap_data[:count]
    print(f"First {count} token integers:")
    print(f"  {list(first_tokens)}")

    print("\n" + "=" * 65)
    print("3. DECODING TOKENS BACK TO CODE")
    print("=" * 65)
    tokenizer = CodeTokenizer()
    for i, tok_id in enumerate(first_tokens):
        tok_str = tokenizer.decode([tok_id])
        display_str = tok_str.replace(" ", "[SPACE]").replace("\n", "[NEWLINE]\n")
        print(f"Token #{i:2d} | ID: {tok_id:<6d} | Text: '{display_str}'")

    print("\n" + "=" * 65)
    print("4. SEQUENCE PACKING: THE <|endoftext|> BOUNDARY")
    print("=" * 65)
    eos_id = tokenizer.eos_id
    # Scan first 50,000 tokens for EOS boundaries
    eos_positions = np.where(mmap_data[:50000] == eos_id)[0]
    print(f"The <|endoftext|> boundary token ID: {eos_id}")

    if len(eos_positions) > 0:
        first_eos = eos_positions[0]
        print(f"Found first file boundary at token index: #{first_eos:,}")
        
        snippet_before = tokenizer.decode(mmap_data[max(0, first_eos - 8):first_eos])
        snippet_after = tokenizer.decode(mmap_data[first_eos + 1:first_eos + 9])
        
        print("\n--> End of File 1 (Tokens right before <|endoftext|>):")
        print("-" * 40)
        print(snippet_before)
        print("-" * 40)
        print(f"--> [BOUNDARY TOKEN #{first_eos}: <|endoftext|>] <--")
        print("-" * 40)
        print("--> Start of File 2 (Tokens right after <|endoftext|>):")
        print("-" * 40)
        print(snippet_after)
        print("-" * 40)
    else:
        print("No EOS boundary found in the inspected window.")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect binary shards")
    parser.add_argument("--pattern", type=str, default=None, help="Glob pattern or file path")
    parser.add_argument("--count", type=int, default=15, help="Number of initial tokens to inspect")
    args = parser.parse_args()

    inspect_binary_shards(pattern=args.pattern, count=args.count)
