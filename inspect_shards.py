"""
Interactive Shard Inspector:
See how binary uint32 tokens are stored on disk and read via zero-copy np.memmap.
"""

import os
import glob
import numpy as np
from src.tokenizer.bpe import CodeTokenizer


def inspect_binary_shards():
    # 1. Find the shards we created in data/test_shards
    shard_files = sorted(glob.glob("data/test_shards/shard_*.bin"))
    if not shard_files:
        print("No shards found! Run dataset creation first.")
        return

    first_shard = shard_files[0]
    file_size_bytes = os.path.getsize(first_shard)
    num_tokens = file_size_bytes // 4  # 4 bytes per uint32 integer

    print("=" * 60)
    print("1. DISK INSPECTION")
    print(f"File path: {first_shard}")
    print(f"File size on disk: {file_size_bytes} bytes ({file_size_bytes / 1024:.2f} KB)")
    print(f"Total uint32 tokens stored: {num_tokens:,}")

    # Read the raw first 16 bytes directly from disk (hexadecimal)
    with open(first_shard, "rb") as f:
        raw_bytes = f.read(16)
    print(f"Raw first 16 bytes on disk (hex): {raw_bytes.hex(' ')}")

    print("\n" + "=" * 60)
    print("2. ZERO-COPY MEMMAP READ")
    # np.memmap does NOT load the file into RAM; it maps the file to memory addresses
    mmap_data = np.memmap(first_shard, dtype=np.uint32, mode="r")
    
    # Grab the first 10 token integers
    first_10_tokens = mmap_data[:10]
    print(f"First 10 token integers: {list(first_10_tokens)}")

    print("\n" + "=" * 60)
    print("3. DECODING TOKENS BACK TO CODE")
    tokenizer = CodeTokenizer()
    for i, tok_id in enumerate(first_10_tokens):
        # Decode individual token
        tok_str = tokenizer.decode([tok_id])
        # Make whitespace visible for learning
        display_str = tok_str.replace(" ", "[SPACE]").replace("\n", "[NEWLINE]\n")
        print(f"Token #{i:2d} | ID: {tok_id:<6d} | Text: '{display_str}'")

    print("\n" + "=" * 60)
    print("4. FINDING THE <|endoftext|> BOUNDARY")
    eos_id = tokenizer.eos_id
    eos_positions = np.where(mmap_data[:1000] == eos_id)[0]
    print(f"The <|endoftext|> separator token ID is: {eos_id}")
    print(f"Found <|endoftext|> at token indices: {list(eos_positions[:5])}")

    # Show the code snippet right around the first EOS boundary
    first_eos = eos_positions[0]
    snippet_before = tokenizer.decode(mmap_data[max(0, first_eos - 10):first_eos])
    snippet_after = tokenizer.decode(mmap_data[first_eos + 1:first_eos + 11])
    print("\nText immediately BEFORE the boundary:")
    print(repr(snippet_before))
    print("\nText immediately AFTER the boundary:")
    print(repr(snippet_after))
    print("=" * 60)


if __name__ == "__main__":
    inspect_binary_shards()
