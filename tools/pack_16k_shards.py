#!/usr/bin/env python3
"""
Packs the 157MB AST-verified 80/20 corpus into uint32 binary shards using mirendil_16k.json.
Splits 95% into train_shard_00000.bin and 5% into val_shard_00000.bin.
"""

import os
import time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def pack_16k_shards(
    corpus_path: str = "data/tokenizer_train_corpus.txt",
    tokenizer_path: str = "checkpoints/tokenizers/mirendil_16k.json",
    output_dir: str = "data/shards_16k",
    val_ratio: float = 0.05,
):
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("PACKING MEMORY-MAPPED SHARDS WITH MIRENDIL 16K TOKENIZER")
    print(f"Corpus:    {corpus_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Output:    {output_dir}")
    print(f"{'='*70}")

    t0 = time.time()
    tok = Tokenizer.from_file(tokenizer_path)
    eot_id = tok.token_to_id("<|endoftext|>") or 0
    print(f"Loaded Tokenizer: Vocab={tok.get_vocab_size():,}, EOT={eot_id}")

    train_tokens = []
    val_tokens = []
    line_count = 0
    total_bytes = 0

    val_interval = int(1.0 / val_ratio)

    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        batch_lines = []
        for line in f:
            line_str = line.strip()
            if not line_str or line_str == "<|endoftext|>":
                continue
            batch_lines.append(line_str)
            line_count += 1
            total_bytes += len(line_str.encode("utf-8"))

            if len(batch_lines) >= 5000:
                encs = tok.encode_batch(batch_lines)
                for idx, enc in enumerate(encs):
                    is_val = ((line_count - len(batch_lines) + idx) % val_interval) == 0
                    tok_ids = enc.ids + [eot_id]
                    if is_val:
                        val_tokens.extend(tok_ids)
                    else:
                        train_tokens.extend(tok_ids)
                batch_lines = []
                if line_count % 50000 == 0:
                    print(f"Processed {line_count:,} snippets... (Train: {len(train_tokens):,}, Val: {len(val_tokens):,} tokens)")

        if batch_lines:
            encs = tok.encode_batch(batch_lines)
            for idx, enc in enumerate(encs):
                is_val = ((line_count - len(batch_lines) + idx) % val_interval) == 0
                tok_ids = enc.ids + [eot_id]
                if is_val:
                    val_tokens.extend(tok_ids)
                else:
                    train_tokens.extend(tok_ids)

    # Save binary uint32 shards
    train_shard = out_p / "train_shard_00000.bin"
    val_shard = out_p / "val_shard_00000.bin"

    print(f"\nWriting uint32 binary shards to disk...")
    np.array(train_tokens, dtype=np.uint32).tofile(train_shard)
    np.array(val_tokens, dtype=np.uint32).tofile(val_shard)

    elapsed = time.time() - t0
    train_size_mb = train_shard.stat().st_size / 1e6
    val_size_mb = val_shard.stat().st_size / 1e6

    print(f"\n{'='*70}")
    print(f"SHARDS COMPLETE in {elapsed:.2f} seconds!")
    print(f"Train Shard: {train_shard} ({len(train_tokens):,} tokens, {train_size_mb:.2f} MB)")
    print(f"Val Shard:   {val_shard} ({len(val_tokens):,} tokens, {val_size_mb:.2f} MB)")
    print(f"Compression: {total_bytes / (len(train_tokens) + len(val_tokens)):.2f} Bytes/Token")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    pack_16k_shards()
