"""
High-Quality Python Educational Corpus Sharder:
Streams and packs 50,000,000 tokens of high-signal Python code into
uint32 memory-mapped binary shards for foundation pretraining.
Mixes:
  1. flytech/python-codes-25k (clean algorithmic and practical Python)
  2. TokenBender/code_instructions_122k_alpaca_style (instruction-following Python)
  3. codeparrot/codeparrot-clean-valid (idiomatic Python modules)
"""

import os
import re
import time
import argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from src.tokenizer.bpe import CodeTokenizer


def is_valid_python(code: str) -> bool:
    """Fast heuristic to ensure the snippet is valid Python code."""
    if len(code.strip()) < 30:
        return False
    # Must have typical Python constructs
    has_def_or_class = bool(re.search(r'\b(def|class|import|from|return|for|while|if)\b', code))
    # Exclude C++ / Java / HTML
    has_foreign = bool(re.search(r'(#include|<html|public class|System\.out)', code))
    return has_def_or_class and not has_foreign


def stream_python_snippets():
    """Generator that yields clean Python code strings from multiple ungated sources."""
    # 1. flytech/python-codes-25k
    print("--> [Source 1/3] Streaming flytech/python-codes-25k...")
    try:
        ds1 = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
        for item in ds1:
            code = item.get("output") or item.get("text") or ""
            # Strip markdown code blocks if present
            code = re.sub(r'```python\n?', '', code)
            code = re.sub(r'```\n?', '', code)
            if is_valid_python(code):
                yield code.strip()
    except Exception as e:
        print(f"Warning: flytech stream error: {e}")

    # 2. TokenBender/code_instructions_122k_alpaca_style
    print("--> [Source 2/3] Streaming TokenBender/code_instructions_122k_alpaca_style...")
    try:
        ds2 = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train", streaming=True)
        for item in ds2:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            # Clean up markdown
            output = re.sub(r'```python\n?', '', output)
            output = re.sub(r'```\n?', '', output)
            if is_valid_python(output):
                # Format with docstring if helpful
                if instruction and "def " in output and '"""' not in output:
                    lines = output.split('\n')
                    if lines and lines[0].startswith("def "):
                        snippet = f"{lines[0]}\n    \"\"\"{instruction}\"\"\"\n" + "\n".join(lines[1:])
                    else:
                        snippet = f"# {instruction}\n{output}"
                else:
                    snippet = output
                yield snippet
    except Exception as e:
        print(f"Warning: TokenBender stream error: {e}")

    # 3. codeparrot/codeparrot-clean-valid
    print("--> [Source 3/3] Streaming codeparrot/codeparrot-clean-valid...")
    try:
        ds3 = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
        for item in ds3:
            code = item.get("content") or ""
            if is_valid_python(code):
                yield code.strip()
    except Exception as e:
        print(f"Warning: CodeParrot stream error: {e}")


def shard_educational_python(
    output_dir: str = "data/shards_edu",
    max_tokens_total: int = 50_000_000,
    tokens_per_shard: int = 5_000_000,
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = CodeTokenizer()

    print(f"\n======================================================================")
    print(f"PACKING 50M EDUCATIONAL PYTHON TOKENS INTO MEMORY-MAPPED BINARY SHARDS")
    print(f"Target Total Tokens:  {max_tokens_total:,}")
    print(f"Tokens Per Shard:     {tokens_per_shard:,} (Total Shards: {max_tokens_total // tokens_per_shard})")
    print(f"Output Directory:     {output_dir}")
    print(f"======================================================================\n")

    buffer = []
    shard_idx = 0
    total_tokens_written = 0
    t0 = time.time()

    pbar = tqdm(total=max_tokens_total, unit="tokens", desc="Packing Shards")

    for snippet in stream_python_snippets():
        # Encode with EOS token
        tokens = tokenizer.encode(snippet) + [tokenizer.eos_id]
        buffer.extend(tokens)

        # Flush shards
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

    # Flush remainder if needed
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
    elapsed = time.time() - t0
    speed = total_tokens_written / elapsed if elapsed > 0 else 0
    print(f"\n======================================================================")
    print(f"SHARDING COMPLETED SUCCESSFULLY in {elapsed:.1f}s ({speed:,.0f} tok/s)")
    print(f"Total Shards Generated: {shard_idx}")
    print(f"Total Tokens Written:   {total_tokens_written:,}")
    print(f"======================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack educational Python into binary shards")
    parser.add_argument("--tokens", type=int, default=50_000_000, help="Total tokens to pack")
    parser.add_argument("--shard-size", type=int, default=5_000_000, help="Tokens per shard")
    parser.add_argument("--output-dir", type=str, default="data/shards_edu", help="Output directory")
    args = parser.parse_args()

    shard_educational_python(
        output_dir=args.output_dir,
        max_tokens_total=args.tokens,
        tokens_per_shard=args.shard_size,
    )
