"""
Experience Replay Shard Packer for RSI Fine-Tuning.
Blends accepted self-play verified code (10%) with pretraining baseline code (90%)
to produce a balanced training shard for M_0 -> M_1 distillation without catastrophic forgetting.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.tokenizer.bpe import CodeTokenizer


def pack_experience_replay_shard(
    verified_solutions_path: str = "data/rsi/verified_solutions.jsonl",
    baseline_shard_path: str = "data/shards/train_shard_00000.bin",
    output_shard_path: str = "data/shards/train_shard_rsi_m1.bin",
    problem_bank_path: str = "data/rsi/problem_bank.json",
    target_tokens: int = 4_000_000,
    replay_ratio: float = 0.10,  # 10% self-play verified code, 90% pretraining code
    bootstrap_if_empty: bool = True,
):
    Path(output_shard_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = CodeTokenizer()

    # 1. Load verified solutions
    verified_records = []
    if os.path.exists(verified_solutions_path):
        with open(verified_solutions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    verified_records.append(json.loads(line))

    print(f"--> Found {len(verified_records)} self-play verified solutions in: {verified_solutions_path}")

    # Bootstrap mechanism if model scored 0 passing candidates (Curriculum Cold-Start Protection)
    if len(verified_records) == 0 and bootstrap_if_empty:
        print("[INFO] Cold-Start Protection Active: Bootstrapping replay buffer with canonical seed solutions...")
        with open(problem_bank_path, "r", encoding="utf-8") as f:
            problems = json.load(f)
        for prob in problems[:10]:  # Use Tier-1 canonical solutions as initial anchor
            verified_records.append({
                "task_id": prob["task_id"],
                "name": prob["name"],
                "prompt": prob["prompt"],
                "solution": prob.get("canonical") or prob.get("canonical_solution"),
            })
        print(f"--> Bootstrapped {len(verified_records)} anchor solutions from Tier-1 problem bank.")

    # 2. Tokenize verified code
    verified_tokens = []
    for rec in verified_records:
        prompt = rec["prompt"].rstrip()
        solution = rec.get("solution") or rec.get("completion") or ""
        # Clean solution to prevent prompt duplication
        if solution.startswith(prompt):
            full_code = solution.strip()
        else:
            full_code = f"{prompt}\n    {solution.strip()}"

        tokens = tokenizer.encode(full_code) + [tokenizer.eos_id]
        verified_tokens.extend(tokens)

    print(f"--> Tokenized Verified Replay Pool: {len(verified_tokens):,} tokens ({len(verified_records)} samples)")

    # 3. Read baseline pretraining shard
    print(f"--> Reading baseline pretraining shard: {baseline_shard_path}")
    baseline_memmap = np.memmap(baseline_shard_path, dtype=np.uint32, mode="r")
    total_baseline_tokens = len(baseline_memmap)
    print(f"--> Baseline Shard Available: {total_baseline_tokens:,} tokens")

    # 4. Synthesize 90/10 Experience Replay Shard
    # Number of verified tokens target and baseline tokens target
    target_verified_tokens = int(target_tokens * replay_ratio)
    target_baseline_tokens = target_tokens - target_verified_tokens

    # Repeat verified tokens cyclically to reach target_verified_tokens
    if len(verified_tokens) == 0:
        raise ValueError("Cannot pack replay shard: zero verified tokens available.")

    repeats = (target_verified_tokens // len(verified_tokens)) + 1
    expanded_verified = (verified_tokens * repeats)[:target_verified_tokens]

    # Sample slice of baseline tokens
    slice_start = np.random.randint(0, max(1, total_baseline_tokens - target_baseline_tokens))
    slice_baseline = baseline_memmap[slice_start : slice_start + target_baseline_tokens].tolist()

    # Interleave 90/10 in sequence-sized chunks (2048 tokens each)
    seq_len = 2048
    mixed_buffer = []

    v_idx = 0
    b_idx = 0

    chunk_verified_size = int(seq_len * replay_ratio)
    chunk_baseline_size = seq_len - chunk_verified_size

    while len(mixed_buffer) < target_tokens and b_idx < len(slice_baseline):
        # Add baseline block
        b_chunk = slice_baseline[b_idx : b_idx + chunk_baseline_size]
        mixed_buffer.extend(b_chunk)
        b_idx += chunk_baseline_size

        # Add verified block
        v_chunk = expanded_verified[v_idx : v_idx + chunk_verified_size]
        mixed_buffer.extend(v_chunk)
        v_idx = (v_idx + chunk_verified_size) % len(expanded_verified)

    mixed_buffer = mixed_buffer[:target_tokens]
    shard_array = np.array(mixed_buffer, dtype=np.uint32)

    with open(output_shard_path, "wb") as f:
        f.write(shard_array.tobytes())

    file_size_mb = os.path.getsize(output_shard_path) / (1024 * 1024)
    print(f"\n{'='*70}")
    print("EXPERIENCE REPLAY SHARD COMPILATION COMPLETE")
    print(f"{'='*70}")
    print(f"Target Output Path:       {output_shard_path}")
    print(f"Total Packed Tokens:      {len(shard_array):,} tokens")
    print(f"File Size on Disk:        {file_size_mb:.2f} MB")
    print(f"Replay Mix Ratio:         {replay_ratio * 100:.1f}% Self-Play / {(1 - replay_ratio) * 100:.1f}% Pretrain Code")
    print(f"{'='*70}\n")

    return output_shard_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", type=str, default="data/rsi/verified_solutions.jsonl")
    parser.add_argument("--baseline", type=str, default="data/shards/train_shard_00000.bin")
    parser.add_argument("--output", type=str, default="data/shards/train_shard_rsi_m1.bin")
    parser.add_argument("--tokens", type=int, default=4_000_000)
    parser.add_argument("--ratio", type=float, default=0.10)
    args = parser.parse_args()

    pack_experience_replay_shard(
        verified_solutions_path=args.verified,
        baseline_shard_path=args.baseline,
        output_shard_path=args.output,
        target_tokens=args.tokens,
        replay_ratio=args.ratio,
    )
