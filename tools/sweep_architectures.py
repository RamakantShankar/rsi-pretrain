#!/usr/bin/env python3
"""
Pillar 1 Architecture Sweep Runner on DGX Grace Blackwell GB10
Executes a 3-arm controlled empirical ablation comparing:
  - Arm 1: Deep Reasoning (L=22, d=768)  -> 151.0M params
  - Arm 2: Balanced Sweet-Spot (L=16, d=896) -> 154.2M params
  - Arm 3: Wide Memory (L=12, d=1024)    -> 152.1M params
All trained on mirendil_16k.json under identical FLOPs and data streams.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
import json
import math
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import ModelConfig, TransformerDecoder
from src.tokenizer.bpe import CodeTokenizer
from src.data.dataset import PackedMemmapDataset
from src.engine.optim import configure_optimizers, get_lr_scheduler


# Define the 3 candidate sweep arms (all ~152M params)
CANDIDATE_ARMS = {
    "arm1_deep": {
        "name": "Arm 1: Deep Reasoning (L=22, d=768)",
        "config": ModelConfig(
            vocab_size=16384,
            dim=768,
            n_layers=22,
            n_heads=12,
            n_kv_heads=4,
            max_seq_len=512,
            tie_embeddings=True,
        ),
    },
    "arm2_balanced": {
        "name": "Arm 2: Balanced Sweet-Spot (L=16, d=896)",
        "config": ModelConfig(
            vocab_size=16384,
            dim=896,
            n_layers=16,
            n_heads=14,
            n_kv_heads=2,
            max_seq_len=512,
            tie_embeddings=True,
        ),
    },
    "arm3_wide": {
        "name": "Arm 3: Wide Memory (L=12, d=1024)",
        "config": ModelConfig(
            vocab_size=16384,
            dim=1024,
            n_layers=12,
            n_heads=16,
            n_kv_heads=4,
            max_seq_len=512,
            tie_embeddings=True,
        ),
    },
}


def evaluate_val_loss(model: TransformerDecoder, val_loader: DataLoader, max_eval_steps: int = 50, device: str = "cuda") -> float:
    """Computes exact validation loss across held-out tokens."""
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            total_loss += loss.item()
            steps += 1
            if steps >= max_eval_steps:
                break
    return total_loss / max(1, steps)


def measure_generation_latency(
    model: TransformerDecoder,
    tokenizer: CodeTokenizer,
    prompt: str = "def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s is palindrome.\"\"\"\n",
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Measures autoregressive generation speed (tokens/sec) and syntax validity."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Warmup
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x[:, -model.config.max_seq_len:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            if next_token.item() == tokenizer.eos_id:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated_ids = x[0].tolist()[len(tokens):]
    num_generated = len(generated_ids)
    gen_tok_per_sec = num_generated / max(1e-5, elapsed)
    completion_text = tokenizer.decode(generated_ids)

    # Syntax test
    import ast
    is_valid_syntax = False
    try:
        ast.parse(prompt + completion_text)
        is_valid_syntax = True
    except Exception:
        pass

    return {
        "num_generated": num_generated,
        "elapsed_sec": elapsed,
        "tokens_per_sec": gen_tok_per_sec,
        "ms_per_token": (elapsed / max(1, num_generated)) * 1000,
        "is_valid_syntax": is_valid_syntax,
        "completion": completion_text.strip()[:100],
    }


def train_single_arm(
    arm_key: str,
    arm_info: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CodeTokenizer,
    total_steps: int = 1000,
    lr: float = 1e-3,
    save_dir: str = "checkpoints/sweep",
    device: str = "cuda",
) -> Dict[str, Any]:
    name = arm_info["name"]
    cfg = arm_info["config"]

    print(f"\n{'='*75}")
    print(f"STARTING SWEEP ARM: {name}")
    print(f"{'='*75}")

    model = TransformerDecoder(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e6:.2f} Million")

    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=lr)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="wsd", warmup_steps=50, total_steps=total_steps)

    model.train()
    data_iter = iter(train_loader)

    t_start = time.time()
    loss_history = []
    tokens_processed = 0

    for step in range(1, total_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        tokens_processed += x.numel()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % 100 == 0 or step == total_steps:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:4d}/{total_steps} | Loss: {loss_val:.4f} | LR: {current_lr:.6f} | Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    train_elapsed = time.time() - t_start
    train_tokens_per_sec = tokens_processed / max(1e-5, train_elapsed)

    # Final Validation Evaluation
    val_loss = evaluate_val_loss(model, val_loader, max_eval_steps=50, device=device)
    val_ppl = math.exp(min(val_loss, 20.0))
    # Bits Per Byte calculation: BPB = (Loss / ln 2) * (Tokens / Bytes)
    # On Mirendil-16k, Bytes/Token = 3.13 => Tokens/Byte = 1 / 3.13 = 0.3195
    tokens_per_byte = 1.0 / 3.13
    val_bpb = (val_loss / math.log(2)) * tokens_per_byte

    # Generation Latency Test
    gen_metrics = measure_generation_latency(model, tokenizer, device=device)

    # Save checkpoint
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{arm_key}.pt"
    torch.save({
        "config": cfg,
        "step": total_steps,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "state_dict": model.state_dict(),
    }, ckpt_path)

    results = {
        "arm_key": arm_key,
        "name": name,
        "layers": cfg.n_layers,
        "dim": cfg.dim,
        "params_m": param_count / 1e6,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "val_bpb": val_bpb,
        "train_tokens_per_sec": train_tokens_per_sec,
        "gen_tokens_per_sec": gen_metrics["tokens_per_sec"],
        "gen_ms_per_token": gen_metrics["ms_per_token"],
        "syntax_pass": gen_metrics["is_valid_syntax"],
        "train_time_sec": train_elapsed,
        "checkpoint": str(ckpt_path),
    }

    print(f"\n--> {name} COMPLETED:")
    print(f"    Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val BPB: {val_bpb:.3f}")
    print(f"    Decoding Speed: {gen_metrics['tokens_per_sec']:.1f} tok/s ({gen_metrics['ms_per_token']:.2f} ms/token)")
    print(f"    Syntax Valid:   {gen_metrics['is_valid_syntax']}")
    return results


def run_sweep(
    train_shard: str = "data/shards_16k/train_shard_00000.bin",
    val_shard: str = "data/shards_16k/val_shard_00000.bin",
    tokenizer_path: str = "checkpoints/tokenizers/mirendil_16k.json",
    steps_per_arm: int = 1000,
    batch_size: int = 32,
    seq_len: int = 512,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*75}")
    print(f"PILLAR 1 ARCHITECTURE SWEEP ON {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Steps Per Arm: {steps_per_arm:,} | Batch Size: {batch_size} | Seq Len: {seq_len}")
    print(f"Train Shard:   {train_shard}")
    print(f"{'='*75}\n")

    tokenizer = CodeTokenizer(tokenizer_path)
    train_dataset = PackedMemmapDataset([train_shard], seq_len=seq_len)
    val_dataset = PackedMemmapDataset([val_shard], seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    all_results = []
    for arm_key, arm_info in CANDIDATE_ARMS.items():
        res = train_single_arm(
            arm_key=arm_key,
            arm_info=arm_info,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            total_steps=steps_per_arm,
            device=device,
        )
        all_results.append(res)

    # Save summary
    summary_path = Path("checkpoints/sweep/sweep_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print Final Scorecard
    print(f"\n{'='*95}")
    print("FINAL PILLAR 1 ARCHITECTURE SWEEP SCORECARD")
    print(f"{'='*95}")
    header = f"{'Architecture':<35} | {'Params':<8} | {'Val Loss':<9} | {'Val BPB':<8} | {'Decode tok/s':<13} | {'Train tok/s':<11}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['name']:<35} | {r['params_m']:<6.1f} M | {r['val_loss']:<9.4f} | {r['val_bpb']:<8.3f} | {r['gen_tokens_per_sec']:<13.1f} | {r['train_tokens_per_sec']:<11.0f}")
    print(f"{'='*95}\n")
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per arm")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    run_sweep(steps_per_arm=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
