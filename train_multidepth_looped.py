#!/usr/bin/env python3
"""
Mirendil-84M Multi-Depth Looped Transformer Pretraining Script
Hardware Target: NVIDIA DGX Grace Blackwell GB10 (10.0.0.133)

Trains an 8-layer physical Looped Transformer with Deep Multi-Exit Supervision
across T = {1, 2, 3, 4} recurrence loops (8, 16, 24, 32 virtual layers).
Calibrates all depth exits simultaneously to eliminate over-cooking and enable
adaptive test-time compute.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

# Add project root
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.model.transformer import ModelConfig
from src.model.looped_transformer import LoopedTransformerDecoder
from src.data.dataset import PackedMemmapDataset
from src.engine.optim import configure_optimizers, get_lr_scheduler
from generate import top_p_sampling


def evaluate_all_depths(model: nn.Module, val_loader: DataLoader, max_eval_steps: int = 30, device: str = "cuda") -> dict:
    """Evaluates cross-entropy loss and BPB across all 4 loop depths (T=1, 2, 3, 4)."""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for loops in [1, 2, 3, 4]:
            total_loss = 0.0
            steps = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, targets=y, num_loops=loops)
                total_loss += loss.item()
                steps += 1
                if steps >= max_eval_steps:
                    break
            avg_loss = total_loss / max(1, steps)
            bpb = (avg_loss / math.log(2)) * (1.0 / 3.08)
            results[f"T={loops}"] = {"loss": avg_loss, "bpb": bpb}
            
    return results


def benchmark_decode_speed(model: nn.Module, tok: Tokenizer, prompt: str = "def quicksort(arr):", max_tokens: int = 64, device: str = "cuda") -> dict:
    """Measures autoregressive decoding throughput (tokens/second) across all depths."""
    model.eval()
    enc = tok.encode(prompt)
    speeds = {}
    
    for loops in [1, 2, 3, 4]:
        x = torch.tensor([enc.ids], dtype=torch.long, device=device)
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                logits, _ = model(x, num_loops=loops)
                tok_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, tok_next], dim=1)
                
        # Timed run
        x = torch.tensor([enc.ids], dtype=torch.long, device=device)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits, _ = model(x, num_loops=loops)
                tok_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, tok_next], dim=1)
        elapsed = time.time() - t0
        speeds[f"T={loops}"] = max_tokens / max(0.001, elapsed)
        
    return speeds


def train_multidepth(
    total_steps: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    depth_weights: list = [0.20, 0.40, 0.25, 0.15],
    eval_interval: int = 100,
    use_wandb: bool = False,
    wandb_project: str = "rsi-coding-pretrain",
    wandb_name: str = None,
    device: str = "cuda",
):
    print("=" * 85, flush=True)
    print("🚀 MIRENDIL-84M MULTI-DEPTH LOOPED PRETRAINING (T = 1, 2, 3, 4)", flush=True)
    print(f"Target Hardware: NVIDIA DGX Grace Blackwell GB10 | Total Steps: {total_steps}", flush=True)
    if use_wandb:
        print(f"Weights & Biases: Active -> {wandb_project} ({wandb_name})", flush=True)
    print("=" * 85, flush=True)

    # 1. Datasets
    train_shard = "data/shards_16k/train_shard_00000.bin"
    val_shard = "data/shards_16k/val_shard_00000.bin"
    tok_path = "checkpoints/tokenizers/mirendil_16k.json"

    train_ds = PackedMemmapDataset([train_shard], seq_len=512)
    val_ds = PackedMemmapDataset([val_shard], seq_len=512)
    tok = Tokenizer.from_file(tok_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # 2. Model Configuration
    cfg = ModelConfig(
        vocab_size=16384,
        dim=896,
        n_layers=8,         # 8 Physical Layers in Silicon
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=512,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    model = LoopedTransformerDecoder(cfg, default_loops=2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"Physical Parameters:   {total_params / 1e6:.2f} Million ({non_emb_params / 1e6:.2f}M non-emb)", flush=True)
    print(f"Virtual Circuit Depth: 8 to 32 Layers (Dynamic T=1, 2, 3, 4)", flush=True)
    print(f"Loss Supervision Mix:  T=1 ({depth_weights[0]*100:.0f}%) | T=2 ({depth_weights[1]*100:.0f}%) | T=3 ({depth_weights[2]*100:.0f}%) | T=4 ({depth_weights[3]*100:.0f}%)\n", flush=True)

    # W&B Initialization
    if use_wandb:
        import wandb
        run_name = wandb_name or f"mirendil-84m-multidepth-{total_steps}s"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "total_steps": total_steps,
                "batch_size": batch_size,
                "lr": lr,
                "vocab_size": cfg.vocab_size,
                "dim": cfg.dim,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads,
                "depth_weights": depth_weights,
            }
        )

    # 3. Optimizer & Scheduler
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=lr)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="cosine", warmup_steps=int(total_steps * 0.1), total_steps=total_steps)

    model.train()
    data_iter = iter(train_loader)
    tokens_processed = 0
    t0 = time.time()

    print(f"Starting Multi-Depth Pretraining Loop ({total_steps} steps)...", flush=True)
    
    best_t2_bpb = float("inf")
    checkpoint_dir = Path("checkpoints/champion")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "mirendil_84m_multidepth.pt"

    for step in range(1, total_steps + 1):
        step_t0 = time.time()
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        tokens_processed += x.numel()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Compute multi-depth loss across all 4 loop exits in a single pass!
            _, total_loss = model(x, targets=y, depth_weights=depth_weights)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step_elapsed = time.time() - step_t0
        step_tok_s = x.numel() / max(0.001, step_elapsed)

        # Logging
        if step % 25 == 0 or step == total_steps:
            lr_cur = scheduler.get_last_lr()[0]
            bpb_train = (total_loss.item() / math.log(2)) * (1.0 / 3.08)
            print(f"  Step {step:4d}/{total_steps} | Total Loss: {total_loss.item():.4f} | BPB: {bpb_train:.3f} | LR: {lr_cur:.6f} | Step: {step_elapsed:.2f}s ({step_tok_s:.0f} tok/s)", flush=True)

        if use_wandb and (step % 5 == 0 or step == total_steps):
            lr_cur = scheduler.get_last_lr()[0]
            bpb_train = (total_loss.item() / math.log(2)) * (1.0 / 3.08)
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/bpb": bpb_train,
                "train/lr": lr_cur,
                "train/tok_s": step_tok_s,
            }, step=step)

        # Periodic Multi-Depth Evaluation
        if step % eval_interval == 0 or step == total_steps:
            print(f"\n--- [Step {step} Multi-Depth Validation] ---", flush=True)
            val_res = evaluate_all_depths(model, val_loader, max_eval_steps=30, device=device)
            print(f"  T=1 (8 Virtual L):   Loss {val_res['T=1']['loss']:.4f} | BPB {val_res['T=1']['bpb']:.3f}", flush=True)
            print(f"  T=2 (16 Virtual L):  Loss {val_res['T=2']['loss']:.4f} | BPB {val_res['T=2']['bpb']:.3f}", flush=True)
            print(f"  T=3 (24 Virtual L):  Loss {val_res['T=3']['loss']:.4f} | BPB {val_res['T=3']['bpb']:.3f}", flush=True)
            print(f"  T=4 (32 Virtual L):  Loss {val_res['T=4']['loss']:.4f} | BPB {val_res['T=4']['bpb']:.3f}", flush=True)
            
            if use_wandb:
                wandb.log({
                    "val/T1_loss": val_res['T=1']['loss'],
                    "val/T1_bpb": val_res['T=1']['bpb'],
                    "val/T2_loss": val_res['T=2']['loss'],
                    "val/T2_bpb": val_res['T=2']['bpb'],
                    "val/T3_loss": val_res['T=3']['loss'],
                    "val/T3_bpb": val_res['T=3']['bpb'],
                    "val/T4_loss": val_res['T=4']['loss'],
                    "val/T4_bpb": val_res['T=4']['bpb'],
                }, step=step)

            # Checkpoint on best T=2 BPB
            if val_res['T=2']['bpb'] < best_t2_bpb:
                best_t2_bpb = val_res['T=2']['bpb']
                torch.save(model.state_dict(), save_path)
                print(f"  ⭐ New Best Checkpoint Saved to {save_path} (T=2 BPB: {best_t2_bpb:.3f})", flush=True)
            print("-------------------------------------------\n", flush=True)
            model.train()

    total_time = time.time() - t0
    avg_speed = tokens_processed / total_time
    print(f"\n Pretraining Complete in {total_time:.1f}s ({total_time/60:.1f} mins) | Overall Speed: {avg_speed:.0f} tok/s", flush=True)

    # Final Decode Speed Benchmark
    print("\n Benchmarking Autoregressive Decode Speed across all depths...", flush=True)
    speeds = benchmark_decode_speed(model, tok, device=device)
    for depth, spd in speeds.items():
        print(f"  {depth}: {spd:.1f} tokens/second", flush=True)

    # Final Comprehensive Scorecard
    final_val = evaluate_all_depths(model, val_loader, max_eval_steps=40, device=device)
    print("\n" + "=" * 90, flush=True)
    print("FINAL MULTI-DEPTH SCORECARD (NVIDIA DGX GRACE BLACKWELL)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Loop Mode':<18} | {'Virtual Depth':<14} | {'Val Loss':<10} | {'Val BPB':<10} | {'Decode Speed':<14}", flush=True)
    print("-" * 75, flush=True)
    for depth in ["T=1", "T=2", "T=3", "T=4"]:
        v_depth = {"T=1": "8 Layers", "T=2": "16 Layers", "T=3": "24 Layers", "T=4": "32 Layers"}[depth]
        print(f"{depth:<18} | {v_depth:<14} | {final_val[depth]['loss']:<10.4f} | {final_val[depth]['bpb']:<10.3f} | {speeds[depth]:<8.1f} tok/s", flush=True)
    print("=" * 90 + "\n", flush=True)

    if use_wandb:
        for depth in ["T=1", "T=2", "T=3", "T=4"]:
            wandb.summary[f"final/{depth}_loss"] = final_val[depth]['loss']
            wandb.summary[f"final/{depth}_bpb"] = final_val[depth]['bpb']
            wandb.summary[f"final/{depth}_decode_tok_s"] = speeds[depth]
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry")
    parser.add_argument("--wandb_project", type=str, default="rsi-coding-pretrain", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    train_multidepth(
        total_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_interval=args.eval_interval,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
