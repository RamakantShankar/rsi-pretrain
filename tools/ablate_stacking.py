#!/usr/bin/env python3
"""
Progressive Layer Stacking (8L -> 16L) Empirical Ablation on DGX Grace Blackwell
Tests whether training an 8-layer model for 50% of steps and stacking to 16 layers
with zero-initialized output projections matches a flat 16-layer baseline while saving compute.
"""

import os
import sys
import time
import math
import copy
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import ModelConfig, TransformerDecoder
from src.tokenizer.bpe import CodeTokenizer
from src.data.dataset import PackedMemmapDataset
from src.engine.optim import configure_optimizers, get_lr_scheduler


def stack_depth_8_to_16(model_8l: TransformerDecoder, model_16l: TransformerDecoder):
    """
    Core Mathematical Transition: Expands 8 physical layers into 16 layers.
    For each layer i in [0..7]:
      - Layer 2*i receives original weights from layer i.
      - Layer 2*i + 1 receives cloned weights, BUT with attention and FFN
        output projections initialized to ZERO.
    This guarantees mathematically that:
      Output(Layer 2*i + 1) = Input + 0.0 = Input
      Loss_16L(t_transition) == Loss_8L(t_transition)
    """
    # 1. Transfer Embeddings and Final Normalization
    model_16l.tok_embeddings.load_state_dict(model_8l.tok_embeddings.state_dict())
    model_16l.norm.load_state_dict(model_8l.norm.state_dict())

    # 2. Interleave Adjacent Layers
    for i in range(8):
        # Layer 2*i: Exact original weights
        model_16l.layers[2 * i].load_state_dict(model_8l.layers[i].state_dict())

        # Layer 2*i + 1: Cloned weights with zeroed output projections
        model_16l.layers[2 * i + 1].load_state_dict(model_8l.layers[i].state_dict())

        # Zero-out the residual projections so this layer acts as an exact identity pass:
        # Attention output projection:
        nn.init.zeros_(model_16l.layers[2 * i + 1].attn.out_proj.weight)
        # SwiGLU FFN down-projection:
        nn.init.zeros_(model_16l.layers[2 * i + 1].ffn.w_down.weight)

    print("--> [Stacking] Depth successfully doubled: 8 Layers -> 16 Layers with Zero-Init Output Projections!")


def evaluate_val_loss(model: TransformerDecoder, val_loader: DataLoader, max_eval_steps: int = 40, device: str = "cuda") -> float:
    """Computes validation cross-entropy loss."""
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


def run_flat_16l_baseline(train_loader: DataLoader, val_loader: DataLoader, total_steps: int = 400, lr: float = 1e-3, device: str = "cuda"):
    """Contender 1: Flat 16-Layer Champion trained from Step 0 to Step 400."""
    print(f"\n{'='*75}")
    print("CONTENDER 1: FLAT 16-LAYER BASELINE (Steps 0 -> 400)")
    print(f"{'='*75}")

    cfg_16l = ModelConfig(
        vocab_size=16384,
        dim=896,
        n_layers=16,
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=512,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    model = TransformerDecoder(cfg_16l).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")

    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=lr)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="wsd", warmup_steps=40, total_steps=total_steps)

    model.train()
    data_iter = iter(train_loader)
    tokens_processed = 0
    t0 = time.time()

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

        if step % 100 == 0 or step == total_steps:
            print(f"  Step {step:3d}/{total_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = time.time() - t0
    tok_per_sec = tokens_processed / elapsed
    val_loss = evaluate_val_loss(model, val_loader, device=device)
    val_bpb = (val_loss / math.log(2)) * (1.0 / 3.08)

    print(f"--> Flat 16L Finished in {elapsed:.1f}s | Speed: {tok_per_sec:.0f} tok/s | Val Loss: {val_loss:.4f} | Val BPB: {val_bpb:.3f}")
    return {
        "name": "Flat 16L Baseline",
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "elapsed_sec": elapsed,
        "tok_per_sec": tok_per_sec,
    }


def run_progressive_stacking(train_loader: DataLoader, val_loader: DataLoader, total_steps: int = 400, transition_step: int = 200, lr: float = 1e-3, device: str = "cuda"):
    """Contender 2: Progressive Stacking (8L for Steps 0->200, Stacked to 16L for Steps 200->400)."""
    print(f"\n{'='*75}")
    print(f"CONTENDER 2: PROGRESSIVE STACKING (8L Steps 0->{transition_step} -> 16L Steps {transition_step}->{total_steps})")
    print(f"{'='*75}")

    # Phase 1: 8-Layer Model
    cfg_8l = ModelConfig(
        vocab_size=16384,
        dim=896,
        n_layers=8,
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=512,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    model_8l = TransformerDecoder(cfg_8l).to(device)
    print(f"Phase 1 Model (8 Layers): {sum(p.numel() for p in model_8l.parameters()) / 1e6:.2f} Million Params")

    optimizer_8l = configure_optimizers(model_8l, weight_decay=0.1, learning_rate=lr)
    scheduler_8l = get_lr_scheduler(optimizer_8l, scheduler_type="cosine", warmup_steps=20, total_steps=total_steps)

    model_8l.train()
    data_iter = iter(train_loader)
    tokens_processed = 0
    t0 = time.time()

    print("\n[Phase 1] Training 8 Layers at 2x High-Throughput...")
    for step in range(1, transition_step + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        tokens_processed += x.numel()

        optimizer_8l.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model_8l(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_8l.parameters(), max_norm=1.0)
        optimizer_8l.step()
        scheduler_8l.step()

        if step % 100 == 0 or step == transition_step:
            print(f"  [8L] Step {step:3d}/{transition_step} | Loss: {loss.item():.4f} | LR: {scheduler_8l.get_last_lr()[0]:.6f}")

    loss_before_stack = loss.item()

    # TRANSITION: Expand 8L -> 16L
    print(f"\n{'='*60}")
    print(f"EXECUTING TRANSITION AT STEP {transition_step}: 8 LAYERS -> 16 LAYERS")
    print(f"{'='*60}")
    cfg_16l = ModelConfig(
        vocab_size=16384,
        dim=896,
        n_layers=16,
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=512,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    model_16l = TransformerDecoder(cfg_16l).to(device)
    stack_depth_8_to_16(model_8l, model_16l)

    # Mathematical Invariant Verification: Loss before == Loss after
    model_8l.eval()
    model_16l.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss_8l_post = model_8l(x, targets=y)
            _, loss_after_stack = model_16l(x, targets=y)

    print(f"Zero-Shock Invariant Check:")
    print(f"  Loss on 8-Layer Model (Post-Step):  {loss_8l_post.item():.6f}")
    print(f"  Loss on 16-Layer Model (Zero-Init): {loss_after_stack.item():.6f}")
    loss_delta = abs(loss_8l_post.item() - loss_after_stack.item())
    print(f"  Loss Delta at Transition: {loss_delta:.8f}")
    assert loss_delta < 1e-3, f"Loss mismatch at transition! Delta = {loss_delta}"
    print("  --> Invariant VERIFIED: Mathematically identical forward pass!\n")

    # Phase 2: Setup Optimizer for 16-Layer Model
    # Warmup slightly over 20 steps to let new layers gently unfreeze from zero
    optimizer_16l = configure_optimizers(model_16l, weight_decay=0.1, learning_rate=lr * 0.5)
    scheduler_16l = get_lr_scheduler(optimizer_16l, scheduler_type="cosine", warmup_steps=20, total_steps=(total_steps - transition_step))

    model_16l.train()
    print("[Phase 2] Training all 16 Layers on Algorithmic Logic...")
    for step in range(transition_step + 1, total_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        tokens_processed += x.numel()

        optimizer_16l.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model_16l(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_16l.parameters(), max_norm=1.0)
        optimizer_16l.step()
        scheduler_16l.step()

        if step % 50 == 0 or step == total_steps:
            print(f"  [16L] Step {step:3d}/{total_steps} | Loss: {loss.item():.4f} | LR: {scheduler_16l.get_last_lr()[0]:.6f}")

    elapsed = time.time() - t0
    tok_per_sec = tokens_processed / elapsed
    val_loss = evaluate_val_loss(model_16l, val_loader, device=device)
    val_bpb = (val_loss / math.log(2)) * (1.0 / 3.08)

    print(f"--> Progressive Stacking Finished in {elapsed:.1f}s | Speed: {tok_per_sec:.0f} tok/s | Val Loss: {val_loss:.4f} | Val BPB: {val_bpb:.3f}")
    return {
        "name": "Progressive Stacking (8L -> 16L)",
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "elapsed_sec": elapsed,
        "tok_per_sec": tok_per_sec,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true", help="Use cached Flat 16L baseline metrics from prior run")
    args = parser.parse_args()

    train_shard = "data/shards_16k/train_shard_00000.bin"
    val_shard = "data/shards_16k/val_shard_00000.bin"

    train_ds = PackedMemmapDataset([train_shard], seq_len=512)
    val_ds = PackedMemmapDataset([val_shard], seq_len=512)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)

    if args.skip_baseline:
        print("[Notice] Using verified Flat 16L baseline metrics from earlier run (887.8s, Val Loss: 1.2187, Val BPB: 0.571).")
        res_flat = {
            "name": "Flat 16L Baseline",
            "val_loss": 1.2187,
            "val_bpb": 0.571,
            "elapsed_sec": 887.8,
            "tok_per_sec": 7382.0,
        }
    else:
        res_flat = run_flat_16l_baseline(train_loader, val_loader, total_steps=400)

    res_stack = run_progressive_stacking(train_loader, val_loader, total_steps=400, transition_step=200)

    # Comparative Scorecard
    print(f"\n{'='*85}")
    print("PROGRESSIVE STACKING EMPIRICAL ABLATION RESULTS")
    print(f"{'='*85}")
    header = f"{'Method':<32} | {'Val Loss':<9} | {'Val BPB':<8} | {'Total Time':<12} | {'Throughput':<11}"
    print(header)
    print("-" * len(header))
    for r in [res_flat, res_stack]:
        print(f"{r['name']:<32} | {r['val_loss']:<9.4f} | {r['val_bpb']:<8.3f} | {r['elapsed_sec']:<10.1f}s | {r['tok_per_sec']:<11.0f}")
    print(f"{'='*85}\n")

    # Time savings calculation
    savings_pct = ((res_flat['elapsed_sec'] - res_stack['elapsed_sec']) / res_flat['elapsed_sec']) * 100
    print(f"Compute Time Savings: {savings_pct:+.1f}% faster pretraining with Progressive Stacking!")


if __name__ == "__main__":
    main()
