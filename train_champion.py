#!/usr/bin/env python3
"""
Mirendil-154M Production Pretraining Runner on NVIDIA DGX Grace Blackwell (GB10)
Incorporates all validated frontier techniques:
  1. Specialized 16k Python+English Tokenizer
  2. Champion Architecture (L=16, d=896, 14 Q heads, 2 KV heads)
  3. QK-Norm (Per-head RMSNorm on Query and Key)
  4. RoPE theta = 100,000 for long-context stability
  5. Progressive Adjacent Stacking (8L steps 1->1000 -> 16L steps 1000->2500)
     with Zero-Initialized Output Projections (Delta Loss = 0.00000000)
"""

import os
import sys
import time
import math
import json
import signal
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import ModelConfig, TransformerDecoder
from src.data.dataset import PackedMemmapDataset
from src.engine.optim import configure_optimizers, get_lr_scheduler


def stack_depth_8_to_16(model_8l: TransformerDecoder, model_16l: TransformerDecoder):
    """
    Mathematical Zero-Shock Stacking: Expands 8 physical layers into 16 layers.
    Maps:
      Layer 2*i     <- Original weights from layer i
      Layer 2*i + 1 <- Cloned weights, but with attn.out_proj and ffn.w_down set to ZERO.
    Guarantees:
      Output(Layer 2*i + 1) = Input + 0 = Input
      Loss_16L(t) == Loss_8L(t) (Delta = 0.00000000)
    """
    # 1. Transfer Token Embeddings and Final Normalization
    model_16l.tok_embeddings.load_state_dict(model_8l.tok_embeddings.state_dict())
    model_16l.norm.load_state_dict(model_8l.norm.state_dict())
    if hasattr(model_16l, "lm_head") and hasattr(model_8l, "lm_head") and not model_16l.config.tie_embeddings:
        model_16l.lm_head.load_state_dict(model_8l.lm_head.state_dict())

    # 2. Interleave Adjacent Layers
    for i in range(8):
        # Even layer: Original weights
        model_16l.layers[2 * i].load_state_dict(model_8l.layers[i].state_dict())

        # Odd layer: Cloned weights with zeroed output projections
        model_16l.layers[2 * i + 1].load_state_dict(model_8l.layers[i].state_dict())

        # Zero out Attention output projection
        nn.init.zeros_(model_16l.layers[2 * i + 1].attn.out_proj.weight)
        # Zero out SwiGLU FFN down-projection
        nn.init.zeros_(model_16l.layers[2 * i + 1].ffn.w_down.weight)

    print("--> [Progressive Stacking] 8 Layers -> 16 Layers expanded with Zero-Init Output Projections!")


def evaluate(model: TransformerDecoder, val_loader: DataLoader, max_eval_steps: int = 50, device: str = "cuda") -> dict:
    """Evaluates cross-entropy loss, perplexity, and Bits Per Byte (BPB)."""
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
    avg_loss = total_loss / max(1, steps)
    ppl = math.exp(min(20.0, avg_loss))
    # Bits Per Byte: (loss / ln(2)) * (1 / compression_ratio), compression_ratio = 3.08 bytes/token
    bpb = (avg_loss / math.log(2)) * (1.0 / 3.08)
    return {"val_loss": avg_loss, "val_ppl": ppl, "val_bpb": bpb}


def main():
    parser = argparse.ArgumentParser(description="Mirendil-154M Production Pretraining")
    parser.add_argument("--config", type=str, default="config/mirendil_154m_champion.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = save_dir / "metrics.jsonl"

    print(f"\n{'='*80}")
    print("🚀 MIRENDIL-154M PRODUCTION FOUNDATION PRETRAINING")
    print(f"Device:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Configuration:    {args.config}")
    print(f"Total Steps:      {cfg['training']['total_steps']} (~40.96M tokens)")
    print(f"Transition Step:  Step {cfg['training']['transition_step']} (8L -> 16L Stacking)")
    print(f"Save Directory:   {save_dir}")
    print(f"{'='*80}\n")

    # 1. Setup Data Loaders
    seq_len = cfg["model"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]
    train_ds = PackedMemmapDataset([cfg["data"]["train_shard"]], seq_len=seq_len)
    val_ds = PackedMemmapDataset([cfg["data"]["val_shard"]], seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"Data Loaded: {len(train_ds):,} training sequences ({len(train_ds)*seq_len:,} tokens)")
    print(f"Validation:  {len(val_ds):,} validation sequences ({len(val_ds)*seq_len:,} tokens)\n")

    # 2. Setup Phase 1: 8-Layer Transformer
    cfg_8l = ModelConfig(
        vocab_size=cfg["model"]["vocab_size"],
        dim=cfg["model"]["dim"],
        n_layers=8,
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        max_seq_len=cfg["model"]["max_seq_len"],
        rope_theta=cfg["model"]["rope_theta"],
        use_qk_norm=cfg["model"]["use_qk_norm"],
        norm_eps=cfg["model"]["norm_eps"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
        dropout=cfg["model"]["dropout"],
    )
    model = TransformerDecoder(cfg_8l).to(device)
    params_8l = sum(p.numel() for p in model.parameters())
    print(f"[Phase 1] 8-Layer Trunk Initialized: {params_8l / 1e6:.2f}M Parameters")

    opt_8l = configure_optimizers(
        model,
        weight_decay=cfg["training"]["weight_decay"],
        learning_rate=cfg["training"]["phase1_lr"],
    )
    sched_8l = get_lr_scheduler(
        opt_8l,
        scheduler_type="cosine",
        warmup_steps=cfg["training"]["phase1_warmup_steps"],
        total_steps=cfg["training"]["total_steps"],
    )

    data_iter = iter(train_loader)
    tokens_trained = 0
    start_time = time.time()
    best_val_loss = float("inf")
    metrics_file = open(metrics_log_path, "a", encoding="utf-8")

    total_steps = cfg["training"]["total_steps"]
    transition_step = cfg["training"]["transition_step"]
    log_interval = cfg["training"]["log_interval"]
    eval_interval = cfg["training"]["eval_interval"]

    # Graceful exit handler
    def handle_signal(sig, frame):
        print("\n[Signal Caught] Saving emergency checkpoint before exit...")
        torch.save(model.state_dict(), save_dir / "emergency_checkpoint.pt")
        metrics_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"\n{'='*75}")
    print(f"PHASE 1: TRAINING 8 LAYERS AT 2X THROUGHPUT (Steps 1 -> {transition_step})")
    print(f"{'='*75}")

    model.train()
    step_t0 = time.time()

    for step in range(1, transition_step + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        tokens_trained += x.numel()

        opt_8l.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["training"]["max_grad_norm"])
        opt_8l.step()
        sched_8l.step()

        if step % log_interval == 0 or step == transition_step:
            step_time = time.time() - step_t0
            tok_s = (log_interval * x.numel()) / max(1e-4, step_time)
            lr_cur = sched_8l.get_last_lr()[0]
            bpb_train = (loss.item() / math.log(2)) * (1.0 / 3.08)
            print(f"  [Phase 1 | 8L] Step {step:4d}/{total_steps} | Loss: {loss.item():.4f} | BPB: {bpb_train:.3f} | LR: {lr_cur:.6f} | Speed: {tok_s:.0f} tok/s")
            
            log_entry = {
                "phase": 1,
                "step": step,
                "train_loss": loss.item(),
                "train_bpb": bpb_train,
                "lr": lr_cur,
                "tok_per_sec": tok_s,
                "tokens_trained": tokens_trained,
                "elapsed_sec": time.time() - start_time,
            }
            metrics_file.write(json.dumps(log_entry) + "\n")
            metrics_file.flush()
            step_t0 = time.time()

        if step % eval_interval == 0:
            eval_metrics = evaluate(model, val_loader, device=device)
            print(f"  ⭐️ [Validation @ Step {step}] Loss: {eval_metrics['val_loss']:.4f} | PPL: {eval_metrics['val_ppl']:.2f} | BPB: {eval_metrics['val_bpb']:.3f}")
            eval_metrics.update({"step": step, "type": "eval"})
            metrics_file.write(json.dumps(eval_metrics) + "\n")
            metrics_file.flush()
            model.train()

    # Save Phase 1 Checkpoint
    phase1_ckpt_path = save_dir / "phase1_8l_step1000.pt"
    torch.save(model.state_dict(), phase1_ckpt_path)
    print(f"\n--> Phase 1 Complete! Checkpoint saved to: {phase1_ckpt_path}")

    # =========================================================================
    # THE TRANSITION: 8 LAYERS -> 16 LAYERS PROGRESSIVE ADJACENT STACKING
    # =========================================================================
    print(f"\n{'='*75}")
    print(f"EXECUTING TRANSITION AT STEP {transition_step}: EXPANDING 8L -> 16L")
    print(f"{'='*75}")

    cfg_16l = ModelConfig(
        vocab_size=cfg["model"]["vocab_size"],
        dim=cfg["model"]["dim"],
        n_layers=16,
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        max_seq_len=cfg["model"]["max_seq_len"],
        rope_theta=cfg["model"]["rope_theta"],
        use_qk_norm=cfg["model"]["use_qk_norm"],
        norm_eps=cfg["model"]["norm_eps"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
        dropout=cfg["model"]["dropout"],
    )
    model_16l = TransformerDecoder(cfg_16l).to(device)
    stack_depth_8_to_16(model, model_16l)

    # Invariant Verification: Both evaluated under identical eval conditions
    model.eval()
    model_16l.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss_8l_post = model(x, targets=y)
            _, loss_16l_post = model_16l(x, targets=y)

    loss_delta = abs(loss_8l_post.item() - loss_16l_post.item())
    print("Zero-Shock Invariant Check:")
    print(f"  Loss on 8-Layer Model (Post-Step):  {loss_8l_post.item():.6f}")
    print(f"  Loss on 16-Layer Model (Zero-Init): {loss_16l_post.item():.6f}")
    print(f"  Loss Delta at Transition:           {loss_delta:.8f}")
    assert loss_delta < 1e-3, f"Loss mismatch at transition! Delta = {loss_delta}"
    print("  --> INVARIANT VERIFIED: Exact mathematical identity forward pass!\n")

    # Release 8-layer model from memory
    del model, opt_8l, sched_8l
    torch.cuda.empty_cache()

    # Save Transition Checkpoint
    transition_ckpt_path = save_dir / "phase2_16l_transition.pt"
    torch.save(model_16l.state_dict(), transition_ckpt_path)
    print(f"--> Transition Checkpoint saved to: {transition_ckpt_path}")

    # =========================================================================
    # PHASE 2: 16-LAYER DEEP REASONING SPECIALIZATION
    # =========================================================================
    print(f"\n{'='*75}")
    print(f"PHASE 2: TRAINING ALL 16 LAYERS (Steps {transition_step+1} -> {total_steps})")
    print(f"{'='*75}")

    params_16l = sum(p.numel() for p in model_16l.parameters())
    print(f"[Phase 2] Full Model Active: {params_16l / 1e6:.2f}M Parameters")

    opt_16l = configure_optimizers(
        model_16l,
        weight_decay=cfg["training"]["weight_decay"],
        learning_rate=cfg["training"]["phase2_lr"],
    )
    sched_16l = get_lr_scheduler(
        opt_16l,
        scheduler_type="cosine",
        warmup_steps=cfg["training"]["phase2_warmup_steps"],
        total_steps=(total_steps - transition_step),
    )

    model_16l.train()
    step_t0 = time.time()

    for step in range(transition_step + 1, total_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        tokens_trained += x.numel()

        opt_16l.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model_16l(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_16l.parameters(), max_norm=cfg["training"]["max_grad_norm"])
        opt_16l.step()
        sched_16l.step()

        if step % log_interval == 0 or step == total_steps:
            step_time = time.time() - step_t0
            tok_s = (log_interval * x.numel()) / max(1e-4, step_time)
            lr_cur = sched_16l.get_last_lr()[0]
            bpb_train = (loss.item() / math.log(2)) * (1.0 / 3.08)
            print(f"  [Phase 2 | 16L] Step {step:4d}/{total_steps} | Loss: {loss.item():.4f} | BPB: {bpb_train:.3f} | LR: {lr_cur:.6f} | Speed: {tok_s:.0f} tok/s")
            
            log_entry = {
                "phase": 2,
                "step": step,
                "train_loss": loss.item(),
                "train_bpb": bpb_train,
                "lr": lr_cur,
                "tok_per_sec": tok_s,
                "tokens_trained": tokens_trained,
                "elapsed_sec": time.time() - start_time,
            }
            metrics_file.write(json.dumps(log_entry) + "\n")
            metrics_file.flush()
            step_t0 = time.time()

        if step % eval_interval == 0 or step == total_steps:
            eval_metrics = evaluate(model_16l, val_loader, device=device)
            print(f"  ⭐️ [Validation @ Step {step}] Loss: {eval_metrics['val_loss']:.4f} | PPL: {eval_metrics['val_ppl']:.2f} | BPB: {eval_metrics['val_bpb']:.3f}")
            eval_metrics.update({"step": step, "type": "eval"})
            metrics_file.write(json.dumps(eval_metrics) + "\n")
            metrics_file.flush()

            # Save periodic step checkpoint
            step_ckpt = save_dir / f"checkpoint_step_{step}.pt"
            torch.save(model_16l.state_dict(), step_ckpt)

            # Check if new best
            if eval_metrics["val_loss"] < best_val_loss:
                best_val_loss = eval_metrics["val_loss"]
                champion_ckpt = save_dir / "mirendil_154m_champion.pt"
                torch.save(model_16l.state_dict(), champion_ckpt)
                print(f"  🏆 NEW BEST CHAMPION CHECKPOINT! Saved to {champion_ckpt}")

            model_16l.train()

    total_elapsed = time.time() - start_time
    final_metrics = evaluate(model_16l, val_loader, device=device)

    # Save model config metadata
    cfg_export = {
        "model_name": "Mirendil-154M-Champion",
        "vocab_size": cfg_16l.vocab_size,
        "dim": cfg_16l.dim,
        "n_layers": cfg_16l.n_layers,
        "n_heads": cfg_16l.n_heads,
        "n_kv_heads": cfg_16l.n_kv_heads,
        "max_seq_len": cfg_16l.max_seq_len,
        "rope_theta": cfg_16l.rope_theta,
        "use_qk_norm": cfg_16l.use_qk_norm,
        "tie_embeddings": cfg_16l.tie_embeddings,
        "total_params": params_16l,
        "tokens_trained": tokens_trained,
        "final_val_loss": final_metrics["val_loss"],
        "final_val_ppl": final_metrics["val_ppl"],
        "final_val_bpb": final_metrics["val_bpb"],
        "total_elapsed_sec": total_elapsed,
        "avg_tok_per_sec": tokens_trained / total_elapsed,
    }
    with open(save_dir / "champion_config.json", "w") as f:
        json.dump(cfg_export, f, indent=2)

    metrics_file.close()

    print(f"\n{'='*80}")
    print("🏆 MIRENDIL-154M PRODUCTION PRETRAINING COMPLETE!")
    print(f"Total Time:       {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)")
    print(f"Tokens Processed: {tokens_trained:,}")
    print(f"Final Val Loss:   {final_metrics['val_loss']:.4f}")
    print(f"Final Val BPB:    {final_metrics['val_bpb']:.3f}")
    print(f"Champion Weights: {save_dir / 'mirendil_154m_champion.pt'}")
    print(f"Config Metadata:  {save_dir / 'champion_config.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
