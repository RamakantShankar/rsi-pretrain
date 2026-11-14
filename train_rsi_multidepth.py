#!/usr/bin/env python3
"""
RSI Policy Fine-Tuning Runner (Generation M_0 -> M_1)
Trains the Multi-Depth Champion on the experience replay buffer using
Prompt-Masked Cross Entropy Loss (25% gold self-play & seed code + 75% baseline pretraining anchor).
Ensures prompt tokens are masked with ignore_index=-100 so the model learns purely to complete code.
Streams live metrics and loss breakdowns to Weights & Biases.
"""

import os
import sys
import time
import math
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from src.model.transformer import ModelConfig
from src.model.looped_transformer import LoopedTransformerDecoder
from src.engine.optim import configure_optimizers, get_lr_scheduler


def format_prompt(raw_prompt: str) -> str:
    """Ensures prompt ends with 4-space indentation to prime function body."""
    return raw_prompt.rstrip() + "\n    "


class PromptMaskedReplayDataset(Dataset):
    """
    Dataset that blends prompt-masked supervised coding pairs with baseline pretraining tokens.
    For supervised pairs: prompt tokens and padding have target = -100 (zero loss gradient).
    For baseline pretraining tokens: full sequence is supervised (preserves general language knowledge).
    """
    def __init__(
        self,
        verified_path: str,
        problem_bank_path: str,
        baseline_shard_path: str,
        tokenizer_path: str,
        seq_len: int = 512,
        replay_ratio: float = 0.25,
        total_samples: int = 4800,
    ):
        self.seq_len = seq_len
        self.replay_ratio = replay_ratio
        self.total_samples = total_samples

        self.tok = Tokenizer.from_file(tokenizer_path)
        self.eot_id = self.tok.token_to_id("<|endoftext|>")

        # 1. Load Supervised Coding Pool (Self-play verified + Canonical anchors)
        self.supervised_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        loaded_tasks = set()

        # A. Load verified self-play solutions
        if os.path.exists(verified_path):
            with open(verified_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        prompt = rec.get("prompt", "")
                        comp = rec.get("completion") or rec.get("solution") or ""
                        # If completion contains full code, strip prompt prefix
                        if prompt and comp.startswith(prompt.rstrip()):
                            comp = comp[len(prompt.rstrip()):].lstrip("\n ")
                        pair = self._build_masked_pair(prompt, comp)
                        if pair is not None:
                            self.supervised_pairs.append(pair)
                            loaded_tasks.add(rec.get("task_id", ""))

        print(f"--> Loaded {len(self.supervised_pairs)} verified self-play pairs ({len(loaded_tasks)} unique tasks).")

        # B. Load canonical seed solutions for coverage of all benchmark categories
        if os.path.exists(problem_bank_path):
            with open(problem_bank_path, "r", encoding="utf-8") as f:
                problems = json.load(f)
            canon_count = 0
            for prob in problems:
                prompt = format_prompt(prob["prompt"])
                canon = prob.get("canonical", "").strip()
                pair = self._build_masked_pair(prompt, canon)
                if pair is not None:
                    self.supervised_pairs.append(pair)
                    canon_count += 1
            print(f"--> Bootstrapped {canon_count} canonical anchor pairs across all 55 benchmark tasks.")

        print(f"--> Total Supervised Replay Pool: {len(self.supervised_pairs)} pairs.")
        assert len(self.supervised_pairs) > 0, "No supervised pairs available!"

        # 2. Setup Baseline Pretraining Memmap
        self.baseline_mmap = np.memmap(baseline_shard_path, dtype=np.uint32, mode="r")
        self.num_baseline_tokens = len(self.baseline_mmap)
        print(f"--> Baseline Pretraining Anchor Pool: {self.num_baseline_tokens:,} tokens ({baseline_shard_path}).")

    def _build_masked_pair(self, prompt: str, completion: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not prompt.strip() or not completion.strip():
            return None

        p_ids = self.tok.encode(prompt).ids
        c_ids = self.tok.encode(completion).ids + [self.eot_id]
        full_ids = p_ids + c_ids

        if len(full_ids) > self.seq_len + 1:
            full_ids = full_ids[:self.seq_len + 1]

        pad_len = (self.seq_len + 1) - len(full_ids)
        x_ids = full_ids[:-1] + [0] * pad_len
        y_ids = full_ids[1:] + [-100] * pad_len

        # Mask prompt tokens in y_ids
        # In y_ids, index i corresponds to full_ids[i+1].
        # For i < len(p_ids) - 1, full_ids[i+1] is a prompt token -> mask with -100
        # At i = len(p_ids) - 1, full_ids[i+1] is full_ids[len(p_ids)] == c_ids[0] (first completion token)
        for i in range(min(len(p_ids) - 1, len(y_ids))):
            y_ids[i] = -100

        x_t = torch.tensor(x_ids, dtype=torch.long)
        y_t = torch.tensor(y_ids, dtype=torch.long)
        return (x_t, y_t)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.replay_ratio:
            # Sample from supervised replay pool
            pair_idx = random.randint(0, len(self.supervised_pairs) - 1)
            return self.supervised_pairs[pair_idx]
        else:
            # Sample continuous chunk from baseline pretraining shard
            max_start = self.num_baseline_tokens - (self.seq_len + 1)
            start_pos = random.randint(0, max_start)
            chunk = self.baseline_mmap[start_pos : start_pos + self.seq_len + 1].astype(np.int64)
            x_t = torch.tensor(chunk[:-1], dtype=torch.long)
            y_t = torch.tensor(chunk[1:], dtype=torch.long)
            return (x_t, y_t)


def train_rsi_m1(
    base_checkpoint: str = "checkpoints/champion/mirendil_84m_multidepth.pt",
    verified_path: str = "data/rsi/verified_solutions_multidepth.jsonl",
    problem_bank_path: str = "data/rsi/problem_bank.json",
    baseline_shard: str = "data/shards_16k/train_shard_00000.bin",
    output_checkpoint: str = "checkpoints/champion/mirendil_84m_rsi_m1.pt",
    total_steps: int = 150,
    batch_size: int = 32,
    lr: float = 1.0e-4,
    replay_ratio: float = 0.25,
    use_wandb: bool = True,
    wandb_project: str = "rsi-coding-pretrain",
    wandb_name: str = "rsi_policy_update_m0_to_m1_masked",
    device: str = "cuda",
):
    print("=" * 80)
    print("🚀 RSI AUTONOMOUS POLICY FINE-TUNING (M_0 -> M_1) WITH PROMPT MASKING")
    print(f"Device:            {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Base Checkpoint:   {base_checkpoint}")
    print(f"Verified Solutions:{verified_path}")
    print(f"Output Model:      {output_checkpoint}")
    print(f"Steps:             {total_steps} | Batch Size: {batch_size} | LR: {lr:.2e}")
    print(f"Replay Ratio:      {replay_ratio*100:.0f}% Supervised / {(1-replay_ratio)*100:.0f}% Pretraining Anchor")
    print("=" * 80 + "\n")

    # 1. Dataset & DataLoader
    train_ds = PromptMaskedReplayDataset(
        verified_path=verified_path,
        problem_bank_path=problem_bank_path,
        baseline_shard_path=baseline_shard,
        tokenizer_path="checkpoints/tokenizers/mirendil_16k.json",
        seq_len=512,
        replay_ratio=replay_ratio,
        total_samples=total_steps * batch_size,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Model & Checkpoint
    cfg = ModelConfig(
        vocab_size=16384,
        dim=896,
        n_layers=8,
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=512,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    model = LoopedTransformerDecoder(cfg, default_loops=2).to(device)
    ckpt = torch.load(base_checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("--> Successfully loaded M_0 base weights into LoopedTransformerDecoder.")

    # 3. Setup W&B
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            job_type="rsi-finetune",
            config={
                "base_checkpoint": base_checkpoint,
                "total_steps": total_steps,
                "lr": lr,
                "batch_size": batch_size,
                "replay_ratio": replay_ratio,
                "masking": "prompt-masked (ignore_index=-100)",
            }
        )

    # 4. Optimizer & Cosine Scheduler
    depth_weights = [0.20, 0.40, 0.25, 0.15]
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=lr)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="cosine", warmup_steps=15, total_steps=total_steps)

    model.train()
    data_iter = iter(train_loader)
    tokens_processed = 0
    t0 = time.time()

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
            _, total_loss = model(x, targets=y, depth_weights=depth_weights)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step_elapsed = time.time() - step_t0
        step_tok_s = x.numel() / max(0.001, step_elapsed)
        bpb = (total_loss.item() / math.log(2)) * (1.0 / 3.08)
        lr_cur = scheduler.get_last_lr()[0]

        if step % 10 == 0 or step == total_steps:
            print(f"  [M_0 -> M_1 SFT] Step {step:3d}/{total_steps} | Loss: {total_loss.item():.4f} | BPB: {bpb:.3f} | LR: {lr_cur:.6f} | Speed: {step_tok_s:.0f} tok/s")

        if use_wandb:
            wandb.log({
                "finetune/step": step,
                "finetune/loss": total_loss.item(),
                "finetune/bpb": bpb,
                "finetune/lr": lr_cur,
                "finetune/tok_s": step_tok_s,
            }, step=step)

    elapsed = time.time() - t0
    print(f"\n--> Fine-Tuning Complete in {elapsed:.1f}s ({elapsed/60:.1f} mins) | Tokens Processed: {tokens_processed:,}")

    # Save M_1 Checkpoint
    Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    print(f"--> Saved M_1 Champion Checkpoint: {output_checkpoint}\n")

    if use_wandb:
        wandb.summary["final_loss"] = total_loss.item()
        wandb.summary["final_bpb"] = bpb
        wandb.summary["elapsed_s"] = elapsed
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/champion/mirendil_84m_multidepth.pt")
    parser.add_argument("--verified-path", type=str, default="data/rsi/verified_solutions_multidepth.jsonl")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/champion/mirendil_84m_rsi_m1.pt")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--name", type=str, default="rsi_policy_update_m0_to_m1_masked")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    train_rsi_m1(
        base_checkpoint=args.base_checkpoint,
        verified_path=args.verified_path,
        output_checkpoint=args.output_checkpoint,
        total_steps=args.steps,
        lr=args.lr,
        replay_ratio=args.ratio,
        use_wandb=args.wandb,
        wandb_name=args.name,
    )
