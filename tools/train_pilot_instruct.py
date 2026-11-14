#!/usr/bin/env python3
"""
Rama-84M Instruct Pilot Training Script
Trains the expanded 16,388-vocab Looped Transformer on the 90/10 Code+ChatML pilot mix.
Uses Prompt-Masked Cross Entropy Loss (targets = -100 on prompt tokens).
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
root_dir = Path(__file__).resolve().parent.parent
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


class PilotMixDataset(Dataset):
    """
    Dataset loading the pre-tokenized 90/10 mix with explicit prompt-masked labels.
    Pads sequences dynamically to max length in batch or fixed block size.
    """
    def __init__(self, json_path: str, max_seq_len: int = 1024, pad_token_id: int = 0):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        print(f"Loaded {len(self.samples)} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        ids = s['input_ids'][:self.max_seq_len]
        lbl = s['labels'][:self.max_seq_len]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.tensor(lbl, dtype=torch.long),
            'type': s.get('type', 'code')
        }


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collates variable-length sequences with right-padding."""
    max_len = max(len(b['input_ids']) for b in batch)
    # Ensure divisible by 8 for Tensor Core efficiency
    max_len = ((max_len + 7) // 8) * 8
    
    batch_size = len(batch)
    padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    for i, b in enumerate(batch):
        seq_len = len(b['input_ids'])
        padded_inputs[i, :seq_len] = b['input_ids']
        padded_labels[i, :seq_len] = b['labels']
        
    return padded_inputs, padded_labels


def main():
    parser = argparse.ArgumentParser(description="Rama-84M Instruct Pilot Trainer")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/champion/rama_84m_base.pt")
    parser.add_argument("--data-path", type=str, default="data/rsi/pilot_mix_90_10_rama.json")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/champion/rama_84m_instruct_pilot.pt")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", type=str, default="rama_84m_instruct_pilot_90_10")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Initialize Weights & Biases
    if args.wandb:
        import wandb
        wandb.init(
            project="rsi-coding-pretrain",
            name=args.name,
            config={
                "base_checkpoint": args.base_checkpoint,
                "data_path": args.data_path,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "vocab_size": 16388,
                "architecture": "LoopedTransformer-84M-Recurrent",
                "experiment": "pilot_instruct_90_10"
            }
        )

    # 2. Build Model Architecture
    config = ModelConfig(
        vocab_size=16388,
        dim=896,
        n_layers=8,
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=1024,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    
    print("\nInitializing Rama-84M Architecture (Vocab: 16,388, Dim: 896)...")
    model = LoopedTransformerDecoder(config=config, default_loops=2).to(device)
    
    # 3. Load Base Checkpoint
    print(f"Loading weights from: {args.base_checkpoint}")
    state = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # 4. Prepare Dataset & Dataloader
    dataset = PilotMixDataset(args.data_path, max_seq_len=1024, pad_token_id=0)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=0),
        drop_last=True
    )

    # 5. Configure Optimizer & Scheduler
    optimizer = configure_optimizers(
        model,
        weight_decay=args.weight_decay,
        learning_rate=args.lr
    )
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type="cosine",
        warmup_steps=15,
        total_steps=args.steps
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # 6. Training Loop
    print("\n" + "=" * 70)
    print("STARTING RAMA-84M INSTRUCT PILOT TRAINING")
    print("=" * 70)
    
    data_iter = iter(dataloader)
    total_tokens_processed = 0
    start_time = time.time()
    
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_tokens = 0
    
    step = 0
    while step < args.steps:
        for micro_step in range(args.grad_accum):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                shift_inputs = inputs[:, :-1].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                
                # Multi-Depth Deep Supervision across loops (T=2)
                logits, loss = model(
                    shift_inputs,
                    targets=shift_targets,
                    depth_weights=[0.3, 0.7]
                )
                loss_scaled = loss / args.grad_accum

            loss_scaled.backward()
            accum_loss += loss.item() / args.grad_accum
            
            # Count supervised tokens (targets != -100)
            supervised_tokens = (shift_targets != -100).sum().item()
            accum_tokens += supervised_tokens

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        step += 1
        total_tokens_processed += accum_tokens
        current_lr = scheduler.get_last_lr()[0]
        bpb = accum_loss / math.log(2) * 0.25  # Approx BPB conversion
        
        if step % 10 == 0 or step == args.steps:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens_processed / max(1e-5, elapsed)
            print(f"  [Step {step:3d}/{args.steps}] Loss: {accum_loss:.4f} | BPB: {bpb:.3f} | LR: {current_lr:.2e} | Speed: {tok_per_sec:.0f} tok/s")
            
            if args.wandb:
                wandb.log({
                    "step": step,
                    "train/loss": accum_loss,
                    "train/bpb": bpb,
                    "train/learning_rate": current_lr,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/total_tokens": total_tokens_processed
                })

        accum_loss = 0.0
        accum_tokens = 0

    elapsed_total = time.time() - start_time
    print(f"\nTraining Complete in {elapsed_total:.1f}s ({elapsed_total/60:.1f} mins)!")
    print(f"Supervised Tokens Processed: {total_tokens_processed:,}")

    # 7. Save Checkpoint
    output_dir = Path(args.output_checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clean state dict
    torch.save(model.state_dict(), args.output_checkpoint)
    print(f"Successfully saved checkpoint to: {args.output_checkpoint}")
    
    if args.wandb:
        wandb.finish()

    print("=" * 70)


if __name__ == "__main__":
    main()
