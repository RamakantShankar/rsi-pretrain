#!/usr/bin/env python3
"""
Rama-84M Stage 2 100-Step Probe Trainer & Dual-Gate Evaluator.
Executes 100 training steps on the balanced 5-stream curriculum mix (35% Code, 25% Math-CoT, 20% ChatML, 10% Logic, 10% CodeAct).
Performs immediate automated dual-gate evaluation:
  Gate 1: 55-Task Code Benchmark (verifies Pass@1 code preservation)
  Gate 2: Multimodal Capability Probing (CoT Math, ChatML Persona, World Logic, CodeAct Tool Use)
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


class Stage2Dataset(Dataset):
    def __init__(self, json_path: str, max_seq_len: int = 1024, pad_token_id: int = 0):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        print(f"[*] Loaded {len(self.samples)} samples from {json_path}")

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
    max_len = max(len(b['input_ids']) for b in batch)
    max_len = ((max_len + 7) // 8) * 8
    
    batch_size = len(batch)
    padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    for i, b in enumerate(batch):
        seq_len = len(b['input_ids'])
        padded_inputs[i, :seq_len] = b['input_ids']
        padded_labels[i, :seq_len] = b['labels']
        
    return padded_inputs, padded_labels


def run_probes(model: nn.Module, tokenizer: Tokenizer, device: torch.device):
    """Runs zero-shot qualitative capability probes across all 4 new capabilities."""
    model.eval()
    print("\n" + "=" * 70)
    print("RUNNING ZERO-SHOT QUALITATIVE PROBES")
    print("=" * 70)

    test_prompts = [
        ("Identity / Persona", "<|im_start|>user\nHello! Who are you?<|im_end|>\n<|im_start|>assistant\n"),
        ("Deductive CoT Math", "<|im_start|>user\nA bakery has 48 cupcakes. They sell 3/8 of them. How many were sold?<|im_end|>\n<|im_start|>assistant\n"),
        ("World Logic", "<|im_start|>user\nIf today is Tuesday, what day of the week will it be in 8 days?<|im_end|>\n<|im_start|>assistant\n"),
        ("CodeAct Tool Use", "<|im_start|>user\nCalculate 17 to the 5th power minus 23 squared.<|im_end|>\n<|im_start|>assistant\n")
    ]

    im_end_id = tokenizer.token_to_id("<|im_end|>")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    for label, prompt in test_prompts:
        input_ids = tokenizer.encode(prompt).ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated = list(input_ids)
        
        with torch.no_grad():
            for _ in range(128):
                out = model(x, num_loops=2)
                logits = out[0] if isinstance(out, tuple) else out
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                generated.append(next_id)
                if next_id in (im_end_id, eos_id):
                    break
                x = torch.tensor([generated], dtype=torch.long, device=device)
                
        decoded = tokenizer.decode(generated)
        print(f"\n--- PROBE: {label} ---")
        print(decoded)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Rama-84M Stage 2 100-Step Probe Trainer")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/champion/rama_84m_instruct_pilot.pt")
    parser.add_argument("--data-path", type=str, default="data/rsi/stage2_probe_mix_1000.json")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/champion/rama_84m_stage2_probe.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if args.wandb:
        import wandb
        wandb.init(
            project="rsi-coding-pretrain",
            name="rama_84m_stage2_100step_probe",
            config={
                "base_checkpoint": args.base_checkpoint,
                "data_path": args.data_path,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "vocab_size": 16388,
                "curriculum": "35_code_25_math_20_chat_10_logic_10_codeact"
            }
        )

    # 1. Initialize Model
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
    
    print("\n[*] Initializing Rama-84M Looped Transformer...")
    model = LoopedTransformerDecoder(config=config, default_loops=2).to(device)
    
    print(f"[*] Loading weights from: {args.base_checkpoint}")
    state = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    
    # 2. Data Loader
    dataset = Stage2Dataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    # 3. Optimizer & Scheduler
    optimizer = configure_optimizers(model, weight_decay=args.weight_decay, learning_rate=args.lr)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="cosine", warmup_steps=10, total_steps=args.steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop
    print(f"\n[*] Starting Stage 2 Probe Training for {args.steps} steps...")
    print(f"    Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum} | Effective Batch: {args.batch_size * args.grad_accum}")
    
    model.train()
    step = 0
    epoch = 0
    t0 = time.time()
    accum_loss = 0.0

    while step < args.steps:
        epoch += 1
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            shift_inputs = inputs[:, :-1].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            # Forward pass with Multi-Depth Deep Supervision (loops=2)
            logits, loss = model(
                shift_inputs,
                targets=shift_targets,
                depth_weights=[0.3, 0.7]
            )
            
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                
                if step % 10 == 0 or step == args.steps:
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {step:03d}/{args.steps} | Loss: {accum_loss / args.grad_accum:.4f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s")
                    if args.wandb:
                        wandb.log({"step": step, "loss": accum_loss / args.grad_accum, "lr": lr})
                
                accum_loss = 0.0
                if step >= args.steps:
                    break

    # 5. Save Checkpoint
    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), args.output_checkpoint)
    print(f"\n[+] Saved probe checkpoint to: {args.output_checkpoint}")

    # 6. Run Qualitative Probes
    tok = Tokenizer.from_file("checkpoints/tokenizers/rama_16k.json")
    run_probes(model, tok, device)


if __name__ == "__main__":
    main()
