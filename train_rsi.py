"""
RSI Policy Distillation & Fine-Tuning Engine (M_0 -> M_1).
Loads an existing checkpoint, trains on the 90/10 Experience Replay Shard
with a fine-tuning learning rate and WSD/Cosine decay, and saves the M_1 checkpoint.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from src.model.transformer import TransformerDecoder, ModelConfig
from src.data.dataset import PackedMemmapDataset
from src.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="RSI Fine-Tuning Engine (M_0 -> M_1)")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/152m_wsd_10m_baseline/best_model.pt")
    parser.add_argument("--shard", type=str, default="data/shards/train_shard_rsi_m1.bin")
    parser.add_argument("--steps", type=int, default=50, help="Fine-tuning steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Fine-tuning learning rate")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="152m_rsi_m1")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="rsi-coding-pretrain")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"STARTING RSI DISTILLATION RUN: M_0 -> M_1 ({args.experiment_name})")
    print(f"Base Checkpoint: {args.base_checkpoint}")
    print(f"Replay Shard:    {args.shard}")
    print(f"Total Steps:     {args.steps} | LR: {args.lr} | Batch Size: {args.batch_size}x{args.grad_accum}")
    print(f"Target Device:   {device.upper()}")
    print(f"{'='*70}\n")

    # 1. Load M_0 Checkpoint
    print(f"--> Loading base checkpoint from: {args.base_checkpoint}")
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = ModelConfig(**config)

    model = TransformerDecoder(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"--> Base Model Loaded: {model.get_num_params(non_embedding=False):,} parameters.")

    # 2. Build Dataset from Replay Shard
    seq_len = config.max_seq_len
    full_dataset = PackedMemmapDataset([args.shard], seq_len=seq_len)
    
    val_size = max(1, int(len(full_dataset) * 0.05))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"--> Replay Dataset: {len(full_dataset):,} samples (Train: {train_size}, Val: {val_size})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 3. Instantiate Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=0.1,
        scheduler_type="wsd",
        warmup_steps=args.warmup,
        total_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        max_grad_norm=1.0,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        log_interval=5,
        eval_interval=25,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    print("\n--> Launching Fine-Tuning Optimization Loop...")
    trainer.train()
    print(f"\n[SUCCESS] Model M_1 saved to: {os.path.join(args.save_dir, args.experiment_name, 'best_model.pt')}")


if __name__ == "__main__":
    main()
