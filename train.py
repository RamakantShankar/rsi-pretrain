"""
Pretraining Runner for Foundation Models.
Loads YAML configuration, sets up packed memory-mapped datasets,
instantiates the modern Transformer, and starts the training engine.
"""

import os
import glob
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.model.transformer import TransformerDecoder, ModelConfig
from src.data.dataset import PackedMemmapDataset, build_synthetic_code_corpus, prepare_shards_from_texts
from src.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain LLM from Scratch")
    parser.add_argument("--config", type=str, default="config/micro_local.yaml", help="Path to config YAML")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry")
    parser.add_argument("--project", type=str, default="rsi-coding-foundation", help="W&B project name")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Loaded Configuration: {args.config}")

    # 1. Check or prepare data shards
    shard_pattern = cfg["data"]["train_shards"]
    shards = sorted(glob.glob(shard_pattern))

    if not shards:
        print(f"No shards found matching '{shard_pattern}'. Generating synthetic code corpus for run...")
        corpus = build_synthetic_code_corpus()
        output_dir = os.path.dirname(shard_pattern) or "data/test_shards"
        shards = prepare_shards_from_texts(corpus, output_dir=output_dir, tokens_per_shard=25_000)
        print(f"Generated {len(shards)} shards in {output_dir}")

    # 2. Build Dataset
    seq_len = cfg["model"]["max_seq_len"]
    full_dataset = PackedMemmapDataset(shards, seq_len=seq_len)
    
    val_split = cfg["data"].get("val_split", 0.1)
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset Size: {len(full_dataset):,} samples | Train: {train_size:,} | Val: {val_size:,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # 3. Instantiate Model
    m_cfg = ModelConfig(**cfg["model"])
    model = TransformerDecoder(m_cfg)
    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"Transformer Initialized: {total_params:,} total parameters ({non_emb_params:,} non-embedding)")

    # Determine W&B flag (CLI flag or YAML config)
    use_wandb = args.wandb or cfg.get("training", {}).get("wandb", False)
    wandb_project = cfg.get("training", {}).get("wandb_project", args.project)

    # 4. Start Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        scheduler_type=cfg["training"]["scheduler_type"],
        warmup_steps=int(cfg["training"]["warmup_steps"]),
        total_steps=int(cfg["training"]["total_steps"]),
        grad_accum_steps=int(cfg["training"]["grad_accum_steps"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        dtype=cfg["training"].get("dtype", "float32"),
        log_interval=int(cfg["training"]["log_interval"]),
        eval_interval=int(cfg["training"]["eval_interval"]),
        experiment_name=cfg.get("experiment_name", "run_01"),
        use_wandb=use_wandb,
        wandb_project=wandb_project,
    )

    trainer.train()


if __name__ == "__main__":
    main()
