"""
Autonomous Overnight Lifelong RSI Mission Driver.
Orchestrates:
  Phase 1: 50M Token Data Shard Packing (10 x 5M binary memory-mapped shards)
  Phase 2: Foundation Pretraining of M_0_v2 (1,000 steps with WSD on DGX Blackwell GB10)
  Phase 3: Lifelong Autonomous RSI Loop (M_0_v2 -> M_10_v2 with dynamic autoscaling)
Runs headlessly and resiliently in the background on the DGX server.
"""

import os
import sys
import glob
import time
import shutil
import argparse
from pathlib import Path
import yaml
import torch

from prepare_starcoder import shard_code_dataset
from train import main as pretrain_main, parse_args as pretrain_parse_args
from src.rsi.orchestrator import RSIOrchestrator


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_mission(
    target_tokens: int = 50_000_000,
    pretrain_config: str = "config/dgx_50m.yaml",
    max_rsi_generations: int = 10,
    use_wandb: bool = True,
):
    log("=" * 80)
    log("STARTING OVERNIGHT AUTONOMOUS MISSION: Mirendil-50M-Lifelong-RSI")
    log(f"Target Pretraining Tokens: {target_tokens:,}")
    log(f"Pretraining Config:        {pretrain_config}")
    log(f"Target Max RSI Generation: M_{max_rsi_generations}")
    log(f"Device:                    {'CUDA (Blackwell GB10)' if torch.cuda.is_available() else 'CPU'}")
    log("=" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1: 50M Token Binary Shard Packing
    # -------------------------------------------------------------------------
    log("\n>>> [PHASE 1] Checking 50M Token Pretraining Shards...")
    shards_dir = Path("data/shards_50m")
    shards_dir.mkdir(parents=True, exist_ok=True)
    expected_shards = target_tokens // 5_000_000

    existing_shards = sorted(glob.glob(str(shards_dir / "train_shard_*.bin")))
    if len(existing_shards) >= expected_shards:
        log(f"--> Found {len(existing_shards)} existing shards in {shards_dir}. Skipping sharding.")
    else:
        log(f"--> Packing {target_tokens:,} tokens into {shards_dir} (10 shards x 5M tokens)...")
        t0 = time.time()
        shard_code_dataset(
            dataset_name="codeparrot/codeparrot-clean-valid",
            output_dir=str(shards_dir),
            max_tokens_total=target_tokens,
            tokens_per_shard=5_000_000,
        )
        log(f"--> [PHASE 1 COMPLETE] Sharding finished in {time.time() - t0:.1f}s!")

    # -------------------------------------------------------------------------
    # PHASE 2: Foundation Pretraining M_0_v2 (50M Baseline)
    # -------------------------------------------------------------------------
    log("\n>>> [PHASE 2] Foundation Pretraining of M_0_v2 (1,000 WSD Steps)...")
    baseline_ckpt_dir = Path("checkpoints/152m_wsd_50m_baseline")
    baseline_ckpt_path = baseline_ckpt_dir / "best_model.pt"

    if baseline_ckpt_path.exists():
        log(f"--> Baseline checkpoint already exists at {baseline_ckpt_path}. Skipping pretraining.")
    else:
        log("--> Launching pretrainer on Blackwell GPU with FlashAttention-2 & BF16...")
        t0 = time.time()
        # Temporarily set sys.argv for train.py
        orig_argv = sys.argv
        sys.argv = ["train.py", "--config", pretrain_config]
        if use_wandb:
            sys.argv.append("--wandb")
        pretrain_main()
        sys.argv = orig_argv
        log(f"--> [PHASE 2 COMPLETE] Foundation model M_0_v2 pretrained in {time.time() - t0:.1f}s!")

    if not baseline_ckpt_path.exists():
        raise FileNotFoundError(f"Expected pretraining checkpoint not found at: {baseline_ckpt_path}")

    # -------------------------------------------------------------------------
    # PHASE 3: Deep Autonomous Lifelong RSI Traversal (M_0_v2 -> M_10_v2)
    # -------------------------------------------------------------------------
    log("\n>>> [PHASE 3] Launching Autonomous Lifelong RSI Loop across Tiers 1-4...")
    v2_ckpt_dir = Path("checkpoints/v2")
    v2_shard_dir = Path("data/shards_v2")
    v2_ckpt_dir.mkdir(parents=True, exist_ok=True)
    v2_shard_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Generation 0 policy from the 50M baseline
    gen0_dir = v2_ckpt_dir / "152m_rsi_m0"
    gen0_dir.mkdir(parents=True, exist_ok=True)
    gen0_ckpt = gen0_dir / "best_model.pt"
    if not gen0_ckpt.exists():
        log(f"--> Initializing M_0 policy in {gen0_ckpt} from {baseline_ckpt_path}...")
        shutil.copy2(str(baseline_ckpt_path), str(gen0_ckpt))

    first_shard = str(shards_dir / "train_shard_00000.bin")
    orchestrator = RSIOrchestrator(
        start_generation=0,
        max_generations=max_rsi_generations,
        checkpoint_dir=str(v2_ckpt_dir),
        shard_dir=str(v2_shard_dir),
        problem_bank_path="data/rsi/problem_bank.json",
        verified_path="data/rsi/verified_solutions_50m.jsonl",
        all_candidates_path="data/rsi/all_candidates_50m.jsonl",
        baseline_shard_path=first_shard,
        steps_per_gen=50,
        learning_rate=1.2e-4,
        candidates_k=5,
        temperature=0.8,
        use_wandb=use_wandb,
        wandb_project="rsi-coding-pretrain",
        autoscale=True,
        k_frontier=18,
        k_mastered=3,
        temp_frontier=0.88,
        temp_mastered=0.65,
    )

    t0 = time.time()
    orchestrator.run_loop()
    log(f"\n--> [PHASE 3 COMPLETE] Overnight RSI loop finished in {time.time() - t0:.1f}s!")

    log("\n" + "=" * 80)
    log("OVERNIGHT MISSION REACHED TARGET HORIZON SUCCESSFULLY!")
    log("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight Autonomous RSI Mission")
    parser.add_argument("--tokens", type=int, default=50_000_000, help="Pretraining tokens")
    parser.add_argument("--config", type=str, default="config/dgx_50m.yaml", help="Pretrain config")
    parser.add_argument("--max_gen", type=int, default=10, help="Max RSI generations")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B")
    args = parser.parse_args()

    run_mission(
        target_tokens=args.tokens,
        pretrain_config=args.config,
        max_rsi_generations=args.max_gen,
        use_wandb=not args.no_wandb,
    )
