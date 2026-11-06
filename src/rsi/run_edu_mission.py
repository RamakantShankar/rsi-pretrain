"""
Autonomous Educational Foundation & 20-Generation Lifelong RSI Mission Driver.
Orchestrates:
  Phase 1: 50M Token Educational Python Shard Packing (10 x 5M uint32 shards)
  Phase 2: Foundation Pretraining of M_0_edu (1,000 steps with WSD on DGX Blackwell GB10)
  Phase 3: Lifelong Autonomous RSI Loop (M_0_edu -> M_20_edu with dynamic autoscaling)
           100% UNCONTAMINATED: Zero human seed anchors!
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

from prepare_python_edu import shard_educational_python
from train import main as pretrain_main
from src.rsi.orchestrator import RSIOrchestrator


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_mission(
    target_tokens: int = 50_000_000,
    pretrain_config: str = "config/dgx_50m_edu.yaml",
    max_rsi_generations: int = 20,
    use_wandb: bool = True,
):
    log("=" * 80)
    log("STARTING DAYTIME AUTONOMOUS MISSION: Mirendil-Edu-50M-Lifelong-RSI")
    log(f"Target Pretraining Tokens: {target_tokens:,}")
    log(f"Pretraining Config:        {pretrain_config}")
    log(f"Target Max RSI Generation: M_{max_rsi_generations}")
    log(f"Device:                    {'CUDA (Blackwell GB10)' if torch.cuda.is_available() else 'CPU'}")
    log("=" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1: 50M Token Educational Python Shard Packing
    # -------------------------------------------------------------------------
    log("\n>>> [PHASE 1] Checking 50M Educational Python Shards...")
    shards_dir = Path("data/shards_edu")
    shards_dir.mkdir(parents=True, exist_ok=True)
    expected_shards = target_tokens // 5_000_000

    existing_shards = sorted(glob.glob(str(shards_dir / "train_shard_*.bin")))
    if len(existing_shards) >= expected_shards:
        log(f"--> Found {len(existing_shards)} existing shards in {shards_dir}. Skipping sharding.")
    else:
        log(f"--> Packing {target_tokens:,} educational Python tokens into {shards_dir} (10 shards x 5M tokens)...")
        t0 = time.time()
        shard_educational_python(
            output_dir=str(shards_dir),
            max_tokens_total=target_tokens,
            tokens_per_shard=5_000_000,
        )
        log(f"--> [PHASE 1 COMPLETE] Sharding finished in {time.time() - t0:.1f}s!")

    # -------------------------------------------------------------------------
    # PHASE 2: Foundation Pretraining M_0_edu (50M Baseline)
    # -------------------------------------------------------------------------
    log("\n>>> [PHASE 2] Foundation Pretraining of M_0_edu (1,000 WSD Steps)...")
    baseline_ckpt_dir = Path("checkpoints/152m_wsd_50m_edu_baseline")
    baseline_ckpt_path = baseline_ckpt_dir / "best_model.pt"

    if baseline_ckpt_path.exists():
        log(f"--> Baseline checkpoint already exists at {baseline_ckpt_path}. Skipping pretraining.")
    else:
        log("--> Launching pretrainer on Blackwell GPU with FlashAttention-2 & BF16...")
        t0 = time.time()
        orig_argv = sys.argv
        sys.argv = ["train.py", "--config", pretrain_config]
        if use_wandb:
            sys.argv.append("--wandb")
        pretrain_main()
        sys.argv = orig_argv
        log(f"--> [PHASE 2 COMPLETE] Foundation model M_0_edu pretrained in {time.time() - t0:.1f}s!")

    if not baseline_ckpt_path.exists():
        raise FileNotFoundError(f"Expected pretraining checkpoint not found at: {baseline_ckpt_path}")

    # -------------------------------------------------------------------------
    # PHASE 3: Deep Autonomous Lifelong RSI Traversal (M_0_edu -> M_20_edu)
    # -------------------------------------------------------------------------
    log("\n>>> [PHASE 3] Launching Autonomous Lifelong RSI Loop across 55 Held-Out Problems...")
    v3_ckpt_dir = Path("checkpoints/v3")
    v3_shard_dir = Path("data/shards_v3")
    v3_ckpt_dir.mkdir(parents=True, exist_ok=True)
    v3_shard_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Generation 0 policy from the 50M edu baseline
    gen0_dir = v3_ckpt_dir / "152m_rsi_m0"
    gen0_dir.mkdir(parents=True, exist_ok=True)
    gen0_ckpt = gen0_dir / "best_model.pt"

    if not gen0_ckpt.exists():
        log(f"--> Copying baseline weights to initial Generation 0 checkpoint: {gen0_ckpt}")
        shutil.copyfile(str(baseline_ckpt_path), str(gen0_ckpt))
        metrics_src = baseline_ckpt_dir / "metrics.jsonl"
        if metrics_src.exists():
            shutil.copyfile(str(metrics_src), str(gen0_dir / "metrics.jsonl"))

    # Initial verified solutions pool: FRESH & ZERO-CONTAMINATION
    verified_pool_path = Path("data/rsi/verified_solutions_edu.jsonl")
    all_candidates_path = Path("data/rsi/all_candidates_edu.jsonl")
    baseline_shard = shards_dir / "train_shard_00000.bin"

    log("--> Verifying configuration for Phase 3 Lifelong RSI:")
    log(f"    Checkpoints Directory: {v3_ckpt_dir}")
    log(f"    Replay Shard Dir:      {v3_shard_dir}")
    log(f"    Verified Pool Path:    {verified_pool_path} (Zero seeds - 100% held-out!)")
    log(f"    Baseline Mixing Shard: {baseline_shard}")
    log(f"    Target Generations:    M_0 -> M_{max_rsi_generations}")
    log(f"    Autoscaling:           ENABLED (Frontier K=18, Mastered K=3)")
    log(f"    Distillation:          50 steps per cycle @ LR 1.2e-4 (WSD cosine decay)")

    orchestrator = RSIOrchestrator(
        start_generation=0,
        max_generations=max_rsi_generations,
        checkpoint_dir=str(v3_ckpt_dir),
        shard_dir=str(v3_shard_dir),
        verified_path=str(verified_pool_path),
        all_candidates_path=str(all_candidates_path),
        baseline_shard_path=str(baseline_shard),
        k_frontier=18,
        k_mastered=3,
        temp_frontier=0.88,
        temp_mastered=0.65,
        autoscale=True,
        steps_per_gen=50,
        learning_rate=1.2e-4,
        use_wandb=use_wandb,
    )

    t_start_rsi = time.time()
    orchestrator.run_loop()
    total_rsi_time = time.time() - t_start_rsi

    log("=" * 80)
    log(f"AUTONOMOUS DAYTIME MISSION COMPLETE in {total_rsi_time / 3600:.2f} hours!")
    log(f"Model successfully traversed from M_0_edu -> M_{max_rsi_generations}_edu")
    log("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autonomous educational foundation & RSI mission")
    parser.add_argument("--tokens", type=int, default=50_000_000, help="Target pretraining tokens")
    parser.add_argument("--config", type=str, default="config/dgx_50m_edu.yaml", help="Pretrain config")
    parser.add_argument("--max-gen", type=int, default=20, help="Max RSI generation")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = parser.parse_args()

    run_mission(
        target_tokens=args.tokens,
        pretrain_config=args.config,
        max_rsi_generations=args.max_gen,
        use_wandb=args.wandb,
    )
