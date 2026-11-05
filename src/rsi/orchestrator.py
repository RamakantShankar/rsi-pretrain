"""
Autonomous Recursive Self-Improvement (RSI) Loop Orchestrator.
Continuously runs the self-play -> verification -> replay synthesis -> distillation loop
with automated regression guards, dynamic K exploration, and cloud telemetry.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader, random_split

from src.model.transformer import TransformerDecoder, ModelConfig
from src.data.dataset import PackedMemmapDataset
from src.engine.trainer import Trainer
from src.rsi.generator import SelfPlayEngine
from src.rsi.pack_replay_shard import pack_experience_replay_shard


class RSIOrchestrator:
    def __init__(
        self,
        start_generation: int = 2,
        max_generations: int = 5,
        base_dir: str = ".",
        checkpoint_dir: str = "checkpoints",
        shard_dir: str = "data/shards",
        problem_bank_path: str = "data/rsi/problem_bank.json",
        verified_path: str = "data/rsi/verified_solutions.jsonl",
        all_candidates_path: str = "data/rsi/all_candidates.jsonl",
        baseline_shard_path: str = "data/shards/train_shard_00000.bin",
        steps_per_gen: int = 50,
        learning_rate: float = 1.2e-4,
        candidates_k: int = 5,
        temperature: float = 0.8,
        use_wandb: bool = True,
        wandb_project: str = "rsi-coding-pretrain",
        autoscale: bool = True,
        k_frontier: int = 18,
        k_mastered: int = 3,
        temp_frontier: float = 0.88,
        temp_mastered: float = 0.65,
    ):
        self.current_gen = start_generation
        self.max_generations = max_generations
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.shard_dir = Path(shard_dir)
        self.problem_bank_path = problem_bank_path
        self.verified_path = verified_path
        self.all_candidates_path = all_candidates_path
        self.baseline_shard_path = baseline_shard_path
        self.steps_per_gen = steps_per_gen
        self.learning_rate = learning_rate
        self.candidates_k = candidates_k
        self.temperature = temperature
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.autoscale = autoscale
        self.k_frontier = k_frontier
        self.k_mastered = k_mastered
        self.temp_frontier = temp_frontier
        self.temp_mastered = temp_mastered

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_history: List[Dict[str, Any]] = []

    def get_checkpoint_path(self, gen: int) -> str:
        return str(self.checkpoint_dir / f"152m_rsi_m{gen}" / "best_model.pt")

    def get_shard_path(self, gen: int) -> str:
        return str(self.shard_dir / f"train_shard_rsi_m{gen}.bin")

    def run_generation_cycle(self, gen: int) -> Dict[str, Any]:
        """
        Executes a single end-to-end self-improvement cycle: M_{gen} -> M_{gen+1}.
        """
        next_gen = gen + 1
        current_ckpt = self.get_checkpoint_path(gen)
        next_ckpt_dir = self.checkpoint_dir / f"152m_rsi_m{next_gen}"
        next_ckpt_path = self.get_checkpoint_path(next_gen)
        next_shard_path = self.get_shard_path(next_gen)

        print(f"\n{'#'*80}")
        print(f"### AUTONOMOUS RSI CYCLE: GENERATION M_{gen} -> M_{next_gen} ###")
        print(f"{'#'*80}")
        print(f"Base Policy:          {current_ckpt}")
        print(f"Target Policy:        {next_ckpt_path}")
        print(f"Candidates per Task:  K={self.candidates_k} | Temp: {self.temperature}")
        print(f"Distillation Steps:   {self.steps_per_gen} steps @ LR {self.learning_rate:.2e}")
        print(f"Device:               {self.device.upper()}")
        print(f"{'#'*80}\n")

        # -------------------------------------------------------------
        # PHASE 1: Self-Play Rollout & Sandboxed Verification
        # -------------------------------------------------------------
        print(f"\n>>> [PHASE 1] Launching Self-Play Exploration on M_{gen}...")
        engine = SelfPlayEngine(checkpoint_path=current_ckpt, device=self.device)
        eval_run_name = f"eval_m{gen}_autonomous_loop"

        eval_summary = engine.run_self_play(
            problem_bank_path=self.problem_bank_path,
            candidates_per_problem=self.candidates_k,
            temperature=self.temperature,
            output_verified_path=self.verified_path,
            all_candidates_path=self.all_candidates_path,
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
            wandb_run_name=eval_run_name,
            model_tag=f"M_{gen}_loop",
            autoscale=self.autoscale,
            k_frontier=self.k_frontier,
            k_mastered=self.k_mastered,
            temp_frontier=self.temp_frontier,
            temp_mastered=self.temp_mastered,
        )

        pass_at_1 = eval_summary["pass_at_1"]
        pass_at_k = eval_summary["pass_at_k"]
        verified_count = eval_summary["verified_count"]
        print(f"--> [PHASE 1 COMPLETE] M_{gen} Pass@1: {pass_at_1:.1f}% | Pass@{self.candidates_k}: {pass_at_k:.1f}% | Total Gold Solutions: {verified_count}")

        # -------------------------------------------------------------
        # PHASE 2: Experience Replay Shard Synthesis
        # -------------------------------------------------------------
        print(f"\n>>> [PHASE 2] Packing 90/10 Experience Replay Shard for M_{next_gen}...")
        pack_experience_replay_shard(
            verified_solutions_path=self.verified_path,
            baseline_shard_path=self.baseline_shard_path,
            output_shard_path=next_shard_path,
            problem_bank_path=self.problem_bank_path,
            target_tokens=4_000_000,
            replay_ratio=0.10,
            bootstrap_if_empty=True,  # Cold-Start bootstrap protection if 0 solutions passed
        )

        # -------------------------------------------------------------
        # PHASE 3: Policy Distillation (Fine-Tuning on GB10)
        # -------------------------------------------------------------
        print(f"\n>>> [PHASE 3] Distilling Search Results into Policy M_{next_gen}...")
        ckpt = torch.load(current_ckpt, map_location=self.device, weights_only=False)
        config = ckpt["config"]
        if isinstance(config, dict):
            config = ModelConfig(**config)

        model = TransformerDecoder(config).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Load synthesized replay shard
        full_dataset = PackedMemmapDataset([next_shard_path], seq_len=config.max_seq_len)
        val_size = max(1, int(len(full_dataset) * 0.05))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        train_run_name = f"distill_m{next_gen}_autonomous_loop"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=self.learning_rate,
            weight_decay=0.1,
            scheduler_type="wsd",
            warmup_steps=10,
            total_steps=self.steps_per_gen,
            grad_accum_steps=4,
            max_grad_norm=1.0,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            log_interval=5,
            eval_interval=25,
            save_dir=str(self.checkpoint_dir),
            experiment_name=f"152m_rsi_m{next_gen}",
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
        )

        trainer.train()
        print(f"--> [PHASE 3 COMPLETE] Saved Model Checkpoint: {next_ckpt_path}")

        # -------------------------------------------------------------
        # PHASE 4: Validation & Regression Guard
        # -------------------------------------------------------------
        print(f"\n>>> [PHASE 4] Running Regression Guard Verification on M_{next_gen}...")
        next_engine = SelfPlayEngine(checkpoint_path=next_ckpt_path, device=self.device)
        verify_run_name = f"verify_m{next_gen}_guard"

        guard_summary = next_engine.run_self_play(
            problem_bank_path=self.problem_bank_path,
            candidates_per_problem=self.candidates_k,
            temperature=self.temperature,
            output_verified_path=self.verified_path,
            all_candidates_path=self.all_candidates_path,
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
            wandb_run_name=verify_run_name,
            model_tag=f"M_{next_gen}_guard",
        )

        next_pass_at_1 = guard_summary["pass_at_1"]
        next_pass_at_k = guard_summary["pass_at_k"]

        delta_p1 = next_pass_at_1 - pass_at_1
        print(f"\n{'='*70}")
        print(f"REGRESSION GUARD VERDICT: M_{gen} ({pass_at_1:.1f}%) -> M_{next_gen} ({next_pass_at_1:.1f}%)")
        print(f"Delta Pass@1: {delta_p1:+.1f}% | Next Pass@{self.candidates_k}: {next_pass_at_k:.1f}%")
        print(f"{'='*70}")

        if delta_p1 >= -2.0:  # Allow 2% stochastic tolerance
            status = "PROMOTED"
            print(f"✅ Checkpoint M_{next_gen} PROMOTED to active baseline policy!")
        else:
            status = "REGRESSION_WARNING"
            print(f"⚠️ Regression detected in M_{next_gen} (Delta: {delta_p1:.1f}%). Retaining M_{gen} as safe baseline.")

        cycle_record = {
            "generation": next_gen,
            "base_checkpoint": current_ckpt,
            "promoted_checkpoint": next_ckpt_path,
            "pass_at_1": next_pass_at_1,
            "pass_at_k": next_pass_at_k,
            "delta_p1": delta_p1,
            "status": status,
            "verified_solutions_count": guard_summary["verified_count"],
        }
        self.generation_history.append(cycle_record)

        # Log cycle milestone to W&B
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    name=f"cycle_m{gen}_to_m{next_gen}_summary",
                    job_type="rsi-cycle-ledger",
                )
                wandb.log({
                    "cycle/generation": next_gen,
                    "cycle/pass_at_1": next_pass_at_1,
                    f"cycle/pass_at_{self.candidates_k}": next_pass_at_k,
                    "cycle/delta_pass_at_1": delta_p1,
                    "cycle/verified_total": guard_summary["verified_count"],
                })
                wandb.finish()
            except Exception:
                pass

        return cycle_record

    def run_loop(self):
        """Runs continuous recursive self-improvement up to max_generations."""
        print(f"\n{'*'*80}")
        print(f"STARTING AUTONOMOUS LIFELONG RSI LOOP (M_{self.current_gen} -> M_{self.max_generations})")
        print(f"{'*'*80}\n")

        while self.current_gen < self.max_generations:
            record = self.run_generation_cycle(self.current_gen)
            if record["status"] == "PROMOTED":
                self.current_gen += 1
            else:
                print(f"[RETRY] Adjusting hyper-parameters for M_{self.current_gen} retry...")
                self.learning_rate *= 0.7  # Lower learning rate to stabilize policy
                self.temperature = max(0.5, self.temperature - 0.1)

        print(f"\n{'='*80}")
        print("AUTONOMOUS RSI LOOP REACHED TARGET GENERATION!")
        print(f"{'='*80}")
        for r in self.generation_history:
            print(f"Gen M_{r['generation']}: Pass@1 = {r['pass_at_1']:.1f}% ({r['delta_p1']:+.1f}%) | Status: {r['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Recursive Self-Improvement Loop")
    parser.add_argument("--start_gen", type=int, default=2, help="Starting model generation")
    parser.add_argument("--max_gen", type=int, default=3, help="Target max generation")
    parser.add_argument("-k", "--candidates", type=int, default=5, help="Candidates per problem")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--steps", type=int, default=50, help="Distillation steps per generation")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Policy distillation learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B dual-logging")
    parser.add_argument("--project", type=str, default="rsi-coding-pretrain")
    parser.add_argument("--autoscale", dest="autoscale", action="store_true", default=True, help="Autoscale search breadth and temperature")
    parser.add_argument("--no-autoscale", dest="autoscale", action="store_false", help="Disable autoscaling")
    parser.add_argument("--k_frontier", type=int, default=18, help="Candidates for unsolved frontier tasks")
    parser.add_argument("--k_mastered", type=int, default=3, help="Candidates for mastered tasks")
    parser.add_argument("--temp_frontier", type=float, default=0.88, help="Temperature for frontier tasks")
    parser.add_argument("--temp_mastered", type=float, default=0.65, help="Temperature for mastered tasks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--shard_dir", type=str, default="data/shards", help="Directory for replay shards")
    parser.add_argument("--verified_path", type=str, default="data/rsi/verified_solutions.jsonl", help="Path to verified solutions JSONL")
    parser.add_argument("--all_candidates_path", type=str, default="data/rsi/all_candidates.jsonl", help="Path to all candidates JSONL")
    parser.add_argument("--baseline_shard", type=str, default="data/shards/train_shard_00000.bin", help="Path to baseline pretrain shard")
    args = parser.parse_args()

    orchestrator = RSIOrchestrator(
        start_generation=args.start_gen,
        max_generations=args.max_gen,
        checkpoint_dir=args.checkpoint_dir,
        shard_dir=args.shard_dir,
        verified_path=args.verified_path,
        all_candidates_path=args.all_candidates_path,
        baseline_shard_path=args.baseline_shard,
        candidates_k=args.candidates,
        temperature=args.temp,
        steps_per_gen=args.steps,
        learning_rate=args.lr,
        use_wandb=args.wandb,
        wandb_project=args.project,
        autoscale=args.autoscale,
        k_frontier=args.k_frontier,
        k_mastered=args.k_mastered,
        temp_frontier=args.temp_frontier,
        temp_mastered=args.temp_mastered,
    )
    orchestrator.run_loop()
