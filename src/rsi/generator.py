"""
Autonomous Self-Play Candidate Generator for RSI.
Loads M_0 (or M_i) checkpoint, generates K candidate solutions per problem,
runs sandboxed verification, and curates accepted verified training data.
Dual-logs all candidate attempts locally (JSONL) and to Weights & Biases (interactive Table & metrics).
"""

import os
import json
import time
import argparse
import collections
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

from src.model.transformer import TransformerDecoder, ModelConfig
from src.tokenizer.bpe import CodeTokenizer
from src.rsi.verifier import CodeVerifier, VerificationResult
from generate import generate, generate_batch


class SelfPlayEngine:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/152m_wsd_10m_baseline/best_model.pt",
        device: str = None,
        timeout_sec: float = 2.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Self-Play Engine on: {self.device.upper()}")
        print(f"Loading Checkpoint from: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = TransformerDecoder(ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer = CodeTokenizer()
        self.verifier = CodeVerifier(timeout_sec=timeout_sec)
        print(f"Model Loaded: {self.model.get_num_params(non_embedding=False):,} parameters.")

    def run_self_play(
        self,
        problem_bank_path: str = "data/rsi/problem_bank.json",
        candidates_per_problem: int = 5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 128,
        max_problems: Optional[int] = None,
        output_verified_path: str = "data/rsi/verified_solutions.jsonl",
        all_candidates_path: str = "data/rsi/all_candidates.jsonl",
        use_wandb: bool = False,
        wandb_project: str = "rsi-coding-pretrain",
        wandb_run_name: Optional[str] = None,
        model_tag: str = "M_0_baseline",
        autoscale: bool = False,
        k_mastered: int = 3,
        k_frontier: int = 18,
        temp_mastered: float = 0.65,
        temp_frontier: float = 0.88,
        mastery_threshold: int = 5,
    ) -> Dict[str, Any]:
        with open(problem_bank_path, "r", encoding="utf-8") as f:
            problems = json.load(f)

        if max_problems:
            problems = problems[:max_problems]

        Path(output_verified_path).parent.mkdir(parents=True, exist_ok=True)
        Path(all_candidates_path).parent.mkdir(parents=True, exist_ok=True)

        # Inventory existing verified solutions for compute-optimal autoscaling
        mastered_counts = collections.Counter()
        if Path(output_verified_path).exists():
            try:
                with open(output_verified_path, "r", encoding="utf-8") as vf:
                    for line in vf:
                        if line.strip():
                            rec = json.loads(line)
                            t_id = rec.get("task_id")
                            if t_id:
                                mastered_counts[t_id] += 1
            except Exception as e:
                print(f"[WARNING] Could not parse existing verified solutions: {e}")

        # Initialize Weights & Biases if requested
        wandb_table = None
        wandb_run = None
        if use_wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"selfplay_{model_tag}",
                    job_type="rsi-eval-selfplay",
                    config={
                        "model_tag": model_tag,
                        "candidates_per_problem": candidates_per_problem,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "num_problems": len(problems),
                        "problem_bank": problem_bank_path,
                        "autoscale": autoscale,
                        "k_mastered": k_mastered if autoscale else None,
                        "k_frontier": k_frontier if autoscale else None,
                        "temp_mastered": temp_mastered if autoscale else None,
                        "temp_frontier": temp_frontier if autoscale else None,
                    },
                )
                wandb_table = wandb.Table(
                    columns=[
                        "task_id",
                        "name",
                        "candidate_k",
                        "passed",
                        "error_type",
                        "error_msg",
                        "exec_time_s",
                        "code_prompt",
                        "model_completion",
                    ]
                )
                print(f"--> [Weights & Biases] Connected! Dashboard: {wandb.run.url}")
            except Exception as e:
                print(f"[WARNING] Weights & Biases connection failed: {e}. Falling back to local logging only.")
                use_wandb = False

        print(f"\n{'='*70}")
        print(f"STARTING SELF-PLAY EXPLORATION ACROSS {len(problems)} PROBLEMS ({model_tag})")
        if autoscale:
            print(f"AUTOSCALING: ENABLED | Mastered: K={k_mastered}, T={temp_mastered} | Frontier: K={k_frontier}, T={temp_frontier}")
        else:
            print(f"AUTOSCALING: DISABLED | Static K={candidates_per_problem} | Temp: {temperature} | Top-p: {top_p}")
        print(f"W&B Dual-Logging: {'ENABLED' if use_wandb else 'DISABLED'}")
        print(f"{'='*70}\n")

        total_attempts = 0
        problems_with_at_least_one_pass = 0
        first_attempt_passes = 0
        all_verified_solutions = []

        # Error taxonomy tracking
        syntax_errors = 0
        assertion_errors = 0
        timeout_errors = 0
        other_errors = 0

        start_time = time.time()

        for idx, prob in enumerate(problems, 1):
            task_id = prob["task_id"]
            name = prob["name"]
            prompt = prob["prompt"]
            tests = prob["tests"]

            # Compute-optimal autoscaling allocation per task (Hysteresis Breadth Rule)
            if autoscale:
                gold_count = mastered_counts.get(task_id, 0)
                if gold_count >= mastery_threshold:
                    cur_k = k_mastered
                    cur_temp = temp_mastered
                    tier_tag = f"MASTERED ({gold_count} gold) -> K={cur_k}, T={cur_temp}"
                elif gold_count > 0:
                    # Developing tasks (< mastery_threshold) retain full exploratory breadth to discover algorithm variants!
                    cur_k = k_frontier
                    cur_temp = temp_frontier
                    tier_tag = f"DEVELOPING ({gold_count} gold) -> K={cur_k}, T={cur_temp}"
                else:
                    cur_k = k_frontier
                    cur_temp = temp_frontier
                    tier_tag = f"FRONTIER (0 gold) -> K={cur_k}, T={cur_temp}"
            else:
                cur_k = candidates_per_problem
                cur_temp = temperature
                tier_tag = f"STATIC -> K={cur_k}, T={cur_temp}"

            print(f"[{idx:2d}/{len(problems)}] Task: {task_id:<8s} ({name}) - {tier_tag}...")
            passed_any_for_this_problem = False

            # High-throughput parallel candidate generation on Blackwell GB10
            completions = generate_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                num_samples=cur_k,
                max_new_tokens=max_new_tokens,
                temperature=cur_temp,
                top_p=top_p,
                device=self.device,
            )

            for k, completion in enumerate(completions):
                total_attempts += 1

                # Verify completion with sandboxed unit tests
                res = self.verifier.verify(completion, tests)

                # Track failure modes
                if not res.passed:
                    if res.error_type in ("SYNTAX_ERROR", "SyntaxError"):
                        syntax_errors += 1
                    elif res.error_type in ("ASSERTION_ERROR", "AssertionError"):
                        assertion_errors += 1
                    elif res.error_type in ("TIMEOUT", "TimeoutExpired"):
                        timeout_errors += 1
                    else:
                        other_errors += 1

                # 1. ALWAYS record every attempt to all_candidates.jsonl
                attempt_record = {
                    "task_id": task_id,
                    "name": name,
                    "candidate_idx": k + 1,
                    "passed": res.passed,
                    "error_type": res.error_type,
                    "error_msg": res.error_msg,
                    "exec_time": round(res.execution_time_sec, 3),
                    "prompt": prompt,
                    "completion": completion,
                }
                with open(all_candidates_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(attempt_record) + "\n")

                # 2. Add to W&B interactive table
                if use_wandb and wandb_table is not None:
                    wandb_table.add_data(
                        task_id,
                        name,
                        k + 1,
                        bool(res.passed),
                        res.error_type or "None",
                        (res.error_msg or "")[:200],
                        round(res.execution_time_sec, 3),
                        prompt,
                        completion,
                    )

                if res.passed:
                    print(f"   Candidate #{k+1}: ✅ PASSED ({res.execution_time_sec:.3f}s)")
                    if not passed_any_for_this_problem:
                        passed_any_for_this_problem = True
                        problems_with_at_least_one_pass += 1
                    if k == 0:
                        first_attempt_passes += 1

                    # 3. Collect accepted verified code for next generation training!
                    record = {
                        "task_id": task_id,
                        "name": name,
                        "prompt": prompt,
                        "solution": completion,
                        "tests": tests,
                        "candidate_idx": k + 1,
                        "exec_time": res.execution_time_sec,
                    }
                    all_verified_solutions.append(record)
                    with open(output_verified_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                else:
                    print(f"   Candidate #{k+1}: ❌ {res.error_type} ({res.error_msg[:40]}...)")

        elapsed = time.time() - start_time
        pass_at_1 = (first_attempt_passes / len(problems)) * 100 if problems else 0.0
        pass_at_k = (problems_with_at_least_one_pass / len(problems)) * 100 if problems else 0.0

        print(f"\n{'='*70}")
        print(f"SELF-PLAY EXPLORATION SUMMARY ({model_tag})")
        print(f"{'='*70}")
        print(f"Total Problems Evaluated:        {len(problems)}")
        print(f"Total Code Candidates Generated: {total_attempts}")
        print(f"Total Solutions Verified Passed: {len(all_verified_solutions)}")
        print(f"Pass@1 Benchmark Accuracy:       {pass_at_1:.1f}% ({first_attempt_passes}/{len(problems)})")
        print(f"Pass@{candidates_per_problem} (Any Valid Solution):      {pass_at_k:.1f}% ({problems_with_at_least_one_pass}/{len(problems)})")
        print(f"Failure Breakdown:               Syntax={syntax_errors}, Assert={assertion_errors}, Timeout={timeout_errors}, Other={other_errors}")
        print(f"Time Elapsed:                    {elapsed:.1f}s")
        print(f"Gold Verified Shard Dataset:     {output_verified_path}")
        print(f"{'='*70}\n")

        # Log summary metrics and table to Weights & Biases
        if use_wandb and wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "eval/pass_at_1": pass_at_1,
                    f"eval/pass_at_{candidates_per_problem}": pass_at_k,
                    "eval/verified_solutions_count": len(all_verified_solutions),
                    "eval/total_candidates": total_attempts,
                    "eval/syntax_error_rate": (syntax_errors / total_attempts) * 100 if total_attempts else 0.0,
                    "eval/assertion_error_rate": (assertion_errors / total_attempts) * 100 if total_attempts else 0.0,
                    "eval/timeout_rate": (timeout_errors / total_attempts) * 100 if total_attempts else 0.0,
                    "eval/elapsed_sec": elapsed,
                    "eval/candidate_attempts_table": wandb_table,
                })
                wandb.finish()
                print("--> [Weights & Biases] Evaluation metrics & candidate table successfully pushed to W&B!")
            except Exception as e:
                print(f"[WARNING] Failed to push metrics to W&B: {e}")

        return {
            "num_problems": len(problems),
            "total_candidates": total_attempts,
            "verified_count": len(all_verified_solutions),
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k,
            "syntax_errors": syntax_errors,
            "assertion_errors": assertion_errors,
            "elapsed_sec": elapsed,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/152m_wsd_10m_baseline/best_model.pt")
    parser.add_argument("--problems", type=str, default="data/rsi/problem_bank.json")
    parser.add_argument("-k", "--candidates", type=int, default=3)
    parser.add_argument("--max_problems", type=int, default=None, help="Subset of problems for fast run (None for all)")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tag", type=str, default="M_0_baseline")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rsi-coding-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    engine = SelfPlayEngine(checkpoint_path=args.checkpoint)
    engine.run_self_play(
        problem_bank_path=args.problems,
        candidates_per_problem=args.candidates,
        max_problems=args.max_problems,
        temperature=args.temp,
        top_p=args.top_p,
        model_tag=args.tag,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
