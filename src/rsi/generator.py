"""
Autonomous Self-Play Candidate Generator for RSI.
Loads M_0 checkpoint, generates K candidate solutions per problem,
runs sandboxed verification, and curates accepted verified training data.
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch

from src.model.transformer import TransformerDecoder, ModelConfig
from src.tokenizer.bpe import CodeTokenizer
from src.rsi.verifier import CodeVerifier, VerificationResult
from generate import generate


class SelfPlayEngine:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/152m_wsd_10m_baseline/best_model.pt",
        device: str = None,
        timeout_sec: float = 2.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Self-Play Engine on: {self.device.upper()}")
        print(f"Loading M_0 Checkpoint from: {checkpoint_path}")

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
        max_problems: int = None,
        output_verified_path: str = "data/rsi/verified_solutions.jsonl",
    ) -> Dict[str, Any]:
        with open(problem_bank_path, "r", encoding="utf-8") as f:
            problems = json.load(f)

        if max_problems:
            problems = problems[:max_problems]

        print(f"\n{'='*70}")
        print(f"STARTING SELF-PLAY EXPLORATION ACROSS {len(problems)} PROBLEMS")
        print(f"Candidates per problem (K): {candidates_per_problem} | Temp: {temperature} | Top-p: {top_p}")
        print(f"{'='*70}\n")

        total_attempts = 0
        problems_with_at_least_one_pass = 0
        first_attempt_passes = 0
        all_verified_solutions = []

        start_time = time.time()

        for idx, prob in enumerate(problems, 1):
            task_id = prob["task_id"]
            name = prob["name"]
            prompt = prob["prompt"]
            tests = prob["tests"]

            print(f"[{idx:2d}/{len(problems)}] Task: {task_id:<8s} ({name})")
            passed_any_for_this_problem = False

            for k in range(candidates_per_problem):
                total_attempts += 1
                completion = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    device=self.device,
                )

                # Verify completion with sandboxed unit tests
                res = self.verifier.verify(completion, tests)

                if res.passed:
                    print(f"   Candidate #{k+1}: ✅ PASSED ({res.execution_time_sec:.3f}s)")
                    if not passed_any_for_this_problem:
                        passed_any_for_this_problem = True
                        problems_with_at_least_one_pass += 1
                    if k == 0:
                        first_attempt_passes += 1

                    # Collect accepted verified code for M_1 training!
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
        pass_at_1 = (first_attempt_passes / len(problems)) * 100
        pass_at_k = (problems_with_at_least_one_pass / len(problems)) * 100

        print(f"\n{'='*70}")
        print("SELF-PLAY EXPLORATION SUMMARY (M_0 BASELINE)")
        print(f"{'='*70}")
        print(f"Total Problems Evaluated:        {len(problems)}")
        print(f"Total Code Candidates Generated: {total_attempts}")
        print(f"Total Solutions Verified Passed: {len(all_verified_solutions)}")
        print(f"Pass@1 Benchmark Accuracy:       {pass_at_1:.1f}% ({first_attempt_passes}/{len(problems)})")
        print(f"Pass@{candidates_per_problem} (Any Valid Solution):      {pass_at_k:.1f}% ({problems_with_at_least_one_pass}/{len(problems)})")
        print(f"Time Elapsed:                    {elapsed:.1f}s")
        print(f"Gold Verified Shard Dataset:     {output_verified_path}")
        print(f"{'='*70}\n")

        return {
            "num_problems": len(problems),
            "total_candidates": total_attempts,
            "verified_count": len(all_verified_solutions),
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k,
            "elapsed_sec": elapsed,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/152m_wsd_10m_baseline/best_model.pt")
    parser.add_argument("--problems", type=str, default="data/rsi/problem_bank.json")
    parser.add_argument("-k", "--candidates", type=int, default=3)
    parser.add_argument("--max_problems", type=int, default=10, help="Subset of problems for fast run")
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()

    engine = SelfPlayEngine(checkpoint_path=args.checkpoint)
    engine.run_self_play(
        problem_bank_path=args.problems,
        candidates_per_problem=args.candidates,
        max_problems=args.max_problems,
        temperature=args.temp,
    )
