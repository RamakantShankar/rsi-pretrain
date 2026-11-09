#!/usr/bin/env python3
"""
Mirendil-154M Champion 55-Task Benchmark Evaluator
Evaluates the champion checkpoint across all 55 tasks in data/rsi/problem_bank.json.
Supports:
  - Standard HumanEval function body priming (4-space indentation)
  - Multi-candidate sampling (K=5 for Pass@5, matching Gen 0 / Gen 1 protocol)
  - AST Valid Syntax Rate (%)
  - Full Category Breakdown
"""

import os
import sys
import json
import time
import ast
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from tokenizers import Tokenizer
from src.model.transformer import ModelConfig, TransformerDecoder
from src.rsi.verifier import CodeVerifier, extract_function_block
from generate import top_p_sampling


def format_prompt(raw_prompt: str) -> str:
    """Ensures prompt ends with 4-space indentation to prime function body."""
    p = raw_prompt.rstrip()
    return p + "\n    "


def clean_completion(completion: str, prompt: str) -> str:
    """Ensures generated lines maintain proper indentation."""
    full = prompt + completion
    return extract_function_block(full)


def evaluate_champion_55(
    checkpoint_path: str = "checkpoints/champion/mirendil_154m_champion.pt",
    tokenizer_path: str = "checkpoints/tokenizers/mirendil_16k.json",
    problem_bank_path: str = "data/rsi/problem_bank.json",
    output_path: str = "data/rsi/champion_eval_results.json",
    num_candidates: int = 5,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 96,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*75}")
    print(f"MIRENDIL-154M CHAMPION: 55-TASK BENCHMARK EVALUATION (K={num_candidates})")
    print(f"Device:      {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Tokenizer:   {tokenizer_path}")
    print(f"Sampling:    K={num_candidates}, Temp={temperature}, Top-p={top_p}")
    print(f"{'='*75}\n")

    # 1. Load Tokenizer & Model
    tok = Tokenizer.from_file(tokenizer_path)
    eot_id = tok.token_to_id("<|endoftext|>")

    cfg = ModelConfig(
        vocab_size=16384,
        dim=896,
        n_layers=16,
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=512,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    model = TransformerDecoder(cfg).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Problem Bank
    with open(problem_bank_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    verifier = CodeVerifier(timeout_sec=2.0)
    results = []
    category_stats = {}

    t0 = time.time()

    for idx, prob in enumerate(problems, 1):
        task_id = prob["task_id"]
        category = prob.get("category", "general")
        name = prob.get("name", task_id)
        raw_prompt = prob["prompt"]
        primed_prompt = format_prompt(raw_prompt)
        unit_tests = prob["tests"]

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0, "ast_valid": 0}
        category_stats[category]["total"] += 1

        enc = tok.encode(primed_prompt)
        prompt_ids = enc.ids

        task_passed = False
        task_ast_valid = False
        best_code = ""
        candidate_records = []

        # Sample K candidates
        for k in range(num_candidates):
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            gen_ids = []
            cur_temp = 0.0 if (k == 0 and num_candidates > 1) else temperature

            with torch.no_grad():
                for step in range(max_tokens):
                    logits, _ = model(x)
                    next_logit = logits[:, -1, :]
                    if step < 4 and eot_id is not None:
                        next_logit[:, eot_id] = -float("inf")
                    next_tok = top_p_sampling(next_logit, temperature=cur_temp, top_p=top_p)
                    tok_id = next_tok.item()
                    if tok_id == eot_id and step >= 4:
                        break
                    gen_ids.append(tok_id)
                    x = torch.cat([x, next_tok], dim=1)

            body = tok.decode(gen_ids)
            # Indent first line if not already indented
            if not body.startswith(" ") and not body.startswith("\t"):
                full_raw = primed_prompt + body
            else:
                full_raw = raw_prompt.rstrip() + "\n" + body

            # Clean and extract
            cleaned_code = extract_function_block(full_raw)

            # AST validation
            try:
                ast.parse(cleaned_code)
                cand_ast = True
            except SyntaxError:
                cand_ast = False

            if cand_ast:
                task_ast_valid = True

            # Unit test verification
            ver = verifier.verify(cleaned_code, unit_tests)
            if ver.passed:
                task_passed = True
                best_code = cleaned_code
                break

            if not best_code and cand_ast:
                best_code = cleaned_code

        if task_passed:
            category_stats[category]["passed"] += 1
        if task_ast_valid:
            category_stats[category]["ast_valid"] += 1

        status_icon = "✅ PASS" if task_passed else ("⚠️ SYNTAX OK (FAIL TESTS)" if task_ast_valid else "❌ SYNTAX ERR")
        print(f"[{idx:2d}/{len(problems)}] {task_id:<12} | {category:<12} | {name:<22} | {status_icon}")

        results.append({
            "task_id": task_id,
            "category": category,
            "name": name,
            "passed": task_passed,
            "ast_valid": task_ast_valid,
            "code": best_code,
        })

    elapsed = time.time() - t0
    total_tasks = len(problems)
    total_passed = sum(1 for r in results if r["passed"])
    total_ast_valid = sum(1 for r in results if r["ast_valid"])

    pass_rate = (total_passed / total_tasks) * 100
    ast_pass_rate = (total_ast_valid / total_tasks) * 100

    print(f"\n{'='*75}")
    print(f"55-TASK BENCHMARK FINAL EVALUATION REPORT (K={num_candidates})")
    print(f"{'='*75}")
    print(f"Total Problems:            {total_tasks}")
    print(f"Total Time:                {elapsed:.1f}s ({elapsed/total_tasks:.2f}s per problem)")
    print(f"AST Valid Syntax Rate:     {ast_pass_rate:.1f}% ({total_ast_valid}/{total_tasks})")
    print(f"Pass@{num_candidates} Test Accuracy:       {pass_rate:.2f}% ({total_passed}/{total_tasks})")
    print(f"{'='*75}\n")

    print(f"{'Category':<16} | {'Tasks':<6} | {'AST Valid':<12} | {'Pass Rate':<15}")
    print("-" * 55)
    for cat, stats in sorted(category_stats.items()):
        c_tot = stats["total"]
        c_ast = (stats["ast_valid"] / c_tot) * 100
        c_pass = (stats["passed"] / c_tot) * 100
        print(f"{cat:<16} | {c_tot:<6} | {stats['ast_valid']:>2d}/{c_tot:<2d} ({c_ast:.0f}%) | {stats['passed']:>2d}/{c_tot:<2d} ({c_pass:.1f}%)")
    print("-" * 55)

    summary_data = {
        "model_name": "Mirendil-154M-Champion",
        "num_candidates": num_candidates,
        "total_tasks": total_tasks,
        "total_passed": total_passed,
        "total_ast_valid": total_ast_valid,
        "pass_rate": pass_rate,
        "ast_pass_rate": ast_pass_rate,
        "elapsed_sec": elapsed,
        "category_breakdown": category_stats,
        "results": results,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    evaluate_champion_55(num_candidates=5)
