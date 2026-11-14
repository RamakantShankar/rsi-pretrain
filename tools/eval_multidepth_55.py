#!/usr/bin/env python3
"""
Mirendil-84M Multi-Depth Looped Transformer 55-Task Benchmark Evaluator
Evaluates the multi-depth checkpoint across all 55 tasks in data/rsi/problem_bank.json.
Tests AST syntax validity and execution across loop depths (T=1, T=2).
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
from src.model.transformer import ModelConfig
from src.model.looped_transformer import LoopedTransformerDecoder
from src.rsi.verifier import CodeVerifier, extract_function_block
from generate import top_p_sampling


def format_prompt(raw_prompt: str) -> str:
    """Ensures prompt ends with 4-space indentation to prime function body."""
    p = raw_prompt.rstrip()
    return p + "\n    "


def evaluate_multidepth_55(
    checkpoint_path: str = "checkpoints/champion/mirendil_84m_multidepth.pt",
    tokenizer_path: str = "checkpoints/tokenizers/mirendil_16k.json",
    problem_bank_path: str = "data/rsi/problem_bank.json",
    output_path: str = "data/rsi/multidepth_eval_results.json",
    num_candidates: int = 5,
    num_loops: int = 2,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 96,
    use_wandb: bool = False,
    wandb_project: str = "rsi-coding-pretrain",
    wandb_name: str = "eval_multidepth_55",
    chatml_mode: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*75}")
    print(f"MIRENDIL-84M MULTI-DEPTH: 55-TASK BENCHMARK EVALUATION (T={num_loops}, K={num_candidates})")
    print(f"Device:      {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Tokenizer:   {tokenizer_path}")
    print(f"Recurrence:  T={num_loops} ({num_loops * 8} Virtual Layers)")
    print(f"Sampling:    K={num_candidates}, Temp={temperature}, Top-p={top_p}")
    print(f"{'='*75}\n")

    # 1. Load Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    eot_id = tok.token_to_id("<|endoftext|>")
    im_end_id = tok.token_to_id("<|im_end|>")
    stop_ids = {eot_id}
    if im_end_id is not None:
        stop_ids.add(im_end_id)
    vocab_size = tok.get_vocab_size()

    # 2. Load Model Architecture
    cfg = ModelConfig(
        vocab_size=vocab_size,
        dim=896,
        n_layers=8,
        n_heads=14,
        n_kv_heads=2,
        max_seq_len=512,
        rope_theta=100000.0,
        use_qk_norm=True,
    )
    model = LoopedTransformerDecoder(cfg, default_loops=num_loops).to(device)

    # 3. Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 4. Load Problem Bank
    with open(problem_bank_path, "r", encoding="utf-8") as f:
        problem_bank = json.load(f)

    # Setup W&B if enabled
    wandb_table = None
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            job_type="eval",
            config={
                "checkpoint": checkpoint_path,
                "num_loops": num_loops,
                "num_candidates": num_candidates,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "num_tasks": len(problem_bank),
            }
        )
        wandb_table = wandb.Table(
            columns=["task_id", "category", "first_passed", "any_passed", "syntax_valid", "best_code"]
        )

    verifier = CodeVerifier(timeout_sec=2.0)
    total_tasks = len(problem_bank)
    passed_at_1_count = 0
    passed_at_5_count = 0
    valid_syntax_count = 0
    total_candidates_generated = 0
    eval_records = []

    t_eval_start = time.time()

    for idx, prob in enumerate(problem_bank, 1):
        task_id = prob["task_id"]
        category = prob.get("category", "General")
        raw_prompt = prob["prompt"]
        unit_tests = prob.get("tests", prob.get("unit_tests", []))
        if chatml_mode:
            primed_prompt = f"<|im_start|>user\n{raw_prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"
        else:
            primed_prompt = format_prompt(raw_prompt)

        task_passed = False
        first_passed = False
        task_ast_valid = False

        for c_idx in range(num_candidates):
            enc = tok.encode(primed_prompt)
            x = torch.tensor([enc.ids], dtype=torch.long, device=device)
            gen_ids = []

            with torch.no_grad():
                for step in range(max_tokens):
                    logits, _ = model(x, num_loops=num_loops)
                    next_tok = top_p_sampling(logits[:, -1, :], temperature=temperature, top_p=top_p)
                    tok_id = next_tok.item()
                    if tok_id in stop_ids and step >= 2:
                        break
                    gen_ids.append(tok_id)
                    x = torch.cat([x, next_tok], dim=1)

            body = tok.decode(gen_ids)
            for token_str in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                body = body.replace(token_str, "")
            stripped_body = body.lstrip()
            full_raw = raw_prompt.rstrip() + "\n    " + stripped_body

            cleaned_code = extract_function_block(full_raw)
            total_candidates_generated += 1

            # AST validation
            try:
                ast.parse(cleaned_code)
                task_ast_valid = True
            except SyntaxError:
                pass

            # Unit test verification
            ver = verifier.verify(cleaned_code, unit_tests)
            if ver.passed:
                task_passed = True
                if c_idx == 0:
                    first_passed = True
                break

        if first_passed:
            passed_at_1_count += 1
        if task_passed:
            passed_at_5_count += 1
        if task_ast_valid:
            valid_syntax_count += 1

        if wandb_table:
            wandb_table.add_data(
                task_id, category, first_passed, task_passed, task_ast_valid, cleaned_code[:200]
            )

        print(f"[{idx:2d}/{total_tasks:2d}] {task_id:<28} | Cat: {category:<14} | Pass@5: {'PASS' if task_passed else 'FAIL'} | Syntax: {'VALID' if task_ast_valid else 'ERR'}")

    total_eval_time = time.time() - t_eval_start
    p1_pct = (passed_at_1_count / total_tasks) * 100
    p5_pct = (passed_at_5_count / total_tasks) * 100
    syntax_pct = (valid_syntax_count / total_tasks) * 100

    print(f"\n{'='*75}")
    print("55-TASK BENCHMARK SUMMARY (MIRENDIL-84M MULTI-DEPTH)")
    print(f"{'='*75}")
    print(f"Total Benchmark Tasks:     {total_tasks}")
    print(f"Total Rollouts Evaluated:  {total_candidates_generated} (K={num_candidates}/task)")
    print(f"Total Evaluation Time:     {total_eval_time:.1f}s ({total_eval_time/total_tasks:.2f}s / task)")
    print(f"AST Valid Syntax Rate:     {valid_syntax_count}/{total_tasks} ({syntax_pct:.1f}%)")
    print(f"Pass@1 Accuracy:           {passed_at_1_count}/{total_tasks} ({p1_pct:.1f}%)")
    print(f"Pass@5 Accuracy:           {passed_at_5_count}/{total_tasks} ({p5_pct:.1f}%)")
    print(f"{'='*75}\n")

    if use_wandb:
        import wandb
        wandb.log({
            "eval/pass_at_1": p1_pct,
            f"eval/pass_at_{num_candidates}": p5_pct,
            "eval/syntax_valid_pct": syntax_pct,
            "eval/total_tasks": total_tasks,
            "eval/total_time_sec": total_eval_time,
            "eval/results_table": wandb_table,
        })
        wandb.summary["pass_at_1"] = p1_pct
        wandb.summary[f"pass_at_{num_candidates}"] = p5_pct
        wandb.summary["syntax_valid_pct"] = syntax_pct
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/champion/mirendil_84m_multidepth.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizers/mirendil_16k.json")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", type=str, default="eval_multidepth_55")
    parser.add_argument("--chatml", action="store_true", help="Evaluate with consistent <|im_start|> ChatML prompt format")
    args = parser.parse_args()
    evaluate_multidepth_55(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        num_loops=args.loops,
        num_candidates=args.k,
        use_wandb=args.wandb,
        wandb_name=args.name,
        chatml_mode=args.chatml
    )
