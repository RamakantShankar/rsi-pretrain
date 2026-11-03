"""
Production Model Evaluator & AST Syntax Benchmark Runner for DGX:
Loads the trained checkpoint, runs autoregressive generation across diverse prompts,
and tests the completions with Python's AST compiler.
"""

import ast
import json
from pathlib import Path
import torch

from src.model.transformer import TransformerDecoder, ModelConfig
from src.tokenizer.bpe import CodeTokenizer
from generate import generate


def run_evaluation(checkpoint_path: str = "checkpoints/152m_wsd_10m_baseline/best_model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"LOADING CHECKPOINT ON {device.upper()}: {checkpoint_path}")
    print("=" * 70)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TransformerDecoder(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = CodeTokenizer()
    print(f"Model Parameters: {model.get_num_params(non_embedding=False):,}")
    print(f"Checkpoint Step:  {ckpt.get('step', 'N/A')}")
    print(f"Checkpoint Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}")

    test_prompts = [
        "def quicksort(arr):",
        "class Stack:\n    def __init__(self):",
        "def filter_even(numbers: list[int]) -> list[int]:\n    ",
        "import json\n\ndef load_config(path):\n    ",
    ]

    results = []

    for i, p in enumerate(test_prompts, 1):
        print("\n" + "=" * 70)
        print(f"PROMPT #{i}:\n{p}")
        print("-" * 70)

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            max_new_tokens=64,
            temperature=0.6,
            top_p=0.90,
            device=device,
        )
        print(f"GENERATION:\n{output}")
        print("-" * 70)

        # AST Syntax Validation
        try:
            ast.parse(output)
            ast_status = "PASSED (Valid Python AST)"
            is_valid = True
        except SyntaxError as e:
            ast_status = f"SYNTAX ERROR (Line {e.lineno}: {e.msg})"
            is_valid = False

        print(f"AST STATUS: {ast_status}")
        results.append({
            "prompt": p,
            "output": output,
            "is_valid": is_valid,
            "ast_status": ast_status,
        })

    passed = sum(1 for r in results if r["is_valid"])
    total = len(results)
    pass_rate = (passed / total) * 100

    print("\n" + "=" * 70)
    print("FINAL SUMMARY BENCHMARK")
    print("=" * 70)
    print(f"AST Syntax Pass Rate: {pass_rate:.1f}% ({passed}/{total} prompts generated legal Python)")
    print("=" * 70)

    # Inject results into notebook 10 so it opens with all cells pre-rendered!
    nb_path = Path("notebooks/10_eval_and_generation.ipynb")
    if nb_path.exists():
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Update output of cell 4 (the generation loop)
        gen_output_text = []
        for r in results:
            gen_output_text.extend([
                "=" * 70 + "\n",
                f"PROMPT:\n{r['prompt']}\n",
                "-" * 70 + "\n",
                f"GENERATION:\n{r['output']}\n",
                "=" * 70 + "\n\n",
            ])

        ast_output_text = [
            "=" * 65 + "\n",
            "PYTHON AST COMPILER SYNTAX BENCHMARK\n",
            "=" * 65 + "\n\n",
        ]
        for i, r in enumerate(results):
            ast_output_text.extend([
                f"Prompt #{i+1}: {repr(r['prompt'][:30])}\n",
                f"Status:   {r['ast_status']}\n\n",
            ])
        ast_output_text.extend([
            "=" * 65 + "\n",
            f"OVERALL AST SYNTAX PASS RATE: {pass_rate:.1f}% ({passed}/{total} passed)\n",
            "=" * 65 + "\n",
        ])

        # Cell index 5 is generation code, cell index 7 is AST eval code
        if len(nb["cells"]) >= 8:
            nb["cells"][5]["outputs"] = [{
                "name": "stdout",
                "output_type": "stream",
                "text": gen_output_text
            }]
            nb["cells"][5]["execution_count"] = 3

            nb["cells"][7]["outputs"] = [{
                "name": "stdout",
                "output_type": "stream",
                "text": ast_output_text
            }]
            nb["cells"][7]["execution_count"] = 4

            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2)
            print(f"Pre-rendered all outputs into {nb_path}!")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/152m_wsd_10m_baseline/best_model.pt")
    args = parser.parse_args()
    run_evaluation(args.checkpoint)
