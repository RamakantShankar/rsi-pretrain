"""
Empirical Self-Correction Probe for Model M_5.
Tests whether M_5 can use unit test feedback / tracebacks to correct its own code.
"""

import sys
import torch
from pathlib import Path
from src.model.transformer import TransformerDecoder, ModelConfig
from src.tokenizer.bpe import CodeTokenizer
from src.rsi.verifier import CodeVerifier
from generate import generate


def run_probe(checkpoint_path: str = "checkpoints/152m_rsi_m5/best_model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"EMPIRICAL SELF-CORRECTION PROBE ON MODEL M_5 ({device.upper()})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = ModelConfig(**config)
    model = TransformerDecoder(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = CodeTokenizer()
    verifier = CodeVerifier(timeout_sec=2.0)

    probe_tasks = [
        {
            "name": "reverse_string",
            "prompt": 'def reverse_string(s: str) -> str:\n    """Return the reversed string."""\n',
            "tests": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
        },
        {
            "name": "is_palindrome",
            "prompt": 'def is_palindrome(s: str) -> bool:\n    """Return True if s is a palindrome."""\n',
            "tests": "assert is_palindrome('radar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('') == True",
        },
        {
            "name": "factorial",
            "prompt": 'def factorial(n: int) -> int:\n    """Return the factorial of n."""\n',
            "tests": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
        },
    ]

    for idx, task in enumerate(probe_tasks, 1):
        name = task["name"]
        prompt = task["prompt"]
        tests = task["tests"]

        print(f"\n--- [TASK {idx}/3: {name}] ---")

        # -------------------------------------------------------------
        # Turn 1: Zero-shot generation
        # -------------------------------------------------------------
        turn1_code = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            device=device,
        )
        res1 = verifier.verify(turn1_code, tests)
        print(f"Turn 1 (Zero-Shot Completion):")
        print(f"  Code:  {repr(turn1_code.strip())}")
        print(f"  Verdict: {'PASS' if res1.passed else f'FAIL ({res1.error_type}: {res1.error_msg[:60]}...)'}")

        if res1.passed:
            print("  --> Already passed zero-shot!")
            continue

        # -------------------------------------------------------------
        # Turn 2: Conditioned on unit test error / traceback
        # -------------------------------------------------------------
        # Format the self-correction prompt showing the bug and asking for the fix
        error_brief = f"{res1.error_type}: {res1.error_msg.splitlines()[0] if res1.error_msg else 'Assertion failed'}"
        turn2_prompt = (
            f"# Buggy implementation:\n"
            f"{turn1_code.strip()}\n\n"
            f"# Unit test failed:\n"
            f"# {error_brief}\n\n"
            f"# Fixed implementation:\n"
            f"{prompt}"
        )

        turn2_code = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=turn2_prompt,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            device=device,
        )
        res2 = verifier.verify(turn2_code, tests)
        print(f"\nTurn 2 (Self-Correction with Traceback Feedback):")
        print(f"  Prompt Context Provided:\n{turn2_prompt}")
        print(f"  Corrected Code: {repr(turn2_code.strip())}")
        print(f"  Verdict: {'PASS (Self-Correction Succeeded!)' if res2.passed else f'FAIL ({res2.error_type}: {res2.error_msg[:60]}...)'}")

    print(f"\n{'='*70}")
    print("PROBE EXECUTION FINISHED.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/152m_rsi_m5/best_model.pt"
    run_probe(ckpt)
