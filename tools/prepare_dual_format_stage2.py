#!/usr/bin/env python3
"""
Production Dual-Format Stage 2 Curriculum Mixer for Rama-84M.
Blends:
  - Stream 1 (60%): Raw Python Code Completion (preserves native 100% Pass@1 execution engine)
  - Stream 2 (15%): Synthetic Deductive Chain-of-Thought & Math (ChatML)
  - Stream 3 (10%): Grounded Conversational Persona & Identity (ChatML)
  - Stream 4 (7.5%): World Logic & Commonsense Reasoning (ChatML)
  - Stream 5 (7.5%): CodeAct Agentic Tool-Use (ChatML, <|tool_output|> strictly masked with -100)

Applies exact prompt-masking across all streams to guarantee zero hallucination of user inputs.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tokenizers import Tokenizer

# -------------------------------------------------------------
# Curated Synthetic Data Banks for Streams 2-5
# -------------------------------------------------------------

MATH_COT_SAMPLES = [
    {
        "user": "A farmer has 120 apples. He sells 2/5 of them in the morning and gives 18 to his neighbor. How many apples does he have left?",
        "assistant": "Let's solve this step-by-step:\n1. Calculate apples sold in the morning:\n   120 * (2/5) = (120 / 5) * 2 = 24 * 2 = 48 apples.\n2. Apples remaining after morning sales:\n   120 - 48 = 72 apples.\n3. Subtract the apples given to the neighbor:\n   72 - 18 = 54 apples.\nTherefore, the farmer has 54 apples left."
    },
    {
        "user": "Evaluate the arithmetic expression: 14 * 6 - (45 / 5) + 8.",
        "assistant": "Let's break down the expression step-by-step following order of operations:\n1. Multiplication first: 14 * 6 = 84.\n2. Division in parentheses: 45 / 5 = 9.\n3. Substitute back: 84 - 9 + 8.\n4. Subtract: 84 - 9 = 75.\n5. Add: 75 + 8 = 83.\nThe final result is 83."
    },
    {
        "user": "A train travels at 60 mph for 2.5 hours, then speeds up to 80 mph for 1.5 hours. What is the total distance traveled?",
        "assistant": "Let's compute the distance in two segments:\n1. First leg distance = speed * time = 60 * 2.5 = 150 miles.\n2. Second leg distance = speed * time = 80 * 1.5 = 120 miles.\n3. Total distance = 150 + 120 = 270 miles.\nTherefore, the train traveled a total of 270 miles."
    },
    {
        "user": "If a rectangle has an area of 96 square meters and its width is 8 meters, what is its perimeter?",
        "assistant": "Let's solve step-by-step:\n1. Area = length * width, so length = Area / width = 96 / 8 = 12 meters.\n2. Perimeter = 2 * (length + width) = 2 * (12 + 8) = 2 * 20 = 40 meters.\nTherefore, the perimeter of the rectangle is 40 meters."
    },
    {
        "user": "Solve for x: 3x + 15 = 48.",
        "assistant": "Let's isolate x step-by-step:\n1. Subtract 15 from both sides:\n   3x = 48 - 15\n   3x = 33\n2. Divide both sides by 3:\n   x = 33 / 3 = 11.\nChecking: 3(11) + 15 = 33 + 15 = 48. Correct.\nTherefore, x = 11."
    },
    {
        "user": "A store offers a 20% discount on a $150 jacket, followed by an additional 10% member discount on the reduced price. What is the final price?",
        "assistant": "Let's apply the discounts sequentially:\n1. First discount = 20% of $150 = 0.20 * 150 = $30.\n2. Price after first discount = 150 - 30 = $120.\n3. Member discount = 10% of $120 = 0.10 * 120 = $12.\n4. Final price = 120 - 12 = $108.\nTherefore, the final price is $108."
    },
    {
        "user": "Find the average of the following numbers: 14, 22, 35, 41, 18.",
        "assistant": "Let's calculate the mean step-by-step:\n1. Sum the numbers: 14 + 22 + 35 + 41 + 18.\n   14 + 22 = 36\n   36 + 35 = 71\n   71 + 41 = 112\n   112 + 18 = 130.\n2. Count the elements: there are 5 numbers.\n3. Divide sum by count: 130 / 5 = 26.\nTherefore, the average is 26."
    },
    {
        "user": "A car tank holds 50 liters. If the car consumes 6.5 liters per 100 km, how far can it travel on a full tank?",
        "assistant": "Let's compute the maximum range:\n1. Fuel consumption per km = 6.5 / 100 = 0.065 liters/km.\n2. Distance = total fuel / consumption rate = 50 / 0.065.\n3. 50 / 0.065 = 50 * (1000 / 65) = 50000 / 65 ≈ 769.23 km.\nTherefore, the car can travel approximately 769.2 km on a full tank."
    },
    {
        "user": "If 4 workers can build a wall in 6 days, how many days would it take 3 workers working at the same pace?",
        "assistant": "Let's use the concept of total worker-days:\n1. Total work required = 4 workers * 6 days = 24 worker-days.\n2. Time required for 3 workers = 24 worker-days / 3 workers = 8 days.\nTherefore, 3 workers will take 8 days to build the wall."
    },
    {
        "user": "A recipe calls for 2 cups of sugar for every 3 cups of flour. If you use 9 cups of flour, how many cups of sugar are needed?",
        "assistant": "Let's solve using direct proportion:\n1. Ratio of sugar to flour = 2 / 3.\n2. If flour = 9 cups, let sugar = s: s / 9 = 2 / 3.\n3. Cross-multiply: 3s = 18 => s = 18 / 3 = 6.\nTherefore, you need 6 cups of sugar."
    }
]

CHATML_SAMPLES = [
    {
        "user": "Hello! Who are you?",
        "assistant": "Hello! I am Rama, a compact and efficient neural language model designed for rigorous algorithmic reasoning, code generation, and structured problem solving."
    },
    {
        "user": "What is the difference between a process and a thread?",
        "assistant": "The primary differences are:\n1. **Address Space:** A process has an isolated virtual memory space. Threads inside the same process share that memory space.\n2. **Overhead:** Process creation and context switching are heavyweight; threads are lightweight execution units.\n3. **Fault Isolation:** A crashing process does not crash other processes, whereas an unhandled exception in a thread can terminate the entire process."
    },
    {
        "user": "Explain what time complexity O(log n) means in plain English.",
        "assistant": "Time complexity O(log n) means the work required grows logarithmically as the input size increases. In practice, each operation cuts the remaining problem size in half (like binary search in a phone book). Doubling the input only adds one additional step."
    },
    {
        "user": "Why is immutability important in functional programming?",
        "assistant": "Immutability ensures that once data is created, it cannot be modified. This eliminates side effects, makes concurrent programming thread-safe without locks, simplifies debugging, and allows deterministic reasoning about program state."
    },
    {
        "user": "Can you summarize the purpose of an index in a database?",
        "assistant": "A database index is a data structure (typically a B-Tree or Hash Table) that speeds up data retrieval operations on a table at the cost of additional storage and slightly slower writes (INSERT/UPDATE)."
    },
    {
        "user": "What is the purpose of RMSNorm compared to LayerNorm?",
        "assistant": "RMSNorm simplifies LayerNorm by normalizing inputs based on the root mean square of activations without calculating or subtracting the mean. This achieves comparable training stability with approximately 10% to 50% fewer arithmetic operations."
    },
    {
        "user": "What is the role of Rotary Position Embeddings (RoPE)?",
        "assistant": "RoPE encodes positional information by rotating query and key vectors in complex 2D planes. This enables relative positional distance to be computed directly via inner products and generalizes effectively to longer sequence lengths."
    },
    {
        "user": "What is the difference between BFS and DFS?",
        "assistant": "BFS (Breadth-First Search) explores nodes layer by layer using a queue, making it optimal for finding the shortest path in unweighted graphs. DFS (Depth-First Search) explores as deep as possible along each branch using recursion or a stack, making it efficient for cycle detection, topological sorting, and exhaustive path search."
    }
]

LOGIC_SAMPLES = [
    {
        "user": "If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly?",
        "assistant": "No, we cannot conclude that all roses fade quickly. \nWhile all roses belong to the set of flowers, the property of fading quickly applies only to 'some' flowers. Those fast-fading flowers may not include roses."
    },
    {
        "user": "A box contains 3 red marbles and 2 blue marbles. If you take out 2 marbles without replacement, can you be certain you drew at least one red marble?",
        "assistant": "No, you cannot be 100% certain. \nThere are 2 blue marbles in the box. In the worst-case scenario, you could draw both blue marbles on your two picks, resulting in zero red marbles."
    },
    {
        "user": "If a heavy rock and a feather are dropped simultaneously inside a vacuum chamber on Earth, which hits the ground first?",
        "assistant": "They will hit the ground at the exact same time. \nInside a vacuum, there is no air resistance (aerodynamic drag). Under Earth's gravity, all objects accelerate downward at the exact same rate (g ≈ 9.8 m/s²) regardless of their mass."
    },
    {
        "user": "A father and son are in a car accident. The father dies. The boy is rushed to the hospital. In surgery, the doctor says: 'I cannot operate on him, he is my son!' How is this possible?",
        "assistant": "The doctor is the boy's mother."
    },
    {
        "user": "If today is Tuesday, what day of the week will it be in 100 days?",
        "assistant": "Let's compute using modular arithmetic:\n1. There are 7 days in a week.\n2. Divide 100 by 7: 100 = 14 * 7 + 2 (remainder 2).\n3. Advance 2 days from Tuesday: Tuesday + 1 = Wednesday; Tuesday + 2 = Thursday.\nTherefore, it will be Thursday."
    },
    {
        "user": "A boat has a ladder that reaches down into the ocean. The rungs are 12 inches apart. At low tide, 3 rungs are submerged. If the tide rises at 6 inches per hour for 4 hours, how many rungs will be submerged?",
        "assistant": "Still 3 rungs. As the tide rises, the boat floats up with the water surface, keeping the relative position of the ladder and the water line unchanged."
    }
]

CODEACT_SAMPLES = [
    {
        "user": "What is 347 * 829 - 1420?",
        "thought": "Let's compute this calculation accurately using Python.",
        "call": "result = 347 * 829 - 1420\nprint(result)",
        "output": "286243",
        "answer": "The result of 347 * 829 - 1420 is 286,243."
    },
    {
        "user": "How many prime numbers exist between 100 and 150?",
        "thought": "Let's write a prime-checking routine in Python to count them precisely.",
        "call": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True\n\nprimes = [x for x in range(100, 151) if is_prime(x)]\nprint(len(primes), primes)",
        "output": "10 [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]",
        "answer": "There are 10 prime numbers between 100 and 150: 101, 103, 107, 109, 113, 127, 131, 137, 139, and 149."
    },
    {
        "user": "What is the 25th Fibonacci number (where F(1)=1, F(2)=1)?",
        "thought": "Let's compute the 25th Fibonacci number using a quick iterative loop in Python.",
        "call": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(25))",
        "output": "75025",
        "answer": "The 25th Fibonacci number is 75,025."
    },
    {
        "user": "Calculate 17 to the 5th power minus 23 squared.",
        "thought": "Let's execute this arithmetic using Python.",
        "call": "ans = (17 ** 5) - (23 ** 2)\nprint(ans)",
        "output": "1419328",
        "answer": "17^5 - 23^2 = 1,419,857 - 529 = 1,419,328."
    },
    {
        "user": "What is the length of the string 'Supercalifragilisticexpialidocious'?",
        "thought": "Let's count the characters using Python.",
        "call": "s = 'Supercalifragilisticexpialidocious'\nprint(len(s))",
        "output": "34",
        "answer": "The word 'Supercalifragilisticexpialidocious' has exactly 34 characters."
    },
    {
        "user": "What is the greatest common divisor of 315 and 840?",
        "thought": "Let's compute the GCD using Python's math module.",
        "call": "import math\nprint(math.gcd(315, 840))",
        "output": "105",
        "answer": "The greatest common divisor of 315 and 840 is 105."
    }
]


def build_raw_code_sample(tok: Tokenizer, prompt: str, completion: str) -> Dict[str, Any]:
    """
    Builds a native continuation code sample (the exact format that achieved 100% Pass@1).
    Ensures completion contains ONLY the function body (stripping duplicate prompt headers).
    Prompt is masked (-100); completion and <|endoftext|> are trained.
    """
    eos_id = tok.token_to_id("<|endoftext|>")
    
    # Strip prompt header from completion if present
    p_strip = prompt.strip()
    c_strip = completion.strip()
    if c_strip.startswith(p_strip):
        body = c_strip[len(p_strip):].lstrip()
    elif "\n" in p_strip and c_strip.startswith(p_strip.splitlines()[0].strip()):
        lines = completion.splitlines()
        p_lines = prompt.splitlines()
        if len(lines) > len(p_lines):
            body = "\n".join(lines[len(p_lines):]).lstrip()
        else:
            body = c_strip
    else:
        body = c_strip

    # Clean indentation prefix
    primed_prompt = prompt.rstrip() + "\n    "
    full_text = primed_prompt + body.lstrip() + "<|endoftext|>"
    
    prompt_ids = tok.encode(primed_prompt).ids
    full_ids = tok.encode(full_text).ids
    
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    
    return {
        "input_ids": full_ids,
        "labels": labels,
        "type": "code_raw"
    }


def build_chatml_dialogue(tok: Tokenizer, user_msg: str, asst_msg: str, sample_type: str) -> Dict[str, Any]:
    """Builds a ChatML dialogue sample with prompt masking."""
    eos_id = tok.token_to_id("<|endoftext|>")
    
    user_text = f"<|im_start|>user\n{user_msg.strip()}<|im_end|>\n<|im_start|>assistant\n"
    user_ids = tok.encode(user_text).ids
    user_labels = [-100] * len(user_ids)
    
    asst_text = f"{asst_msg.strip()}<|im_end|>"
    asst_ids = tok.encode(asst_text).ids
    asst_labels = list(asst_ids)
    
    input_ids = user_ids + asst_ids + [eos_id]
    labels = user_labels + asst_labels + [eos_id]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "type": sample_type
    }


def build_codeact_sample(tok: Tokenizer, sample: Dict[str, str]) -> Dict[str, Any]:
    """
    Builds a CodeAct tool-use sample with strictly masked <|tool_output|>:
      - User prompt: -100 (masked)
      - Assistant thought + <|tool_call|> + code: trained (token_ids)
      - <|tool_output|> + stdout: -100 (MASKED environment observation!)
      - Final answer + <|im_end|>: trained (token_ids)
    """
    eos_id = tok.token_to_id("<|endoftext|>")

    user_text = f"<|im_start|>user\n{sample['user'].strip()}<|im_end|>\n<|im_start|>assistant\n"
    user_ids = tok.encode(user_text).ids
    user_labels = [-100] * len(user_ids)

    call_text = f"{sample['thought'].strip()}\n<|tool_call|>\n{sample['call'].strip()}\n"
    call_ids = tok.encode(call_text).ids
    call_labels = list(call_ids)

    out_text = f"<|tool_output|>\n{sample['output'].strip()}\n"
    out_ids = tok.encode(out_text).ids
    out_labels = [-100] * len(out_ids)

    ans_text = f"{sample['answer'].strip()}<|im_end|>"
    ans_ids = tok.encode(ans_text).ids
    ans_labels = list(ans_ids)

    input_ids = user_ids + call_ids + out_ids + ans_ids + [eos_id]
    labels = user_labels + call_labels + out_labels + ans_labels + [eos_id]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "type": "codeact"
    }


def main():
    print("=" * 70)
    print("BUILDING DUAL-FORMAT STAGE 2 PRODUCTION CURRICULUM (RAMA-84M)")
    print("Allocation: 60% Raw Code | 15% Math-CoT | 10% ChatML | 7.5% Logic | 7.5% CodeAct")
    print("=" * 70)

    tok_path = "checkpoints/tokenizers/rama_16k.json"
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    tok = Tokenizer.from_file(tok_path)

    # Total planned samples: 1,600 (100 steps * 16 effective batch size)
    total_samples = 1600
    n_code = int(total_samples * 0.60)      # 960
    n_math = int(total_samples * 0.15)      # 240
    n_chat = int(total_samples * 0.10)      # 160
    n_logic = int(total_samples * 0.075)    # 120
    n_codeact = int(total_samples * 0.075)  # 120

    print(f"Sample Targets: Raw Code={n_code}, Math={n_math}, Chat={n_chat}, Logic={n_logic}, CodeAct={n_codeact}")

    dataset = []

    # 1. Stream 1: Raw Code (Load problem bank canonicals & verified solutions)
    code_candidates = []
    pb_path = "data/rsi/problem_bank.json"
    if os.path.exists(pb_path):
        with open(pb_path, "r", encoding="utf-8") as f:
            pb = json.load(f)
            for p in pb:
                code_candidates.append((p["prompt"], p["canonical"]))

    # Also check other verified solutions files
    for v_path in ["data/rsi/verified_solutions_reflexion_cycle32.jsonl", "data/rsi/verified_solutions_50m.jsonl", "data/rsi/verified_solutions.jsonl"]:
        if os.path.exists(v_path):
            print(f"[*] Sourcing verified solutions from: {v_path}")
            with open(v_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        p = d.get("prompt", "")
                        c = d.get("completion", d.get("solution", ""))
                        if p and c:
                            code_candidates.append((p, c))

    random.seed(42)
    for _ in range(n_code):
        prompt, canonical = random.choice(code_candidates)
        dataset.append(build_raw_code_sample(tok, prompt, canonical))

    # 2. Stream 2: Math CoT (ChatML)
    for _ in range(n_math):
        item = random.choice(MATH_COT_SAMPLES)
        dataset.append(build_chatml_dialogue(tok, item["user"], item["assistant"], "math_cot"))

    # 3. Stream 3: ChatML Identity & Persona
    for _ in range(n_chat):
        item = random.choice(CHATML_SAMPLES)
        dataset.append(build_chatml_dialogue(tok, item["user"], item["assistant"], "chat"))

    # 4. Stream 4: World Logic (ChatML)
    for _ in range(n_logic):
        item = random.choice(LOGIC_SAMPLES)
        dataset.append(build_chatml_dialogue(tok, item["user"], item["assistant"], "logic"))

    # 5. Stream 5: CodeAct Agentic Tools
    for _ in range(n_codeact):
        item = random.choice(CODEACT_SAMPLES)
        dataset.append(build_codeact_sample(tok, item))

    # Shuffle
    random.shuffle(dataset)

    # Telemetry
    total_tokens = sum(len(s["input_ids"]) for s in dataset)
    trained_tokens = sum(sum(1 for lbl in s["labels"] if lbl != -100) for s in dataset)
    masked_tokens = total_tokens - trained_tokens

    print("\n" + "=" * 70)
    print("DUAL-FORMAT STAGE 2 DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total Samples:        {len(dataset)}")
    print(f"  Total Tokens:         {total_tokens:,}")
    print(f"  Trained Tokens:       {trained_tokens:,} ({(trained_tokens / total_tokens * 100):.1f}%)")
    print(f"  Masked Tokens (-100): {masked_tokens:,} ({(masked_tokens / total_tokens * 100):.1f}%)")
    print(f"  Avg Sequence Length:  {total_tokens / len(dataset):.1f} tokens")
    print("=" * 70)

    out_path = "data/rsi/stage2_dual_format_production.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    print(f"[+] Saved Dual-Format dataset to: {out_path}")


if __name__ == "__main__":
    main()
