import json
import random
from pathlib import Path
from tokenizers import Tokenizer

print("=" * 70)
print("BUILDING 90/10 PILOT DATASET MIX (CODE + CHATML)")
print("=" * 70)

tok_path = "checkpoints/tokenizers/rama_16k.json"
tok = Tokenizer.from_file(tok_path)

im_start_id = tok.token_to_id("<|im_start|>")
im_end_id = tok.token_to_id("<|im_end|>")

# 1. Load Verified Code Data (Anchor)
code_in_path = "data/rsi/verified_solutions_reflexion_cycle32.jsonl"
code_samples = []
with open(code_in_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            code_samples.append(json.loads(line))

print(f"Total available verified code solutions: {len(code_samples)}")
random.seed(42)
if len(code_samples) > 900:
    selected_code = random.sample(code_samples, 900)
else:
    selected_code = code_samples

# 2. Curate High-Density ChatML Dialogues
chat_dialogues = [
    {
        "messages": [
            {"role": "user", "content": "Hello! Who are you?"},
            {"role": "assistant", "content": "Hello! I am Rama, an intelligent and compact language model designed for efficient reasoning, coding, and algorithmic problem-solving."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is your name and what can you do?"},
            {"role": "assistant", "content": "My name is Rama. I specialize in writing clean Python code, executing multi-step logical reasoning, and assisting you with technical workflows."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Can you tell me about yourself?"},
            {"role": "assistant", "content": "I am Rama, an 84-million parameter looped transformer model. Despite my compact footprint, I utilize deep recurrent reasoning to solve complex problems with high precision."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hi Rama, nice to meet you."},
            {"role": "assistant", "content": "Nice to meet you! How can I assist you with your code or reasoning tasks today?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Are you an AI?"},
            {"role": "assistant", "content": "Yes, I am Rama, an AI assistant developed for fast, efficient coding and step-by-step reasoning."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is 15 multiplied by 4, and then minus 7?"},
            {"role": "assistant", "content": "Let us calculate this step by step:\n1. First, multiply 15 by 4: 15 * 4 = 60.\n2. Next, subtract 7 from 60: 60 - 7 = 53.\nThe final result is 53."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "If a train travels at 60 mph for 2.5 hours, how far does it go?"},
            {"role": "assistant", "content": "To find the total distance:\n1. Distance = Speed * Time.\n2. Here, Speed = 60 miles per hour, and Time = 2.5 hours.\n3. 60 * 2.5 = 150 miles.\nThe train travels 150 miles."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Is 29 a prime number? Explain your reasoning."},
            {"role": "assistant", "content": "Let us check if 29 is prime:\n1. A prime number has no positive divisors other than 1 and itself.\n2. The square root of 29 is approximately 5.38, so we only need to test prime divisors up to 5: 2, 3, and 5.\n3. 29 is odd, so not divisible by 2.\n4. Sum of digits is 2 + 9 = 11 (not divisible by 3).\n5. It does not end in 0 or 5, so not divisible by 5.\nSince none divide 29, 29 is indeed a prime number."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Which is larger: 3/4 or 5/8?"},
            {"role": "assistant", "content": "Let us compare the two fractions by finding a common denominator:\n1. The common denominator for 4 and 8 is 8.\n2. Convert 3/4: (3 * 2) / (4 * 2) = 6/8.\n3. Now compare 6/8 and 5/8.\nSince 6 is greater than 5, 3/4 is larger than 5/8."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain the difference between a list and a tuple in Python in two bullet points."},
            {"role": "assistant", "content": "Here is the key difference:\n* Mutability: Lists are mutable (elements can be added or modified), while tuples are immutable (cannot be changed after creation).\n* Syntax: Lists use square brackets [1, 2, 3], whereas tuples use parentheses (1, 2, 3)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What does strip() do to a string in Python?"},
            {"role": "assistant", "content": "In Python, str.strip() returns a copy of the string with all leading and trailing whitespace (spaces, tabs, newlines) removed."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do you check if a key exists in a Python dictionary?"},
            {"role": "assistant", "content": "You can use the in keyword: if key in my_dict: ... Alternatively, you can use my_dict.get(key)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hi Rama, can you help me write a Python function?"},
            {"role": "assistant", "content": "Of course! What function would you like to create?"},
            {"role": "user", "content": "A function to count vowels in a word."},
            {"role": "assistant", "content": "Here is a clean implementation:\ndef count_vowels(s: str) -> int:\n    vowels = set('aeiouAEIOU')\n    return sum(1 for char in s if char in vowels)"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "assistant", "content": "The capital of Japan is Tokyo."},
            {"role": "user", "content": "And what is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
]

variations = []
for _ in range(7):
    for item in chat_dialogues:
        variations.append(item)
variations = variations[:100]

print(f"Total curated ChatML dialogues: {len(variations)}")

combined_samples = []

for sample in selected_code:
    prompt = sample.get("prompt", "")
    # Use completion (function body only), NOT solution (full function)
    completion = sample.get("completion", "")
    if not completion and "solution" in sample:
        # Fallback if completion missing: strip prompt from solution
        sol = sample["solution"]
        if sol.startswith(prompt.strip()):
            completion = sol[len(prompt.strip()):]
        else:
            completion = sol
            
    full_text = prompt.rstrip() + "\n    " + completion.lstrip() + "<|endoftext|>"
    
    prompt_ids = tok.encode(prompt.rstrip() + "\n    ").ids
    full_ids = tok.encode(full_text).ids
    
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    
    combined_samples.append({
        "type": "code",
        "input_ids": full_ids,
        "labels": labels
    })

for chat in variations:
    full_input_ids = []
    labels = []
    
    for msg in chat["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        turn_header = f"<|im_start|>{role}\n"
        turn_body = f"{content}<|im_end|>\n" if role in ("user", "system") else f"{content}<|im_end|>"
        
        header_ids = tok.encode(turn_header).ids
        body_ids = tok.encode(turn_body).ids
        
        full_input_ids.extend(header_ids)
        full_input_ids.extend(body_ids)
        
        if role in ("user", "system"):
            labels.extend([-100] * (len(header_ids) + len(body_ids)))
        elif role == "assistant":
            labels.extend([-100] * len(header_ids))
            labels.extend(body_ids)
            
    combined_samples.append({
        "type": "chat",
        "input_ids": full_input_ids,
        "labels": labels
    })

random.shuffle(combined_samples)

out_dataset_path = "data/rsi/pilot_mix_90_10_rama.json"
with open(out_dataset_path, "w", encoding="utf-8") as f:
    json.dump(combined_samples, f)

print(f"\nSuccessfully created Pilot Mix dataset: {out_dataset_path}")
print(f"Total Samples: {len(combined_samples)} (Code: {len(selected_code)}, Chat: {len(variations)})")
print(f"Ratio: {len(selected_code) / len(combined_samples) * 100:.1f}% Code / {len(variations) / len(combined_samples) * 100:.1f}% Chat")

sample_chat = [s for s in combined_samples if s["type"] == "chat"][0]
supervised_count = sum(1 for l in sample_chat["labels"] if l != -100)
masked_count = sum(1 for l in sample_chat["labels"] if l == -100)
print(f"Sample Chat Turn -> Supervised Tokens: {supervised_count}, Masked Tokens: {masked_count}")
assert supervised_count > 0 and masked_count > 0, "Prompt masking verification failed!"
print("Prompt Masking Verification: 100% SUCCESS!")
print("=" * 70)
