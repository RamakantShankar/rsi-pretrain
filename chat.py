#!/usr/bin/env python3
"""
Interactive Chat & Code Playground for the 152M RSI Champion Model (M_156).
Supports two interaction modes:
  1. Base Code Completion (e.g., 'def is_palindrome(s):')
  2. Conversational Coding Assistant (few-shot instruction wrapper)
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

from src.model.transformer import TransformerDecoder, ModelConfig
from src.tokenizer.bpe import CodeTokenizer


def top_p_sampling(logits: torch.Tensor, temperature: float = 0.6, top_p: float = 0.95) -> torch.Tensor:
    if temperature <= 0.01:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_response(
    model: TransformerDecoder,
    tokenizer: CodeTokenizer,
    prompt: str,
    max_new_tokens: int = 96,
    temperature: float = 0.4,
    top_p: float = 0.95,
    stop_tokens: list = None,
    device: str = "cpu",
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = []
    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        next_token = top_p_sampling(next_token_logits, temperature=temperature, top_p=top_p)
        token_val = next_token.item()

        if token_val == tokenizer.eos_id:
            break

        generated_ids.append(token_val)
        x = torch.cat((x, next_token), dim=1)

        # Check for stop tokens or stop words (e.g. newline followed by 'User:')
        current_text = tokenizer.decode(generated_ids)
        if stop_tokens:
            if any(stop_word in current_text for stop_word in stop_tokens):
                for stop_word in stop_tokens:
                    if stop_word in current_text:
                        current_text = current_text.split(stop_word)[0]
                return current_text

    return tokenizer.decode(generated_ids)


def load_champion_model(checkpoint_path: str = None, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        # Check standard paths
        candidates = [
            "checkpoints/152m_champion_m156.pt",
            "checkpoints/152m_wsd_10m_baseline/best_model.pt",
            "../checkpoints/152m_champion_m156.pt",
        ]
        for c in candidates:
            if Path(c).exists():
                checkpoint_path = c
                break

    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Champion checkpoint not found! Checked: {candidates}")

    print(f"Loading champion model from: {checkpoint_path} onto [{device}]...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config") or ModelConfig()

    model = TransformerDecoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = CodeTokenizer()
    print("Model and Tokenizer loaded successfully!\n")
    return model, tokenizer, device


# Few-shot Conversational Coding Wrapper
CHAT_PROMPT_HEADER = '''"""
Interactive Python Coding Assistant.
Answers user requests by writing clean, tested Python code.
"""

# User: Write a function to check if a number is positive.
# Assistant:
def is_positive(n):
    return n > 0

# User: Write a function to multiply two numbers.
# Assistant:
def multiply_two_numbers(a, b):
    return a * b

# User: {query}
# Assistant:
'''


def run_interactive_chat():
    print("=" * 65)
    print("  🤖 Mirendil-152M RSI Champion (Generation M_156) Chat Portal")
    print("  Trained on 42,738 verified Python programs & autonomous self-play")
    print("=" * 65)
    print("Modes:")
    print("  1. Chat Mode: Type natural language instructions (e.g. 'reverse a string')")
    print("  2. Code Mode: Type Python signatures (e.g. 'def is_palindrome(s):')")
    print("Type 'exit' or 'quit' to end session.\n")

    model, tokenizer, device = load_champion_model()

    mode = "chat"
    print(f"Active Mode: [CHAT MODE] (Type ':code' to switch to raw Python completion)")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if user_input == ":code":
                mode = "code"
                print("Switched to [RAW CODE MODE]. Enter Python signatures directly.")
                continue
            elif user_input == ":chat":
                mode = "chat"
                print("Switched to [CHAT MODE]. Enter conversational instructions.")
                continue

            print("\nModel > Thinking...", end="\r")

            if mode == "chat":
                prompt = CHAT_PROMPT_HEADER.format(query=user_input)
                response = generate_response(
                    model, tokenizer, prompt,
                    max_new_tokens=96,
                    temperature=0.3,
                    stop_tokens=["# User:", "<|endoftext|>"]
                )
            else:
                response = generate_response(
                    model, tokenizer, user_input,
                    max_new_tokens=80,
                    temperature=0.2,
                    stop_tokens=["<|endoftext|>"]
                )

            # Clean output
            clean_resp = response.strip()
            print(f"Model > \n{clean_resp}\n")

        except KeyboardInterrupt:
            print("\nSession ended.")
            break


if __name__ == "__main__":
    run_interactive_chat()
