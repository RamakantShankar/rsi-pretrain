"""
Autoregressive Code Generation & Sampling with Temperature and Top-p.
"""

import argparse
import torch
import torch.nn.functional as F

from src.model.transformer import TransformerDecoder, ModelConfig
from src.tokenizer.bpe import CodeTokenizer


def top_p_sampling(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.95) -> torch.Tensor:
    """Samples next token from logits using Temperature and Nucleus (Top-p) filtering."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens with cumulative probability above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices right to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


from typing import List

@torch.no_grad()
def generate(
    model: TransformerDecoder,
    tokenizer: CodeTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cpu",
) -> str:
    return generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_samples=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )[0]


@torch.no_grad()
def generate_batch(
    model: TransformerDecoder,
    tokenizer: CodeTokenizer,
    prompt: str,
    num_samples: int = 1,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "cpu",
) -> List[str]:
    """Generates num_samples candidate completions in a single batched forward pass."""
    model.eval()
    model.to(device)

    token_ids = tokenizer.encode(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        next_token = top_p_sampling(next_token_logits, temperature=temperature, top_p=top_p)
        x = torch.cat((x, next_token), dim=1)

    completions = []
    for row in x.tolist():
        generated = row[len(token_ids):]
        if tokenizer.eos_id in generated:
            eos_idx = generated.index(tokenizer.eos_id)
            generated = generated[:eos_idx]
        completions.append(tokenizer.decode(token_ids + generated))
    return completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/micro_sanity_run/best_model.pt")
    parser.add_argument("--prompt", type=str, default="def quicksort(arr):")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint} onto {device}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = TransformerDecoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = CodeTokenizer()

    print(f"\n--- Prompt ---\n{args.prompt}")
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        device=device,
    )
    print(f"\n--- Generated Code ---\n{output}\n")


if __name__ == "__main__":
    main()
