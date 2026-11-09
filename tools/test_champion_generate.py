import json
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from src.model.transformer import ModelConfig, TransformerDecoder
from generate import top_p_sampling

def run_champion_inference(prompt: str, max_tokens: int = 100, temp: float = 0.6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file("checkpoints/tokenizers/mirendil_16k.json")
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
    
    ckpt_path = "checkpoints/champion/mirendil_154m_champion.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    enc = tok.encode(prompt)
    x = torch.tensor([enc.ids], dtype=torch.long, device=device)

    t0 = time.time()
    generated = []
    with torch.no_grad():
        for i in range(max_tokens):
            logits, _ = model(x)
            next_logit = logits[:, -1, :]
            # Don't let model emit EOT immediately if we want code
            if i < 10 and eot_id is not None:
                next_logit[:, eot_id] = -float("inf")
            next_tok = top_p_sampling(next_logit, temperature=temp, top_p=0.95)
            tok_id = next_tok.item()
            if tok_id == eot_id and i >= 10:
                break
            generated.append(tok_id)
            x = torch.cat([x, next_tok], dim=1)

    elapsed = time.time() - t0
    gen_text = tok.decode(generated)
    speed = len(generated) / max(0.001, elapsed)
    return gen_text, speed, len(generated)

if __name__ == "__main__":
    tok = Tokenizer.from_file("checkpoints/tokenizers/mirendil_16k.json")
    print(f"Vocab size: {tok.get_vocab_size()}")
    print(f"EOT ID: {tok.token_to_id('<|endoftext|>')}")
    print(f"Token 0: {repr(tok.id_to_token(0))}")
    print(f"Token 1: {repr(tok.id_to_token(1))}")
    
    prompts = [
        "def binary_search(arr, target):\n    \"\"\"Binary search on sorted array.\"\"\"\n",
        "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "def reverse_string(s: str) -> str:\n    \"\"\"Reverse the input string.\"\"\"\n",
    ]
    print(f"\n{'='*70}\nMIRENDIL-154M CHAMPION LIVE INFERENCE EVALUATION\n{'='*70}\n")
    for p in prompts:
        print(f">>> PROMPT:\n{p}")
        out, spd, n_tok = run_champion_inference(p, max_tokens=64, temp=0.6)
        print(f">>> GENERATION ({n_tok} tokens @ {spd:.1f} tok/s):\n{out}\n")
        print("-" * 70)
