"""
Byte-Level BPE Tokenizer Wrapper
Supports fast encoding/decoding using tiktoken (GPT-4 / cl100k_base pattern)
with custom special tokens for code pretraining.
"""

import os
from typing import List, Union
from pathlib import Path


class CodeTokenizer:
    """
    High-performance byte-pair encoding tokenizer for code.
    Supports either standard tiktoken cl100k_base or custom HuggingFace BPE JSON models (e.g. mirendil_16k.json).
    """
    
    SPECIAL_TOKENS = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|pad|>": 100261,
    }

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = str(encoding_name)
        self.is_custom_json = self.encoding_name.endswith(".json") or Path(self.encoding_name).exists()
        
        if self.is_custom_json:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(self.encoding_name)
            self.vocab_size = self._tok.get_vocab_size()
            eot = self._tok.token_to_id("<|endoftext|>")
            pad = self._tok.token_to_id("<|pad|>")
            self.eot_token = eot if eot is not None else 0
            self.pad_token = pad if pad is not None else (self.eot_token + 1)
        else:
            import tiktoken
            self._enc = tiktoken.get_encoding(self.encoding_name)
            self.eot_token = self.SPECIAL_TOKENS["<|endoftext|>"]
            self.pad_token = self.SPECIAL_TOKENS["<|pad|>"]
            self.vocab_size = self._enc.n_vocab + len(self.SPECIAL_TOKENS)

    def encode(self, text: str, allowed_special: Union[str, set] = "all") -> List[int]:
        """Encodes string to a list of token IDs."""
        if self.is_custom_json:
            return self._tok.encode(text).ids
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        """Decodes token IDs back to human-readable string."""
        if self.is_custom_json:
            return self._tok.decode(tokens)
        return self._enc.decode(tokens)

    @property
    def bos_id(self) -> int:
        return self.eot_token

    @property
    def eos_id(self) -> int:
        return self.eot_token


if __name__ == "__main__":
    tokenizer = CodeTokenizer()
    sample_code = 'def calculate_fibonacci(n: int) -> int:\n    """Compute n-th fibonacci."""\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)\n'
    
    tokens = tokenizer.encode(sample_code)
    print(f"Original length in chars: {len(sample_code)}")
    print(f"Encoded token count: {len(tokens)}")
    print(f"Sample tokens: {tokens[:15]}...")
    decoded = tokenizer.decode(tokens)
    assert decoded == sample_code, "Round-trip decode failed!"
    print("Round-trip encoding test passed successfully!")
