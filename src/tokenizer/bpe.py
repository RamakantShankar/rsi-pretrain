"""
Byte-Level BPE Tokenizer Wrapper
Supports fast encoding/decoding using tiktoken (GPT-4 / cl100k_base pattern)
with custom special tokens for code pretraining.
"""

from typing import List, Union
import tiktoken


class CodeTokenizer:
    """
    High-performance byte-pair encoding tokenizer for code.
    Defaults to the cl100k_base regex pattern used in GPT-4, which
    splits on whitespace, contractions, punctuation, and multi-space runs cleanly.
    """
    
    SPECIAL_TOKENS = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|pad|>": 100261,
    }

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        
        # Total vocabulary size including special tokens
        self.eot_token = self.SPECIAL_TOKENS["<|endoftext|>"]
        self.pad_token = self.SPECIAL_TOKENS["<|pad|>"]
        self.vocab_size = self._enc.n_vocab + len(self.SPECIAL_TOKENS)

    def encode(self, text: str, allowed_special: Union[str, set] = "all") -> List[int]:
        """Encodes string to a list of token IDs."""
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        """Decodes token IDs back to human-readable string."""
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
