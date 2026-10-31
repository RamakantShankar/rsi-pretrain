import torch
import torch.nn as nn
from typing import Tuple


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cosine and sine frequency matrices for Rotary Position Embedding (RoPE).
    
    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum context window
        theta: Base frequency (standard 10,000; or 500,000 for long context)
    """
    assert dim % 2 == 0, f"Head dimension must be even, got {dim}"
    
    # Calculate angular frequencies: theta_i = 1 / (theta ^ (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Time indices m = [0, 1, ..., max_seq_len - 1]
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: [max_seq_len, dim // 2]
    freqs = torch.outer(t, freqs)
    
    # Precompute cos and sin: repeat each element twice to match (dim)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [max_seq_len, dim]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # [max_seq_len, dim]
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims: [-x2, x1, -x4, x3, ...]"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to Query and Key tensors.
    
    Args:
        xq: [Batch, Heads_q, Seq_len, Head_dim]
        xk: [Batch, Heads_k, Seq_len, Head_dim]
        cos: [Seq_len, Head_dim]
        sin: [Seq_len, Head_dim]
    """
    seq_len = xq.shape[2]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(1)  # [1, 1, Seq_len, Head_dim]
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(1)  # [1, 1, Seq_len, Head_dim]

    # RoPE formula: x_rot = (x * cos) + (rotate_half(x) * sin)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)
