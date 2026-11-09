import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.norm import RMSNorm
from src.model.rope import apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with native FlashAttention / SDPA support and QK-Norm.
    
    If n_kv_heads == n_heads: Multi-Head Attention (MHA)
    If n_kv_heads == 1: Multi-Query Attention (MQA)
    If 1 < n_kv_heads < n_heads: Grouped-Query Attention (GQA)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.head_dim = dim // n_heads
        self.num_rep = self.n_heads // self.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        # Linear projections (no bias, as standard in modern LLMs)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # QK-Norm: per-head RMSNorm prevents attention entropy collapse and loss spikes
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 2. Reshape into [Batch, Heads, Seq_len, Head_dim]
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm (normalizing along head_dim before RoPE)
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # 3. Apply Rotary Position Embeddings to Q and K
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # 4. Repeat KV heads to match Q heads if using GQA (num_rep > 1)
        if self.num_rep > 1:
            xk = xk.repeat_interleave(self.num_rep, dim=1)
            xv = xv.repeat_interleave(self.num_rep, dim=1)

        # 5. Scaled Dot-Product Attention with causal masking
        # PyTorch automatically selects FlashAttention on GPU or SDPA on CPU
        dropout_p = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True
        )

        # 6. Reshape back: [Batch, Heads, Seq_len, Head_dim] -> [Batch, Seq_len, Dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        return self.out_proj(output)
