import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUFFN(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network.
    
    Formula:
        FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 256):
        super().__init__()
        if hidden_dim is None:
            # 8/3 scaling rule from LLaMA / PaLM
            hidden_dim = int(2 * (4 * dim) / 3)
            # Round up to nearest multiple_of (e.g. 256) for GPU tensor core efficiency
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w_gate(x)) * w_up(x)
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
