"""
Looped / Recurrent Transformer Architecture for RSI.
Enables virtual depth expansion via parameter-sharing recurrence loops.
Allows adaptive test-time compute: an 8-layer physical model can execute T=1, T=2, or T=3 loops
to achieve 8, 16, or 24 virtual layers with zero extra parameter memory.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.norm import RMSNorm
from src.model.rope import precompute_freqs_cis
from src.model.transformer import ModelConfig, TransformerBlock


class LoopedTransformerDecoder(nn.Module):
    """
    Looped (Recurrent) Transformer Decoder.
    
    Key Mechanism:
      Allocates n_layers physical blocks (e.g. 8 layers).
      In each forward pass, input activations loop through the same n_layers blocks
      for num_loops iterations (T = 1, 2, 3...).
      
    Features:
      - Inter-Loop RMSNorm: Prevents activation variance explosion across recurrence passes.
      - Dynamic Test-Time Compute: Inference can vary num_loops per token/prompt.
      - RoPE Invariant: Preserves spatial token positions across recurrence cycles.
    """

    def __init__(self, config: ModelConfig, default_loops: int = 2):
        super().__init__()
        self.config = config
        self.default_loops = default_loops

        # 1. Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # 2. Physical Transformer Blocks (e.g. 8 physical layers)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # 3. Inter-Loop Recurrent Normalization
        # Prevents activation variance drift between Pass 1 and Pass 2
        self.loop_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # 4. Final Output Normalization and LM Head
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share token embeddings with LM head
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        num_loops: Optional[int] = None,
        depth_weights: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with recurrent looping through physical layers.
        
        Args:
            idx: [batch_size, seq_len] Token IDs
            targets: Optional [batch_size, seq_len] Target token IDs
            num_loops: Optional override for recurrence loop count (default: self.default_loops)
            depth_weights: Optional list of loss weights for Multi-Depth Deep Supervision (e.g. [0.2, 0.4, 0.25, 0.15])
        """
        if depth_weights is not None:
            loops = len(depth_weights)
        else:
            loops = num_loops if num_loops is not None else self.default_loops

        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.max_seq_len, f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"

        # 1. Embed input tokens
        x = self.tok_embeddings(idx)

        # 2. Slice RoPE positional frequencies (spatial positions remain identical across loops)
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        # 3. Recurrent Execution Loop
        total_loss = 0.0 if (targets is not None and depth_weights is not None) else None

        for loop_idx in range(loops):
            # Pass through all physical layers
            for layer in self.layers:
                x = layer(x, cos, sin)

            # Deep Multi-Exit Supervision: evaluate loss at this loop exit
            if targets is not None and depth_weights is not None:
                normed_x = self.norm(x)
                logits_loop = self.lm_head(normed_x)
                loss_loop = F.cross_entropy(
                    logits_loop.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + depth_weights[loop_idx] * loss_loop

            # Apply Inter-Loop Normalization between passes (except after final pass)
            if loop_idx < loops - 1:
                x = self.loop_norm(x)

        # 4. Final Output Normalization
        x = self.norm(x)

        # 5. Project to Vocabulary Logits
        logits = self.lm_head(x)

        # 6. Standard Single-Exit Loss if targets provided and depth_weights was not used
        if targets is not None and depth_weights is None:
            total_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, total_loss
