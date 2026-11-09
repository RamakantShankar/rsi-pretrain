import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.norm import RMSNorm
from src.model.rope import precompute_freqs_cis
from src.model.attention import GroupedQueryAttention
from src.model.mlp import SwiGLUFFN


@dataclass
class ModelConfig:
    vocab_size: int = 16384  # Default to specialized Mirendil Python+English vocab
    dim: int = 896          # Default to Arm 2 Champion width
    n_layers: int = 16      # Default to Arm 2 Champion depth
    n_heads: int = 14       # 14 query heads (head_dim = 64)
    n_kv_heads: Optional[int] = 2  # GQA: 14 Q heads, 2 KV heads (7:1 ratio)
    hidden_dim: Optional[int] = None
    max_seq_len: int = 2048
    rope_theta: float = 100000.0   # Scaled RoPE base for long-context code (100k)
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0
    use_qk_norm: bool = True       # QK-Norm enabled for mathematical stability


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            use_qk_norm=config.use_qk_norm,
            norm_eps=config.norm_eps,
        )
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(dim=config.dim, hidden_dim=config.hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-LN Transformer Architecture: residual + sublayer(norm(x))
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerDecoder(nn.Module):
    """
    Modern Foundation Model Transformer Decoder.
    Features:
      - Pre-RMSNorm
      - RoPE (Rotary Position Embeddings)
      - SwiGLU FFN
      - Grouped-Query Attention (GQA) with native FlashAttention/SDPA
      - Optional Weight Tying
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with output projection
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies (registered as non-persistent buffers)
        head_dim = config.dim // config.n_heads
        cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights using standard scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for next-token prediction.
        
        Args:
            idx: Tensor of token IDs [batch_size, seq_len]
            targets: Optional ground-truth next token IDs [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.max_seq_len, (
            f"Input sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # 1. Embed input tokens
        x = self.tok_embeddings(idx)

        # 2. Slice precomputed RoPE frequencies for current sequence length
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        # 3. Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x, cos, sin)

        # 4. Final normalization
        x = self.norm(x)

        # 5. Output projection to vocab logits
        logits = self.lm_head(x)

        # 6. Compute Cross-Entropy Loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Counts total trainable parameters (optionally excluding embeddings)."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and self.config.tie_embeddings:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params


if __name__ == "__main__":
    # Test instantiation with a micro config for sanity check
    micro_cfg = ModelConfig(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        tie_embeddings=True,
    )
    model = TransformerDecoder(micro_cfg)
    total_params = model.get_num_params(non_embedding=False)
    print(f"Model created successfully! Total parameters: {total_params:,}")

    # Forward pass test
    dummy_x = torch.randint(0, 1000, (2, 32))
    dummy_y = torch.randint(0, 1000, (2, 32))
    logits, loss = model(dummy_x, dummy_y)

    print(f"Input shape:  {dummy_x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Cross-entropy loss: {loss.item():.4f}")
    assert logits.shape == (2, 32, 1000), "Logits shape mismatch!"
    assert loss is not None and not torch.isnan(loss), "Loss computation failed!"
    print("Transformer forward pass and loss test PASSED!")
