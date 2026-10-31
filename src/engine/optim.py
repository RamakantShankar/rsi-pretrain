import math
import torch
import torch.nn as nn
from typing import Tuple


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 3e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Configures AdamW optimizer with selective weight decay.
    Weight decay is applied ONLY to 2D weights (matrices).
    1D parameters (biases, RMSNorm scales, embeddings) have weight decay set to 0.0.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Weight decay only for matrix weights with 2 or more dimensions
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer Parameter Groups: {len(decay_params)} decay ({num_decay:,} params), "
          f"{len(no_decay_params)} no-decay ({num_nodecay:,} params)")

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),  # Fast CUDA fused kernel if on GPU
    )
    return optimizer


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "wsd",
    warmup_steps: int = 100,
    total_steps: int = 1000,
    min_lr_ratio: float = 0.1,
    stable_ratio: float = 0.7,
):
    """
    Learning Rate Scheduler supporting both 'cosine' and 'wsd' (Warmup-Stable-Decay).
    """

    def lr_lambda(current_step: int) -> float:
        # 1. Linear Warmup Phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        if scheduler_type == "cosine":
            # Cosine decay from 1.0 down to min_lr_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        elif scheduler_type == "wsd":
            # Warmup -> Stable -> Linear/Cosine Decay
            stable_steps = int(total_steps * stable_ratio)
            decay_steps = total_steps - stable_steps
            if current_step < stable_steps:
                return 1.0  # Constant high LR during stable representation learning
            else:
                progress = float(current_step - stable_steps) / float(max(1, decay_steps))
                progress = min(max(progress, 0.0), 1.0)
                # Cosine decay in the final phase
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
