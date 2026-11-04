import os
import time
import json
import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import TransformerDecoder
from src.engine.optim import configure_optimizers, get_lr_scheduler


class Trainer:
    def __init__(
        self,
        model: TransformerDecoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        scheduler_type: str = "wsd",
        warmup_steps: int = 50,
        total_steps: int = 500,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        log_interval: int = 10,
        eval_interval: int = 100,
        save_dir: str = "checkpoints",
        experiment_name: str = "run_01",
        use_wandb: bool = False,
        wandb_project: str = "rsi-coding-foundation",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = os.path.join(self.save_dir, "metrics.jsonl")

        # Determine precision dtype
        is_cuda = "cuda" in str(self.device)
        self.device_type = "cuda" if is_cuda else "cpu"
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.autocast_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        else:
            self.autocast_dtype = torch.float32

        # Setup Optimizer and LR Scheduler
        self.optimizer = configure_optimizers(
            self.model,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.step = 0
        self.best_val_loss = float("inf")

        # Weights & Biases Dual-Logging
        self.use_wandb = use_wandb
        if self.use_wandb:
            try:
                import wandb
                model_cfg = getattr(self.model, "config", {})
                wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config={
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "scheduler_type": scheduler_type,
                        "warmup_steps": warmup_steps,
                        "total_steps": total_steps,
                        "grad_accum_steps": grad_accum_steps,
                        "dtype": dtype,
                        "model": model_cfg.__dict__ if hasattr(model_cfg, "__dict__") else str(model_cfg),
                    },
                )
                print(f"--> [Weights & Biases] Connected! Live Run Dashboard: {wandb.run.url}")
            except Exception as e:
                print(f"[WARNING] Weights & Biases connection failed: {e}. Falling back to local logging only.")
                self.use_wandb = False

    def _log_metric(self, metrics: Dict[str, Any]):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=metrics.get("step", self.step))
            except Exception:
                pass

    @torch.no_grad()
    def evaluate(self, num_batches: int = 20) -> float:
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        batches = 0

        for i, (x, y) in enumerate(self.val_loader):
            if i >= num_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype, enabled=(self.device_type == "cuda")):
                _, loss = self.model(x, y)
            total_loss += loss.item()
            batches += 1

        self.model.train()
        return total_loss / max(1, batches)

    def train(self):
        print(f"\n{'='*20} Starting Pretraining: {self.save_dir} {'='*20}")
        print(f"Device: {self.device} | Autocast: {self.autocast_dtype} | Total Steps: {self.total_steps}")
        print(f"Gradient Accumulation: {self.grad_accum_steps} | Batch Size per step: {self.train_loader.batch_size}")

        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        start_time = time.time()
        tokens_processed = 0

        while self.step < self.total_steps:
            self.optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation loop
            for micro_step in range(self.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)
                tokens_processed += x.numel()

                with torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype, enabled=(self.device_type == "cuda")):
                    _, loss = self.model(x, y)
                    loss = loss / self.grad_accum_steps

                # Check for NaN / Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[WARNING] NaN/Inf detected at step {self.step}! Skipping batch.")
                    continue

                loss.backward()
                accum_loss += loss.item()

            # Gradient clipping to prevent gradient explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            # Logging
            if self.step % self.log_interval == 0:
                elapsed = time.time() - start_time
                tok_per_sec = tokens_processed / max(1e-5, elapsed)
                current_lr = self.scheduler.get_last_lr()[0]
                avg_train_loss = accum_loss / self.log_interval
                
                train_ppl = math.exp(min(avg_train_loss, 20.0))
                log_data = {
                    "step": self.step,
                    "train_loss": round(avg_train_loss, 4),
                    "train_ppl": round(train_ppl, 2),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": float(f"{current_lr:.6e}"),
                    "tokens_per_sec": int(tok_per_sec),
                }
                print(
                    f"Step {self.step:5d}/{self.total_steps} | "
                    f"Loss: {log_data['train_loss']:.4f} | "
                    f"PPL: {log_data['train_ppl']:.2f} | "
                    f"GradNorm: {log_data['grad_norm']:.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {log_data['tokens_per_sec']:,} tok/s"
                )
                self._log_metric(log_data)
                accum_loss = 0.0

            # Evaluation and Checkpointing
            if self.step % self.eval_interval == 0 or self.step == self.total_steps:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20.0))
                print(f"--> [Evaluation] Step {self.step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                self._log_metric({
                    "step": self.step,
                    "val_loss": round(val_loss, 4),
                    "val_ppl": round(val_ppl, 2),
                })

                # Save best checkpoint
                if val_loss < self.best_val_loss or self.val_loader is None:
                    self.best_val_loss = val_loss
                    ckpt_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save({
                        "step": self.step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.model.config,
                        "val_loss": val_loss,
                    }, ckpt_path)
                    print(f"--> [Checkpoint] Saved model checkpoint to {ckpt_path}")

        print(f"\nTraining completed! All logs written to {self.log_file}")
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
