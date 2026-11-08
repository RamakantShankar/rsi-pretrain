import torch

d = torch.load("checkpoints/sweep/arm1_deep.pt", map_location="cpu", weights_only=False)
cfg = d["config"]
print(f"Arm 1 ({cfg.n_layers}L, {cfg.dim}d): Val Loss = {d['val_loss']:.4f} | Val BPB = {d['val_bpb']:.3f} | Step = {d['step']}")
