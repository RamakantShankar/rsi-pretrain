import torch
import os
from pathlib import Path

src_path = Path("/home/ramakantshankar/rsi-pretrain/checkpoints/v3/152m_rsi_m156/best_model.pt")
dst_path = Path("/home/ramakantshankar/rsi-pretrain/checkpoints/152m_champion_m156.pt")

if not src_path.exists():
    # Fallback to local
    src_path = Path("checkpoints/152m_wsd_10m_baseline/best_model.pt")
    dst_path = Path("checkpoints/152m_champion_m156.pt")

print(f"Loading {src_path}...")
ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
print("Keys in checkpoint:", list(ckpt.keys()))

export_dict = {
    "model_state_dict": ckpt["model_state_dict"],
    "config": ckpt.get("config"),
    "step": ckpt.get("step", 0),
}

dst_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving stripped model weights to {dst_path}...")
torch.save(export_dict, dst_path)
size_mb = os.path.getsize(dst_path) / 1024 / 1024
print(f"Successfully saved {dst_path} ({size_mb:.2f} MB)")
