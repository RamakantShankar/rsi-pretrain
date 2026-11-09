import torch
from src.model.transformer import ModelConfig, TransformerDecoder

cfg = ModelConfig(dim=896, n_layers=16, n_heads=14, n_kv_heads=2, max_seq_len=1024)
m = TransformerDecoder(cfg).cuda()
x = torch.randint(0, 16384, (16, 1024), device="cuda")
y = torch.randint(0, 16384, (16, 1024), device="cuda")
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    out, loss = m(x, targets=y)
loss.backward()
opt.step()
print(f"1024 seq_len success! Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
