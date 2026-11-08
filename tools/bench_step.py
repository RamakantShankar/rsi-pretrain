import time
import torch
from src.model.transformer import ModelConfig, TransformerDecoder

cfg = ModelConfig(vocab_size=16384, dim=768, n_layers=22, n_heads=12, n_kv_heads=4, max_seq_len=512)
model = TransformerDecoder(cfg).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 16384, (32, 512), device="cuda")
y = torch.randint(0, 16384, (32, 512), device="cuda")

# Warmup
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    _, loss = model(x, targets=y)
loss.backward()
opt.step()
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    opt.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, targets=y)
    loss.backward()
    opt.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"10 steps took: {elapsed:.3f}s ({elapsed/10*1000:.1f} ms/step)")
