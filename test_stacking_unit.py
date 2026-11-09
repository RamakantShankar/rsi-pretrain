import torch
from tools.ablate_stacking import stack_depth_8_to_16
from src.model.transformer import ModelConfig, TransformerDecoder

cfg8 = ModelConfig(n_layers=8, dim=256, n_heads=4, n_kv_heads=2, max_seq_len=64)
cfg16 = ModelConfig(n_layers=16, dim=256, n_heads=4, n_kv_heads=2, max_seq_len=64)

m8 = TransformerDecoder(cfg8)
m16 = TransformerDecoder(cfg16)

stack_depth_8_to_16(m8, m16)

# Verify zero projections
for i in range(8):
    twin = m16.layers[2 * i + 1]
    assert torch.all(twin.attn.out_proj.weight == 0.0), f"Attn out_proj not zero at layer {2*i+1}"
    assert torch.all(twin.ffn.w_down.weight == 0.0), f"FFN w_down not zero at layer {2*i+1}"

# Test forward pass equivalence
x = torch.randint(0, 1000, (2, 32))
y = torch.randint(0, 1000, (2, 32))
m8.eval()
m16.eval()
with torch.no_grad():
    _, loss8 = m8(x, targets=y)
    _, loss16 = m16(x, targets=y)

print(f"Loss 8L: {loss8.item():.6f} | Loss 16L: {loss16.item():.6f}")
delta = abs(loss8.item() - loss16.item())
print(f"Delta: {delta:.8f}")
assert delta < 1e-4, f"Delta {delta} exceeds tolerance!"
print("--> ALL STACKING UNIT TESTS PASSED PERFECTLY!")
