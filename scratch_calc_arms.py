from src.model.transformer import ModelConfig, TransformerDecoder

# Notice: with tie_embeddings=True, lm_head shares weights with tok_embeddings!
# Baseline: V=100262, dim=768, L=12 -> 77.0M embed + (12 * 6.29M) = 152.5M params.
# If V=16384 (12.6M embed):
# To reach ~152M with tie_embeddings=True:
# Trunk params need to be: 152.5M - 12.6M = 139.9M params!

arms = [
    # Arm 1: Deep Reasoning (L=22, dim=768, 12 heads, 4 kv)
    ("Arm 1: Deep Reasoning (L=22, d=768)", ModelConfig(vocab_size=16384, dim=768, n_layers=22, n_heads=12, n_kv_heads=4)),
    # Arm 2: Balanced Sweet-Spot (L=16, dim=896, 14 heads, 2 kv)
    ("Arm 2: Balanced (L=16, d=896)", ModelConfig(vocab_size=16384, dim=896, n_layers=16, n_heads=14, n_kv_heads=2)),
    # Arm 3: Wide Memory (L=12, dim=1024, 16 heads, 4 kv)
    ("Arm 3: Wide Memory (L=12, d=1024)", ModelConfig(vocab_size=16384, dim=1024, n_layers=12, n_heads=16, n_kv_heads=4)),
    # Baseline
    ("Baseline 100k (L=12, d=768)", ModelConfig(vocab_size=100262, dim=768, n_layers=12, n_heads=12, n_kv_heads=4)),
]

print(f"{'Configuration':<38} | {'Vocab':<7} | {'L':<3} | {'Dim':<5} | {'Heads/KV':<9} | {'Params':<8}")
print("-" * 75)
for name, cfg in arms:
    m = TransformerDecoder(cfg)
    total_p = sum(p.numel() for p in m.parameters())
    print(f"{name:<38} | {cfg.vocab_size:<7} | {cfg.n_layers:<3} | {cfg.dim:<5} | {cfg.n_heads}/{cfg.n_kv_heads:<6} | {total_p / 1e6:.2f} M")

