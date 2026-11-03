import torch

print("=" * 60)
print(f"PyTorch Version:         {torch.__version__}")
print(f"CUDA Available:          {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:             {torch.cuda.get_device_name(0)}")
    print(f"Device Count:            {torch.cuda.device_count()}")
    print(f"BF16 Supported:          {torch.cuda.is_bf16_supported()}")
    print(f"Compute Capability:      {torch.cuda.get_device_capability(0)}")
    print(f"Current Device Index:    {torch.cuda.current_device()}")
    
    # Test tensor allocation on Blackwell GB10
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    z = torch.matmul(x, y)
    print(f"Blackwell BF16 Matmul:   Verified! Shape = {list(z.shape)}")
print("=" * 60)
