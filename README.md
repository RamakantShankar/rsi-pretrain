# RSI Pretrain: Autonomous Foundation Model Pretraining from Scratch

> *"Pretraining at scale without a pretraining team."* — Inspired by Behnam Neyshabur's Mirendil-Base research.

This repository contains a complete, production-grade foundation model pretraining stack and educational laboratory built from ground zero in pure PyTorch. It demonstrates how a single researcher working alongside autonomous AI agents can design, train, ablate, and monitor foundation coding models.

---

## 📚 Interactive Curriculum & Visual Notebooks

Every topic is taught through our strict **5-Step Pedagogy**:
1. **Intuition & Metaphor First** (Real-world analogies)
2. **Presentation-Grade Vector Diagrams** (Clean PNG graphics, zero ASCII art)
3. **Pencil & Paper Exercises** (Hand-calculating with small vectors of size 2 or 4)
4. **Single Unified Notebooks** (100% clean HTML/Unicode, zero KaTeX parsing errors)
5. **Runnable PyTorch Code Verification** (Immediate tensor verification)

| Notebook | Topic | Core Concept |
| :--- | :--- | :--- |
| **[`00_paper_pencil_rmsnorm.ipynb`](notebooks/00_paper_pencil_rmsnorm.ipynb)** | **RMSNorm** | Normalization without mean-centering; 10–15% faster GPU memory bandwidth. Scale-invariance verified by hand. |
| **[`01_rope_intuition.ipynb`](notebooks/01_rope_intuition.ipynb)** | **Rotary Position (RoPE)** | 2D clock face rotations. Explains why odd dimensions (like 3) cannot work; hand-calculates 2D & 4D coordinate rotations. |
| **[`02_swiglu_explained.ipynb`](notebooks/02_swiglu_explained.ipynb)** | **SwiGLU Gating** | The Consultant (`w_up`), Fact-Checker (`w_gate`), and Summarizer (`w_down`). Context via self-attention; dynamic gating with smooth $\text{SiLU}$. |
| **[`03_gqa_explained.ipynb`](notebooks/03_gqa_explained.ipynb)** | **Grouped-Query Attention** | One reference book, multiple distinct questions. Explains `repeat_interleave` and proves stride-0 zero-copy memory pointers saving 66.7% to 87.5% VRAM! |
| **[`04_loss_and_perplexity.ipynb`](notebooks/04_loss_and_perplexity.ipynb)** | **Loss & Perplexity** | The 100-sided die metaphor. Proves why step-0 loss must equal $\ln(V)$ ($11.51$) and why 1.0 is the theoretical minimum. |
| **[`05_adamw_and_weight_decay.ipynb`](notebooks/05_adamw_and_weight_decay.ipynb)** | **AdamW & Splitting** | Zipf's Law in language. How $m_t / \sqrt{v_t}$ auto-scales gradients of 100.0 and 0.001. Why weight decay must NEVER touch RMSNorm. |
| **[`06_learning_rate_schedules.ipynb`](notebooks/06_learning_rate_schedules.ipynb)** | **Cosine vs. WSD** | The Spacecraft Landing. Why warm restarts cause catastrophic loss spikes, and why WSD allows indefinite training + a single clean landing. |
| **[`07_gradient_clipping_and_stability.ipynb`](notebooks/07_gradient_clipping_and_stability.ipynb)** | **Gradient Clipping** | The electrical circuit breaker. Hand-calculated $L_2$ norm scaling preserving direction and stopping `NaN` run explosions. |
| **[`08_scaling_laws_and_ablations.ipynb`](notebooks/08_scaling_laws_and_ablations.ipynb)** | **Chinchilla Scaling Laws** | The Gourmet Chef & Ingredients. Solves compute-optimal $C = 120 N^2$ by hand ($D \approx 20 \times N$) before writing code. |
| **[`09_the_rsi_agent_loop.ipynb`](notebooks/09_the_rsi_agent_loop.ipynb)** | **The Agentic Loop (RSI)** | Flight Director (human) + Autonomous Co-Pilot (agent). Automated triage, self-healing checkpoints, and experiment ledger management. |

---

## 🛠️ Architecture & Codebase Layout

```text
rsi-pretrain/
├── config/
│   ├── micro_local.yaml       # Fast CPU sanity-check configuration (~20M params)
│   └── dgx_base.yaml          # NVIDIA DGX Grace Blackwell (GB10) production config
├── notebooks/                 # 10 Interactive visual tutorial notebooks
│   └── assets/                # High-resolution presentation diagram PNGs
├── src/
│   ├── tokenizer/             # Byte-level BPE tokenizer (cl100k_base / FIM tokens)
│   ├── data/                  # Zero-padding packed sequence sharder (np.memmap uint32)
│   ├── model/                 # Pure PyTorch: RMSNorm, RoPE, SwiGLU, GQA, Decoder
│   └── engine/                # AdamW optimizer (decoupled decay), WSD scheduler, Trainer
├── train.py                   # Pretraining entrypoint (CPU, single-GPU, torchrun DDP)
├── generate.py                # Autoregressive generation with Temperature & Top-p
├── run_ablation.py            # Automated ablation engine & experiment ledger
└── inspect_shards.py          # Low-level binary shard & hex byte inspector
```

---

## 🚀 Quick Start

### 1. Installation
Using [`uv`](https://github.com/astral-sh/uv) (fast Python package manager):
```bash
git clone https://github.com/RamakantShankar/rsi-pretrain.git
cd rsi-pretrain
uv sync
```

### 2. Inspecting Binary Shards
Inspect raw disk bytes, token packing, and `<|endoftext|>` boundaries:
```bash
uv run python inspect_shards.py
```

### 3. Running Pretraining Locally (CPU / Sanity Run)
```bash
uv run python train.py --config config/micro_local.yaml
```

### 4. Testing Autoregressive Code Generation
Generate Python completions from a saved checkpoint:
```bash
uv run python generate.py --prompt "def quicksort(arr):" --checkpoint checkpoints/micro_sanity_run/best_model.pt
```

### 5. Running Automated Ablations (RSI Loop)
Launch an autonomous comparative study (e.g., GQA vs. MHA) and display the leaderboard:
```bash
uv run python run_ablation.py
```

---

## ⚡ Scaling to NVIDIA DGX Grace Blackwell (GB10)

For full-scale pretraining on NVIDIA DGX Blackwell hardware:

```bash
# Multi-GPU Distributed Data Parallel (DDP) with BF16 Tensor Cores
torchrun --nproc_per_node=8 train.py --config config/dgx_base.yaml
```

The model architecture automatically leverages:
* **FlashAttention-3 / CuDNN fused kernels** via PyTorch's native `F.scaled_dot_product_attention`.
* **Native Blackwell `bfloat16` and TensorFloat-32 (TF32)** for peak matrix-multiplication throughput.
* **Zero-copy stride-0 pointers** for Grouped-Query Attention (GQA), saving 66.7% of physical VRAM.
