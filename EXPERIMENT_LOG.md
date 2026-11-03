# 🚀 Mirendil-Base Reproduction: Pretraining & Ablation Ledger

> **Research Mission:** Pretraining foundation coding language models at scale with 1 researcher + AI agents via Recursive Self-Improvement (RSI), replicating Behnam Neyshabur's Mirendil-Base methodology on NVIDIA Grace Blackwell hardware.

---

## 🖥️ Hardware & Infrastructure Environment

| Property | Value |
| :--- | :--- |
| **Cluster Node** | NVIDIA DGX Spark (`spark-320a.local` / `10.0.0.133`) |
| **GPU Architecture** | NVIDIA GB10 (Grace Blackwell, Compute Capability 12.1) |
| **Host Processor** | 20-Core ARM Cortex-X925 / A725 with 121 GB unified RAM |
| **Precision** | Native Tensor Core `bfloat16` |
| **Live Telemetry** | [Weights & Biases Dashboard](https://wandb.ai/ramakant_shankar/rsi-coding-pretrain) + Local `metrics.jsonl` |
| **Git Repository** | [GitHub: rsi-pretrain](https://github.com/RamakantShankar/rsi-pretrain) |

---

## 📈 Visual Architecture & Learning Curves

* **Live Dashboard URL:** [https://wandb.ai/ramakant_shankar/rsi-coding-pretrain](https://wandb.ai/ramakant_shankar/rsi-coding-pretrain)
* **WSD vs. Cosine Schedule:** [lr_schedules.png](file:///M:/Rama/Study/llm/LLM/RSI/notebooks/assets/lr_schedules.png)
* **Catastrophic Restart Spikes:** [lr_restart_spike.png](file:///M:/Rama/Study/llm/LLM/RSI/notebooks/assets/lr_restart_spike.png)
* **AdamW Parameter Splitting:** [optimizer_split.png](file:///M:/Rama/Study/llm/LLM/RSI/notebooks/assets/optimizer_split.png)
* **Gradient Clipping Norm:** [grad_clipping.png](file:///M:/Rama/Study/llm/LLM/RSI/notebooks/assets/grad_clipping.png)
* **Chinchilla Scaling Laws:** [scaling_laws.png](file:///M:/Rama/Study/llm/LLM/RSI/notebooks/assets/scaling_laws.png)
* **RSI Agent State Machine:** [rsi_agent_loop.png](file:///M:/Rama/Study/llm/LLM/RSI/notebooks/assets/rsi_agent_loop.png)

---

## 📊 Master Experiment & Ablation Ledger

| Run ID | Architecture | Data & Tokens | Schedule | Min Train Loss | Min Val Loss | Val PPL | Status & Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`micro_local_smoke`** | 20.8M (d=256, L=6, H=4, KV=2) | Synthetic (50k) | WSD (50 steps) | 0.4814 | N/A | N/A | ✅ **Passed.** Baseline CPU smoke test verified convergence and gradient backprop. |
| **`ablation_gqa_vs_mha`** | 20.8M (KV=2 vs KV=6) | Synthetic (50k) | WSD (40 steps) | 0.4901 (GQA)<br>0.4907 (MHA) | N/A | N/A | ✅ **Passed.** Proved GQA matches full MHA loss within 0.0006 while cutting KV cache by 66%. |
| **`152m_wsd_10m_baseline`** | 152.5M (d=768, L=12, H=12, KV=4, Tied) | `codeparrot-clean` (10M) | WSD (500 steps) | *Pending* | *Pending* | *Pending* | ⏳ **Queued.** First Blackwell GB10 production run on real Python code. |

---

## 🧠 Architectural Decisions & Key Learnings

### 1. The "Vocabulary Tax" in Small LLMs
* **The Insight:** With a modern 100,262 token vocabulary (OpenAI `cl100k_base` + coding FIM tokens), a single embedding matrix of shape `[100262, 768]` consumes **77 Million parameters**.
* **The Danger of Untying:** Untying embeddings (`tie_embeddings: false`) adds another 77M for the output head, meaning embeddings alone (154M) would be double the entire transformer body (75.5M), ballooning the model to 230M!
* **The Solution:** Tying weights (`tie_embeddings: true`) shares the 77M matrix between input and output, keeping the model at a balanced **152.5 Million parameters** (standard practice in Gemma-2B, TinyLlama, and GPT-2).

### 2. Sequence Packing & Zero-Copy Binary Shards
* **The Problem:** Naive padding `<pad>` wastes 90%+ of GPU compute on zeros. Deserializing JSON/Parquet on the fly starves the GPU.
* **The Solution:** We pack Python files continuously with `<|endoftext|>` boundaries into raw binary `uint32` files (4 bytes/token). At training time, `np.memmap` maps the files directly into Linux virtual memory with zero RAM duplication. 10M tokens packed in 5 seconds on DGX (~2M tok/s).

### 3. Warmup-Stable-Decay (WSD) vs. Classic Cosine Decay
* **The Problem:** Cosine decay requires fixing the token budget in advance. Extending a cosine run triggers massive loss spikes due to optimizer state $v_t$ contamination and narrow-valley ejection.
* **The Solution:** WSD maintains a stable learning rate plateau indefinitely, allowing continuous training. Cooling decay is triggered only when the compute budget is reached.

### 4. Parameter-Split AdamW Optimizer
* **Rule:** 2D weight matrices (Attention $W_q, W_k, W_v, W_o$, MLP $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$, Embeddings) receive Weight Decay = **0.1**.
* **Rule:** 1D vectors (RMSNorm scale gains $\gamma$) receive Weight Decay = **0.0** (decaying scale gains shrinks layer activations and causes gradient vanishing).
* **$\beta_2$ Setting:** Set to **0.95** (modern LLM standard) instead of PyTorch default 0.999 to forget historical gradient variance faster and avoid loss spikes.

---

*This document is automatically updated after every training run and ablation trial.*
