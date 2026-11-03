"""
Generates crisp, high-resolution presentation diagrams as clean PNG files
for all tutorial notebooks.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("notebooks/assets", exist_ok=True)

# Set high-DPI styling
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

# -------------------------------------------------------------
# 1. LR Schedules Diagram (Cosine vs WSD)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor='#0d1117')
ax1.set_facecolor('#161b22')
ax2.set_facecolor('#161b22')

# Panel 1: Cosine
t = np.linspace(0, 1000, 1000)
warmup = 100
lr_max = 0.001
lr_cosine = np.zeros_like(t)
for i, step in enumerate(t):
    if step < warmup:
        lr_cosine[i] = lr_max * (step / warmup)
    else:
        progress = (step - warmup) / (1000 - warmup)
        lr_cosine[i] = 0.0001 + (lr_max - 0.0001) * 0.5 * (1 + np.cos(np.pi * progress))

ax1.plot(t, lr_cosine, color='#d29922', lw=2.5, label='Cosine Decay')
ax1.axvline(warmup, color='#8b949e', linestyle=':', alpha=0.5)
ax1.text(warmup / 2, 0.0004, 'Warmup', ha='center', color='#8b949e', fontsize=9, fontweight='bold')
ax1.set_title("1. Classic Cosine Decay (GPT-3 / LLaMA)", fontsize=11, fontweight='bold', color='#d29922', pad=10)
ax1.set_xlabel("Steps (T - Fixed Budget)", fontsize=9)
ax1.set_ylabel("Learning Rate", fontsize=9)
ax1.grid(True, color='#30363d', alpha=0.5)
ax1.text(500, 0.00085, "❌ Must fix total steps T before starting!", color='#f85149', fontsize=9, ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#f85149', alpha=0.9))

# Panel 2: WSD
lr_wsd = np.zeros_like(t)
stable = 800
for i, step in enumerate(t):
    if step < warmup:
        lr_wsd[i] = lr_max * (step / warmup)
    elif step < stable:
        lr_wsd[i] = lr_max
    else:
        progress = (step - stable) / (1000 - stable)
        lr_wsd[i] = 0.0001 + (lr_max - 0.0001) * 0.5 * (1 + np.cos(np.pi * progress))

ax2.plot(t, lr_wsd, color='#58a6ff', lw=3, label='WSD')
ax2.axvspan(warmup, stable, color='#58a6ff', alpha=0.08)
ax2.text((warmup + stable) / 2, 0.0005, 'STABLE PLATEAU\n(Train as long as you want!)',
         ha='center', va='center', color='#58a6ff', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22', edgecolor='#58a6ff'))

ax2.axvspan(stable, 1000, color='#3fb950', alpha=0.1)
ax2.text(900, 0.0003, 'Decay\nPhase', ha='center', color='#3fb950', fontsize=9, fontweight='bold')

ax2.set_title("2. Modern WSD (MiniCPM / Mirendil-Base)", fontsize=11, fontweight='bold', color='#58a6ff', pad=10)
ax2.set_xlabel("Steps (Flexible Budget)", fontsize=9)
ax2.grid(True, color='#30363d', alpha=0.5)
ax2.text(500, 0.00015, "✅ Land the model whenever you decide to stop!", color='#3fb950', fontsize=9, ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#3fb950', alpha=0.9))

plt.tight_layout()
plt.savefig("notebooks/assets/lr_schedules.png", dpi=180, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 2. Optimizer Spectrum & Parameter Split
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='#0d1117')
ax1.set_facecolor('#161b22')
ax2.set_facecolor('#161b22')

# Optimizer Spectrum Table
ax1.axis('off')
ax1.set_title("The 4 Core Optimizers", fontsize=11, fontweight='bold', color='#58a6ff', pad=12)

table_data = [
    ["1. SGD", "Blind Hiker", "Bounces wildly in ravines"],
    ["2. Momentum", "Bowling Ball", "Uses same speed for all params"],
    ["3. RMSprop", "Adaptive Step", "Lacks momentum smoothing"],
    ["4. AdamW", "All-Terrain (King)", "Momentum + Adaptive + Decoupled"]
]

col_labels = ["Optimizer", "Metaphor", "Behavior"]
table = ax1.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.0, 1.8)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#30363d')
    if r == 0:
        cell.set_facecolor('#1f2937')
        cell.set_text_props(color='#58a6ff', fontweight='bold')
    elif r == 4:
        cell.set_facecolor('#1c2738')
        cell.set_text_props(color='#3fb950', fontweight='bold')
    else:
        cell.set_facecolor('#161b22')
        cell.set_text_props(color='#c9d1d9')

# Parameter Split Flow
ax2.axis('off')
ax2.set_title("The Parameter Splitting Rule", fontsize=11, fontweight='bold', color='#3fb950', pad=12)

# Root node
ax2.text(0.5, 0.85, "All Model Parameters", ha='center', va='center', fontsize=11, fontweight='bold', color='#58a6ff',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f2937', edgecolor='#58a6ff', lw=1.5))

# Connectors
ax2.annotate('', xy=(0.25, 0.55), xytext=(0.42, 0.77),
             arrowprops=dict(arrowstyle="->", color='#8b949e', lw=2))
ax2.annotate('', xy=(0.75, 0.55), xytext=(0.58, 0.77),
             arrowprops=dict(arrowstyle="->", color='#8b949e', lw=2))

# Group 1 Box (2D)
ax2.text(0.25, 0.42, "Group 1: 2D Matrices\n(w_gate, w_up, q_proj...)\n\n✓ Weight Decay = 0.1\n(Prevents Explosion)",
         ha='center', va='center', fontsize=9, fontweight='bold', color='#3fb950',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#16281e', edgecolor='#3fb950', lw=1.5))

# Group 2 Box (1D)
ax2.text(0.75, 0.42, "Group 2: 1D Parameters\n(RMSNorm gamma, Biases)\n\n❌ Weight Decay = 0.0\n(Keeps Volume Knob Alive!)",
         ha='center', va='center', fontsize=9, fontweight='bold', color='#f85149',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a1818', edgecolor='#f85149', lw=1.5))

plt.tight_layout()
plt.savefig("notebooks/assets/optimizer_split.png", dpi=180, bbox_inches='tight')
plt.close()

print("Presentation diagrams generated successfully!")
