"""
Generates the RSI Agent Autonomous Pretraining Loop diagram.
"""

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 180
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

fig, ax = plt.subplots(figsize=(11, 4.8), facecolor='#0d1117')
ax.set_facecolor('#161b22')
ax.axis('off')

# Title
ax.text(0.5, 0.94, "The Recursive Self-Improvement (RSI) Pretraining Loop",
        ha='center', va='center', fontsize=12, fontweight='bold', color='#58a6ff')
ax.text(0.5, 0.88, "How 1 Human Researcher + Autonomous Agents Pretrained Mirendil-Base",
        ha='center', va='center', fontsize=9.5, color='#8b949e')

# Boxes Definition
boxes = [
    ("1. Propose Hypothesis\n& Generate Config", 0.12, 0.55, '#1f2937', '#58a6ff'),
    ("2. Launch GPU Run\n(PyTorch / DDP)", 0.35, 0.55, '#16281e', '#3fb950'),
    ("3. Real-Time Monitor\n(Loss & Spike Check)", 0.58, 0.55, '#2a2214', '#d29922'),
    ("4. Evaluate Checkpoint\n(Val Loss & Pass@1)", 0.81, 0.55, '#271f3a', '#bc8cff'),
]

for text, x, y, bg, border in boxes:
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color=border,
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bg, edgecolor=border, lw=1.5))

# Forward Arrows
for x1, x2 in [(0.21, 0.27), (0.43, 0.50), (0.66, 0.73)]:
    ax.annotate('', xy=(x2, 0.55), xytext=(x1, 0.55),
                arrowprops=dict(arrowstyle="->", color='#8b949e', lw=2))

# Failure Recovery Loop (Triage)
ax.annotate('', xy=(0.35, 0.44), xytext=(0.58, 0.44),
            arrowprops=dict(arrowstyle="->", color='#f85149', lw=2, linestyle='--'))
ax.text(0.465, 0.36, "⚡ Anomaly Detected:\nRevert Checkpoint + Adjust LR",
        ha='center', va='center', fontsize=8, color='#f85149', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#211818', edgecolor='#f85149', lw=1))

# Global Iteration Loop (Ledger Feedback)
ax.annotate('', xy=(0.12, 0.67), xytext=(0.81, 0.67),
            arrowprops=dict(arrowstyle="->", color='#3fb950', lw=2, connectionstyle="arc3,rad=-0.25"))
ax.text(0.465, 0.77, "🔄 5. Update Experiment Ledger & Queue Next Ablation",
        ha='center', va='center', fontsize=9, color='#3fb950', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#16281e', edgecolor='#3fb950', lw=1))

# Human Role Annotation
ax.text(0.5, 0.12, "🧑‍🚀 Lone Human Role: Set High-Level Research Direction & Judge Final Models\n🤖 Agent Role: 99% of Code Writing, Cluster Orchestration, Anomaly Triage, & Ablation Sweeps",
        ha='center', va='center', fontsize=9, color='#c9d1d9',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f2937', edgecolor='#30363d', lw=1))

plt.tight_layout()
plt.savefig("notebooks/assets/rsi_agent_loop.png", dpi=180, bbox_inches='tight')
plt.close()
print("RSI agent diagram generated successfully!")
