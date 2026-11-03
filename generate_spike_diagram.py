"""
Generates the Warm Restart Loss Spike diagram comparing Cosine Restart vs WSD.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set dark theme styling
plt.rcParams['figure.dpi'] = 180
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor='#0d1117')
ax1.set_facecolor('#161b22')
ax2.set_facecolor('#161b22')

# Panel 1: What happens if you re-increase LR (Cosine with Warm Restarts)
steps = np.linspace(0, 1000, 1000)

# Simulate LR and Loss spike
lr_restart = np.zeros_like(steps)
loss_restart = np.zeros_like(steps)

for i, s in enumerate(steps):
    if s < 500:
        # First cycle
        p = s / 500
        lr_restart[i] = 0.0001 + 0.0009 * 0.5 * (1 + np.cos(np.pi * p))
        loss_restart[i] = 4.0 - 2.0 * p + 0.1 * np.sin(s/10)
    else:
        # Re-increase LR spike at step 500!
        p = (s - 500) / 500
        lr_restart[i] = 0.0001 + 0.0009 * 0.5 * (1 + np.cos(np.pi * p))
        # Loss explodes back up from 2.0 to 3.8 and slowly recovers
        loss_restart[i] = 2.0 + 1.8 * np.exp(-(s - 500) / 80) - 0.5 * p + 0.1 * np.sin(s/10)

ax1.plot(steps, loss_restart, color='#f85149', lw=2.2, label='Training Loss')
ax1.axvline(500, color='#d29922', linestyle='--', lw=2)
ax1.text(500, 3.7, '⚡ Sudden LR Spike!\n(Weights blown out of valley)', color='#f85149', fontsize=9, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#f85149'))

ax1.set_title("Re-Increasing LR: Catastrophic Loss Spike", fontsize=11, fontweight='bold', color='#f85149', pad=10)
ax1.set_xlabel("Steps", fontsize=9)
ax1.set_ylabel("Loss", fontsize=9)
ax1.grid(True, color='#30363d', alpha=0.5)

# Panel 2: The WSD Solution
loss_wsd = np.zeros_like(steps)
for i, s in enumerate(steps):
    if s < 800:
        loss_wsd[i] = 4.0 - 1.8 * (s / 800) + 0.08 * np.sin(s/10)
    else:
        # Decay phase drops smoothly
        p = (s - 800) / 200
        loss_wsd[i] = 2.2 - 1.2 * p + 0.03 * np.sin(s/10)

ax2.plot(steps, loss_wsd, color='#58a6ff', lw=2.5, label='WSD Loss')
ax2.axvspan(800, 1000, color='#3fb950', alpha=0.12)
ax2.text(900, 2.0, 'Single Smooth\nDecay Phase\n(No spikes!)', color='#3fb950', fontsize=9, fontweight='bold', ha='center')

ax2.set_title("WSD Solution: Stable Learning + Single Clean Landing", fontsize=11, fontweight='bold', color='#58a6ff', pad=10)
ax2.set_xlabel("Steps", fontsize=9)
ax2.grid(True, color='#30363d', alpha=0.5)

plt.tight_layout()
plt.savefig("notebooks/assets/lr_restart_spike.png", dpi=180, bbox_inches='tight')
plt.close()
print("Loss spike diagram generated successfully!")
