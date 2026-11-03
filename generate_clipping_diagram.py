"""
Generates the Gradient Clipping vector geometry diagram.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 180
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor='#0d1117')
ax1.set_facecolor('#161b22')
ax2.set_facecolor('#161b22')

# Panel 1: Vector Geometry on 2D plane
# Threshold circle of radius 1.0
circle = plt.Circle((0, 0), 1.0, color='#3fb950', fill=True, alpha=0.1, lw=2, linestyle='--')
ax1.add_artist(circle)
circle_border = plt.Circle((0, 0), 1.0, color='#3fb950', fill=False, lw=1.5, linestyle='--')
ax1.add_artist(circle_border)

# Dangerous gradient [3.0, 4.0]
ax1.annotate('', xy=(3.0, 4.0), xytext=(0, 0),
             arrowprops=dict(facecolor='#f85149', edgecolor='#f85149', width=2.5, headwidth=9))
ax1.text(2.6, 4.2, 'Dangerous Raw Gradient\nNorm = 5.0 (Crashes GPU!)', color='#f85149', fontsize=9, fontweight='bold')

# Clipped gradient [0.6, 0.8]
ax1.annotate('', xy=(0.6, 0.8), xytext=(0, 0),
             arrowprops=dict(facecolor='#58a6ff', edgecolor='#58a6ff', width=3.5, headwidth=10))
ax1.text(0.7, 0.5, 'Clipped Gradient\nNorm = 1.0 (Safe!)', color='#58a6ff', fontsize=9, fontweight='bold')

ax1.set_xlim(-0.5, 4.0)
ax1.set_ylim(-0.5, 5.0)
ax1.set_aspect('equal')
ax1.set_title("Vector Geometry: Scale Down, Keep Direction", fontsize=11, fontweight='bold', color='#58a6ff', pad=10)
ax1.set_xlabel("Gradient Dimension 1", fontsize=9)
ax1.set_ylabel("Gradient Dimension 2", fontsize=9)
ax1.grid(True, color='#30363d', alpha=0.5)

# Panel 2: The Training Loss Trajectory
steps = np.arange(0, 100)
# Loss without clipping (explodes at step 50)
loss_unclipped = 4.0 - 1.5 * (steps / 100)
loss_unclipped[50:] = np.nan  # NaN crash!

# Loss with clipping (stays stable)
loss_clipped = 4.0 - 1.5 * (steps / 100) + 0.05 * np.sin(steps / 2)
# Tiny bump at step 50
loss_clipped[50] += 0.3

ax2.plot(steps[:50], loss_unclipped[:50], color='#8b949e', lw=1.5)
ax2.plot([50, 50], [loss_unclipped[49], 8.0], color='#f85149', lw=2.5, linestyle=':')
ax2.scatter([50], [8.0], color='#f85149', marker='x', s=100, zorder=5)
ax2.text(52, 7.5, '❌ Without Clipping:\nNaN Explosion at Step 50!', color='#f85149', fontsize=9, fontweight='bold')

ax2.plot(steps, loss_clipped, color='#3fb950', lw=2.5, label='With Gradient Clipping (max_norm=1.0)')
ax2.text(52, 2.5, '✅ With Clipping:\nSurge absorbed safely!', color='#3fb950', fontsize=9, fontweight='bold')

ax2.set_title("Run Survival: The Circuit Breaker in Action", fontsize=11, fontweight='bold', color='#3fb950', pad=10)
ax2.set_xlabel("Training Steps", fontsize=9)
ax2.set_ylabel("Loss", fontsize=9)
ax2.set_ylim(1.5, 8.5)
ax2.grid(True, color='#30363d', alpha=0.5)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig("notebooks/assets/grad_clipping.png", dpi=180, bbox_inches='tight')
plt.close()
print("Grad clipping diagram generated successfully!")
