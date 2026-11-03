"""
Generates the Chinchilla Scaling Law Pareto Frontier diagram.
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

# Panel 1: IsoFLOP Curves and Pareto Frontier
tokens = np.linspace(10, 200, 300)

# 3 Different compute budgets (FLOPs)
loss_budget_1 = 3.2 + 8.0 / (tokens**0.4) + 0.005 * tokens
loss_budget_2 = 2.6 + 6.0 / (tokens**0.4) + 0.003 * tokens
loss_budget_3 = 2.0 + 4.5 / (tokens**0.4) + 0.0018 * tokens

ax1.plot(tokens, loss_budget_1, color='#d29922', lw=1.8, linestyle='--', label='Compute Budget C1')
ax1.plot(tokens, loss_budget_2, color='#bc8cff', lw=1.8, linestyle='--', label='Compute Budget C2')
ax1.plot(tokens, loss_budget_3, color='#58a6ff', lw=1.8, linestyle='--', label='Compute Budget C3')

# Minima of each curve (Pareto optimal points)
min1_idx = np.argmin(loss_budget_1)
min2_idx = np.argmin(loss_budget_2)
min3_idx = np.argmin(loss_budget_3)

pareto_x = [tokens[min1_idx], tokens[min2_idx], tokens[min3_idx]]
pareto_y = [loss_budget_1[min1_idx], loss_budget_2[min2_idx], loss_budget_3[min3_idx]]

ax1.plot(pareto_x, pareto_y, color='#3fb950', lw=2.5, marker='o', markersize=8, label='Chinchilla Optimal Frontier')
ax1.text(tokens[min3_idx] + 8, loss_budget_3[min3_idx] - 0.05, 'Optimal (D ≈ 20 × N)', color='#3fb950', fontsize=9, fontweight='bold')

ax1.set_title("IsoFLOP Curves & The Chinchilla Frontier", fontsize=11, fontweight='bold', color='#3fb950', pad=10)
ax1.set_xlabel("Tokens (D) for Given Parameters (N)", fontsize=9)
ax1.set_ylabel("Validation Loss", fontsize=9)
ax1.grid(True, color='#30363d', alpha=0.5)
ax1.legend(loc='upper right', fontsize=8.5)

# Panel 2: The Historic Shift (Gopher vs Chinchilla)
ax2.axis('off')
ax2.set_title("The Historic Paradigm Shift (2020 vs 2022)", fontsize=11, fontweight='bold', color='#58a6ff', pad=12)

table_data = [
    ["Gopher (2021)", "280 Billion", "300 Billion", "1.07 : 1 (Severely Undertrained!)"],
    ["GPT-3 (2020)", "175 Billion", "300 Billion", "1.71 : 1 (Undertrained)"],
    ["Chinchilla (2022)", "70 Billion", "1.4 Trillion", "20 : 1 (Compute Optimal!)"],
    ["Llama 3 (2024)", "8 Billion", "15 Trillion", "1875 : 1 (Over-trained for Fast Inference)"]
]

col_labels = ["Model", "Parameters (N)", "Tokens (D)", "Ratio (Tokens / Params)"]
table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 1.8)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#30363d')
    if r == 0:
        cell.set_facecolor('#1f2937')
        cell.set_text_props(color='#58a6ff', fontweight='bold')
    elif r == 3:
        cell.set_facecolor('#1c2738')
        cell.set_text_props(color='#3fb950', fontweight='bold')
    elif r == 4:
        cell.set_facecolor('#20293a')
        cell.set_text_props(color='#58a6ff', fontweight='bold')
    else:
        cell.set_facecolor('#161b22')
        cell.set_text_props(color='#c9d1d9')

plt.tight_layout()
plt.savefig("notebooks/assets/scaling_laws.png", dpi=180, bbox_inches='tight')
plt.close()
print("Scaling laws diagram generated successfully!")
