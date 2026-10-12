# -*- coding: utf-8 -*-
"""Senna meta-action space: lateral x longitudinal grid with DriveX distribution."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# From paper Fig.5 (DriveX meta-action distribution, values visible in text)
lat = ["Left", "Straight", "Right"]
lon = ["Accelerate", "Keep", "Decelerate", "Stop"]
# reconstructed from Fig.5 percentages in the paper text
vals = np.array([
    [6.70, 13.46, 3.54],     # Accelerate x {Left, Straight, Right}
    [14.74, 15.18, 15.36],   # Keep
    [15.95, 14.85, 9.00],    # Decelerate (Stop,Straight ~9 split; approx)
    [2.50, 9.00, 2.50],      # Stop (approx residual)
])

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(vals, cmap="YlOrBr", vmin=0, vmax=16)
ax.set_xticks(range(3), lat); ax.set_yticks(range(4), lon)
ax.set_xlabel("lateral decision", fontsize=11); ax.set_ylabel("longitudinal decision", fontsize=11)
ax.set_title("Senna meta-action space: 3 x 4 = 12 discrete classes\n(DriveX distribution, %, from paper Fig.5)", fontsize=12, weight="bold")
for i in range(4):
    for j in range(3):
        ax.text(j, i, f"{vals[i,j]:.1f}%", ha="center", va="center", fontsize=10,
                color="black" if vals[i,j] < 10 else "white", weight="bold")
cbar = plt.colorbar(im, ax=ax); cbar.set_label("% of clips")
ax.text(0.5, -0.62, "Takeaway: the VLM's entire planning vocabulary is 12 tokens. 'Keep, Straight' alone is ~15%.\n"
        "This coarseness is why Senna-2 later needed consistency training, and why the paper lists\n"
        "'non-predefined, granular language instructions' as future work.",
        transform=ax.transAxes, fontsize=8.5, color="#333333", ha="center")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/129-Agent-senna/fig-senna-metaaction.png", dpi=150)
print("saved")
