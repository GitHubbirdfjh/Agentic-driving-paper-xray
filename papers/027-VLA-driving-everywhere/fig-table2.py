"""Self-drawn figure for 027 xray: magnitude of Table 2 improvements (LLaDA paper).

Shows that the headline claim "outperforms on all metrics" rests on a 1cm L2
reduction and a 0.02pp collision-rate reduction, with 1s-horizon metrics unchanged.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/hatch/workspace/your_files/paper-xray-100/papers/027-VLA-driving-everywhere/fig-table2.png"

# Table 2 numbers, verified against the paper text
horizons = ["1s", "2s", "3s", "Avg."]
l2_base = np.array([0.27, 0.59, 1.04, 0.63])
l2_ours = np.array([0.27, 0.58, 1.02, 0.62])
cr_base = np.array([0.22, 0.45, 1.07, 0.58])
cr_ours = np.array([0.22, 0.41, 1.04, 0.56])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
x = np.arange(len(horizons))
w = 0.36

for ax, base, ours, title, unit in [
    (axes[0], l2_base, l2_ours, "L2 error (m)  lower is better", "m"),
    (axes[1], cr_base, cr_ours, "Collision rate (%)  lower is better", "%"),
]:
    b1 = ax.bar(x - w / 2, base, w, label="GPT-Driver", color="#9aa5b1", edgecolor="white")
    b2 = ax.bar(x + w / 2, ours, w, label="GPT-Driver + LLaDA", color="#2f6f4f", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(unit)
    # annotate deltas
    for i in range(len(horizons)):
        d = base[i] - ours[i]
        if d > 0:
            ax.annotate(f"-{d:.2f}", xy=(x[i] + w / 2, ours[i]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=9, color="#2f6f4f",
                        fontweight="bold")
        else:
            ax.annotate("no change", xy=(x[i], base[i]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color="#666666")

axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Table 2 replotted: how big is 'outperforms on all metrics' really?",
             fontsize=12, fontweight="bold", y=1.02)
fig.text(0.5, -0.02,
         "Singapore-tuned GPT-Driver evaluated on Boston (nuScenes val). Avg L2: 0.63 -> 0.62 m (-1 cm); "
         "avg collision: 0.58 -> 0.56 % (-0.02 pp). No seeds / error bars reported.",
         ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
