"""TTC scoring curve + score aggregation, with the numbers that reproduce Tab.3."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9

OUT, OP, WIRE = "#9e5c00", "#0b7a5c", "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]

tau = np.linspace(0.05, 4.0, 400)
score = np.where(tau > 2, 100.0, np.maximum(0, 100 - 1 / tau))
ax.plot(tau, score, lw=2.2, color=OUT)
ax.axvline(2.0, ls="--", color=WIRE, lw=1.2)
ax.text(2.05, 88, "2 s safety margin", fontsize=8, color=WIRE)
ax.set_xlabel("min TTC  (s)")
ax.set_ylabel("TTC score")
ax.set_title("TTC -> score is a soft cliff, not a wall", fontsize=10, weight="bold")
ax.set_xlim(0, 4)
ax.set_ylim(0, 105)
ax.grid(alpha=0.25)
# annotate a worked point
ax.plot([1.0], [99.0], "o", color=OUT, ms=7)
ax.annotate("gap 10 m, closing 10 m/s\ntau = 1.0 s -> 99",
            xy=(1.0, 99), xytext=(1.7, 70),
            arrowprops=dict(arrowstyle="->", color=WIRE), fontsize=8, color=WIRE)

ax2 = axes[1]
ax2.axis("off")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.text(5, 9.3, "Driving Score, worked from Tab.3 (GPT-4, 3-shot)", ha="center",
         fontsize=10, weight="bold")
lines = [
    "Score_i = 0.5 x 69.4 + 0.3 x 72.0 + 0.2 x 76.7",
    "        = 34.70 + 21.60 + 15.34 = 71.64",
    "",
    "Driving Score = alpha x mean(Score_i) - beta x 500",
    "              = 0.847 x 71.64 - 0.010 x 500",
    "              = 60.68 - 5.00 = 55.68  ~= 55.9 (table)",
    "",
    "collision penalty dominates:",
    "GPT-4-Turbo 0-shot: 53.9 - 15.5 = 38.4 ~= 39.1",
]
ax2.text(0.6, 8.2, "\n".join(lines), fontsize=9, family="monospace",
         va="top", color="#3d3d3a",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ed", ec=WIRE))
ax2.text(5, 0.5, "alpha = completion rate, beta = collision rate;\n"
         "timeouts score 0 but still count in alpha", ha="center", fontsize=8, color=WIRE)

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/078-Agent-lampilot/fig-scoring.png",
            dpi=150, bbox_inches="tight")
print("saved fig-scoring.png")
