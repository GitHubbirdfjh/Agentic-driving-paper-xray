"""Bar chart decomposing Table II minADE across ablation variants (Co-MTP paper)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

labels = ["1 veh-only", "2 infra-only", "3 stitch",
          "4 stitch+\nfull-infra", "5 +AV plan",
          "6 +other fut", "Co-HTTP", "Co-MTP"]
values = np.array([0.9887, 0.9375, 0.9123, 0.8518, 0.8378, 0.7992, 0.8258, 0.7647])

# stage colors: history-data / fusion-method / future-domain
colors = ["#1a6fa3", "#1a6fa3", "#1a6fa3",
          "#6a55c2",
          "#9e5c00", "#9e5c00", "#9e5c00", "#0b7a5c"]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.2, width=0.68, zorder=3)

for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9.5)

# annotate deltas between key neighbors
def delta(i, j, y, text):
    ax.annotate("", xy=(j, y), xytext=(i, y),
                arrowprops=dict(arrowstyle="<->", color="#3d3d3a", lw=1.2))
    ax.text((i + j) / 2, y + 0.028, text, ha="center", fontsize=9,
            color="#3d3d3a", style="italic")

top = values.max() + 0.10
delta(0, 2, top, "-7.7% multi-agent data (stitch)")
delta(2, 3, top - 0.055, "-6.6% method+data confounded")
delta(3, 7, top - 0.11, "-10.2% future domain")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("minADE (K=6) ↓", fontsize=11)
ax.set_title("Co-MTP ablation (V2X-Seq, Table II): where the minADE gains come from",
             fontsize=12, pad=14)
ax.set_ylim(0, top + 0.10)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

from matplotlib.patches import Patch
legend = [Patch(facecolor="#1a6fa3", label="history data availability"),
          Patch(facecolor="#6a55c2", label="fusion method (+more infra data)"),
          Patch(facecolor="#9e5c00", label="future-domain variants"),
          Patch(facecolor="#0b7a5c", label="full Co-MTP")]
ax.legend(handles=legend, fontsize=9, loc="upper right", framealpha=0.95)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/co-mtp-xray/fig-ablation.png",
            dpi=150)
print("saved")
