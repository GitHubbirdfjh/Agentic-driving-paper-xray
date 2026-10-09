"""Fig: ablation bars - AP@0.5 drop when removing each component (Table 3).
Shows which component is load-bearing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, WIRE, LOSS = "#faf9f5", "#6e737b", "#b34a06"
LOSS_F = "#f9e3d6"

# full model AP@0.5 per scenario, and AP after removing component
full = {"pp8-pp4": 77.2, "pp8-sd1": 79.4, "pp8-vn6": 72.8}
removed = {
    "w/o channel selection": {"pp8-pp4": 75.6, "pp8-sd1": 78.1, "pp8-vn6": 64.1},
    "w/o spatial attention": {"pp8-pp4": 64.4, "pp8-sd1": 65.9, "pp8-vn6": 62.0},
    "w/o general prompt": {"pp8-pp4": 78.3, "pp8-sd1": 78.0, "pp8-vn6": 70.9},
    "w/o specific prompt": {"pp8-pp4": 23.3, "pp8-sd1": 38.7, "pp8-vn6": 14.3},
    "w/o prompt init\n(random)": {"pp8-pp4": 76.8, "pp8-sd1": 78.2, "pp8-vn6": 68.7},
}
scenarios = ["pp8-pp4", "pp8-sd1", "pp8-vn6"]
labels = list(removed.keys())

drops = {lab: [full[s] - removed[lab][s] for s in scenarios] for lab in labels}

x = np.arange(len(labels))
w = 0.22
fig, ax = plt.subplots(figsize=(11, 4.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
colors = ["#1a6fa3", "#0b7a5c", "#9e5c00"]
for k, s in enumerate(scenarios):
    vals = [drops[lab][k] for lab in labels]
    bars = ax.bar(x + (k - 1) * w, vals, width=w, label=s, color=colors[k],
                  edgecolor=colors[k], linewidth=1.2)
    for b, v in zip(bars, vals):
        if abs(v) > 3:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=LOSS)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("AP@0.5 drop (percentage points)", fontsize=11, color=WIRE)
ax.set_title("Ablation (Table 3): how much AP@0.5 is lost when a component is removed",
             fontsize=12.5)
ax.axhline(0, color=WIRE, linewidth=0.8)
ax.legend(frameon=False, fontsize=10)
ax.set_ylim(-3, 62)
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/085-Agent-polyinter/fig-ablation.png",
            dpi=150, facecolor=BG)
print("saved")
