"""Self-drawn figure for 133-WM-drivewm: Drive-WM tree-based rollout planning.
Shows: 3 candidate trajectories -> world model generates multiview future videos
-> image-based rewards (map x object) -> select best branch -> tree advances."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.35, 1]})
fig.patch.set_facecolor("#faf9f5")

CANDS = [
    ("Go right lane", "#1a6fa3", [0.62, 0.45, 0.50]),
    ("Go straight",   "#0b7a5c", [0.71, 0.55, 0.30]),
    ("Go left lane",   "#6a55c2", [0.78, 0.80, 0.82]),
]

# ---- Panel 1: candidates at time T ----
ax = axes[0]
ax.set_facecolor("#faf9f5")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
for i in range(3):
    ax.fill_betweenx([0, 10], 2.4 + i * 2.4, 4.4 + i * 2.4, color="#e8e6dc", alpha=0.8)
    ax.plot([2.4 + i * 2.4, 4.4 + i * 2.4], [0, 0], color="white", lw=2)
ax.add_patch(mpatches.Rectangle((3.2, 1.2), 1.2, 2.0, fc="#141413", ec="none", alpha=0.9))
ax.text(3.8, 0.6, "ego @ T", ha="center", fontsize=10, color="#141413")
for name, color, _ in CANDS:
    dx = {"Go right lane": 2.2, "Go straight": 0.0, "Go left lane": -2.2}[name]
    ax.annotate("", xy=(3.8 + dx, 8.6), xytext=(3.8, 3.4),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.4,
                                connectionstyle="arc3,rad=0.12" if dx else "arc3,rad=0"))
    ax.text(3.8 + dx * 1.02, 9.2, name, ha="center", fontsize=9, color=color, weight="bold")
ax.set_title("(a) Planner samples 3 candidates at T", fontsize=11, color="#141413")
ax.axis("off")

# ---- Panel 2: world model futures + rewards ----
ax = axes[1]
ax.set_facecolor("#faf9f5")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(5, 9.5, "World model imagines each future (multiview video)", ha="center",
        fontsize=10, color="#141413")
for k, (name, color, rws) in enumerate(CANDS):
    y0 = 7.6 - k * 2.5
    # imagined future strip: frames T+K, T+2K with reward bars
    for f, (tlabel, rw) in enumerate(zip(["T+K", "T+2K"], rws[1:])):
        x0 = 1.2 + f * 3.4
        ax.add_patch(mpatches.Rectangle((x0, y0 - 0.9), 2.6, 1.5, fc=color, ec="none", alpha=0.16))
        ax.add_patch(mpatches.Rectangle((x0, y0 - 0.9), 2.6, 1.5, fc="none", ec=color, lw=1.2))
        ax.text(x0 + 1.3, y0 - 0.15, f"{tlabel}", ha="center", fontsize=8, color=color)
        # reward bar
        ax.add_patch(mpatches.Rectangle((x0 + 0.2, y0 - 0.75), 2.2 * rw, 0.28,
                                        fc="#9e5c00", ec="none", alpha=0.85))
        ax.text(x0 + 2.45, y0 - 0.61, f"{rw:.2f}", ha="right", va="center", fontsize=7.5,
                color="#9e5c00")
    ax.text(0.15, y0 - 0.15, name, ha="left", va="center", fontsize=9, color=color, weight="bold")
    if name == "Go straight":
        ax.text(8.9, y0 - 0.15, "drops at T+2K\n(too close to\nfront car)",
                fontsize=7.5, color="#b34a06", va="center")
ax.text(5, 0.35, "reward = map reward x object reward  (image-based, on generated video)",
        ha="center", fontsize=9, color="#3d3d3a", style="italic")
ax.set_title("(b) Reward each imagined future", fontsize=11, color="#141413")
ax.axis("off")

# ---- Panel 3: decision ----
ax = axes[2]
ax.set_facecolor("#faf9f5")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
for i in range(3):
    ax.fill_betweenx([0, 10], 2.4 + i * 2.4, 4.4 + i * 2.4, color="#e8e6dc", alpha=0.8)
ax.add_patch(mpatches.Rectangle((3.2, 1.2), 1.2, 2.0, fc="#141413", ec="none", alpha=0.9))
# rejected (faded)
ax.annotate("", xy=(6.0, 8.6), xytext=(3.8, 3.4),
            arrowprops=dict(arrowstyle="->", color="#1a6fa3", lw=1.5, alpha=0.25,
                            connectionstyle="arc3,rad=0.12"))
ax.annotate("", xy=(3.8, 8.6), xytext=(3.8, 3.4),
            arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=1.5, alpha=0.25))
# accepted
ax.annotate("", xy=(1.6, 8.6), xytext=(3.8, 3.4),
            arrowprops=dict(arrowstyle="->", color="#6a55c2", lw=3.2,
                            connectionstyle="arc3,rad=0.12"))
ax.text(1.6, 9.3, "ACCEPTED: left lane", ha="center", fontsize=10, color="#6a55c2", weight="bold")
ax.text(6.0, 9.3, "rejected", ha="center", fontsize=9, color="#1a6fa3", alpha=0.5)
ax.text(3.8, 9.3, "rejected", ha="center", fontsize=9, color="#0b7a5c", alpha=0.5)
ax.text(5, 0.5, "tree advances one step; repeat at T+1",
        ha="center", fontsize=9, color="#3d3d3a", style="italic")
ax.set_title("(c) Max-reward branch extends the tree", fontsize=11, color="#141413")
ax.axis("off")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/133-WM-drivewm/fig-tree.png"
plt.savefig(out, dpi=150, facecolor="#faf9f5")
print("saved", out)
