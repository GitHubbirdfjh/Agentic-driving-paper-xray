"""Pyramid Fusion 前景加权选择机制的玩具算例：
两个智能体的前景图 -> 每个 BEV 格子上的 softmax 权重。
与正文第 5 节的数字完全对应。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# 0-indexed cells; true vehicles at (0,0) and (1,2); camera false alarm at (3,3)
S1 = np.full((4, 4), 0.10)   # agent 1 (LiDAR): calibrated
S1[0, 0] = 0.90
S1[1, 2] = 0.80
S2 = np.full((4, 4), 0.10)   # agent 2 (camera): weak + one false alarm
S2[0, 0] = 0.30
S2[1, 2] = 0.25
S2[3, 3] = 0.70

E1, E2 = np.exp(S1), np.exp(S2)
W1 = E1 / (E1 + E2)          # softmax over agents, per cell -> weight of agent 1
W2 = 1.0 - W1

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
panels = [
    (S1, "agent 1 (LiDAR)\nforeground map S", "#1f5b99"),
    (S2, "agent 2 (camera)\nforeground map S", "#8a5a1b"),
    (W1, "fusion weight of agent 1\nW = softmax(S1, S2) over agents", "#1f5b99"),
]
vmax = 1.0
for ax, (M, title, _) in zip(axes, panels):
    im = ax.imshow(M, vmin=0, vmax=vmax, cmap="YlOrRd")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[r, c] > 0.55 else "black")
    # mark the three interesting cells
    for (r, c), lab in [((0, 0), "car A"), ((1, 2), "car B"), ((3, 3), "false alarm")]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="#0b7a5c", lw=2))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Toy example: per-cell, per-scale agent selection in Pyramid Fusion\n"
             "car A: W1 = e^0.90/(e^0.90+e^0.30) = 0.65  |  "
             "car B: W1 = e^0.80/(e^0.80+e^0.25) = 0.63  |  "
             "false alarm: W1 = e^0.10/(e^0.10+e^0.70) = 0.35",
             fontsize=10.5, y=1.02)
fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/heal-xray/fig-foreground-select.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
print("check: W1(0,0)=%.3f W1(1,2)=%.3f W1(3,3)=%.3f" % (W1[0,0], W1[1,2], W1[3,3]))
