# -*- coding: utf-8 -*-
"""图1: 码本量化的几何动作 —— "查字典"
每个 BEV 位置的特征向量是空间中的一个点；码本是 nL 个锚点；
量化 = 把每个点换成离它最近的锚点的编号（整数码）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

rng = np.random.default_rng(7)

# 三类"特征点": 模拟不同语义区域的 BEV 特征向量(2D 示意, 实际是 C 维)
clusters = [
    (np.array([1.2, 1.0]), 0.35, 26),   # 前景: 车辆
    (np.array([3.4, 1.4]), 0.30, 20),   # 前景: 行人
    (np.array([2.2, 3.2]), 0.45, 30),   # 背景: 路面
]
pts = np.vstack([c + rng.normal(0, s, (n, 2)) for c, s, n in clusters])

# 码本: 6 个锚点 (实际 nL 更大, 这里示意)
codebook = np.array([
    [1.2, 1.0], [3.4, 1.4], [2.2, 3.3],
    [0.4, 2.6], [4.2, 3.0], [2.4, 0.3],
])

# 最近邻指派
d2 = ((pts[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
assign = d2.argmin(1)

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# 特征点
ax.scatter(pts[:, 0], pts[:, 1], s=26, c="#1a6fa3", alpha=0.55,
           edgecolors="none", zorder=2, label="BEV feature vectors")
# 指派连线: 点 -> 最近的码
for p, a in zip(pts[::3], assign[::3]):
    ax.plot([p[0], codebook[a, 0]], [p[1], codebook[a, 1]],
            c="#6e737b", lw=0.7, alpha=0.5, zorder=1)
# 码本锚点
ax.scatter(codebook[:, 0], codebook[:, 1], s=170, marker="D",
           facecolors="#f8ecd2", edgecolors="#9e5c00", linewidths=1.8,
           zorder=3, label="codebook anchors")
for k, (x, y) in enumerate(codebook):
    ax.text(x, y, f"{k}", ha="center", va="center", fontsize=10,
            color="#9e5c00", fontweight="bold", zorder=4)

ax.set_xlim(-0.4, 5.0)
ax.set_ylim(-0.6, 4.4)
ax.set_xlabel("feature dim 1 (示意: 实际为 C 维)")
ax.set_ylabel("feature dim 2 (示意: 实际为 C 维)")
ax.set_title("码本量化 = 把每个特征向量换成最近锚点的整数编号", fontsize=13, pad=10)
ax.legend(frameon=True, loc="upper left", fontsize=10)
ax.text(0.98, 0.02,
        "传输: 整数码 k (log2 nL 位)\n而非 C 维 float 向量 (32C 位)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#dcf2e8", ec="#0b7a5c", alpha=0.9))
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/codefilling-xray/fig-codebook-quant.png",
            bbox_inches="tight")
print("saved fig-codebook-quant.png")
