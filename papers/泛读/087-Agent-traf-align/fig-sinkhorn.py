"""Offset loss 的 Sinkhorn 软匹配：n=3 的小例子。
生成的 3 个点（圆）与真值的 3 个点（叉）无序，
代价矩阵 C 是 L1 距离，Sinkhorn 给出软匹配矩阵 P，
损失 = sum(P * C)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

gen = np.array([[2, 3], [5, 1], [4, 4]])   # 生成的点
gt = np.array([[2, 4], [5, 2], [4, 5]])     # 真值点
C = np.abs(gen[:, None, :] - gt[None, :, :]).sum(-1)  # L1 代价

# 简化的 Sinkhorn（熵正则，迭代归一化）
Kmat = np.exp(-C / 0.5)
u = np.ones(3)
for _ in range(50):
    v = 1.0 / (Kmat.T @ u + 1e-12)
    u = 1.0 / (Kmat @ v + 1e-12)
P = np.diag(u) @ Kmat @ np.diag(v)
loss = float((P * C).sum())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor="white")

ax = axes[0]
for i, (x, y) in enumerate(gen):
    ax.plot(x, y, "o", color="#1a6fa3", ms=10)
    ax.text(x + 0.15, y + 0.15, f"g{i+1}", color="#1a6fa3", fontsize=11)
for i, (x, y) in enumerate(gt):
    ax.plot(x, y, "x", color="#b34a06", ms=10, mew=2.5)
    ax.text(x + 0.15, y + 0.15, f"t{i+1}", color="#b34a06", fontsize=11)
# 按 P 画匹配线，粗细正比于匹配概率
for i in range(3):
    for j in range(3):
        if P[i, j] > 0.05:
            ax.plot([gen[i, 0], gt[j, 0]], [gen[i, 1], gt[j, 1]],
                    color="#6a55c2", lw=P[i, j] * 6, alpha=0.7)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.set_title("(a) 点集与软匹配", fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax = axes[1]
im = ax.imshow(C, cmap="YlOrBr", vmin=0)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=12)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(["t1", "t2", "t3"])
ax.set_yticklabels(["g1", "g2", "g3"])
ax.set_title(r"(b) 代价矩阵 $C$（L1 距离）", fontsize=12)

ax = axes[2]
im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=12,
                color="white" if P[i, j] > 0.5 else "black")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(["t1", "t2", "t3"])
ax.set_yticklabels(["g1", "g2", "g3"])
ax.set_title(r"(c) Sinkhorn 匹配 $P$（双随机）", fontsize=12)

fig.suptitle(r"$\mathcal{L}_{offset}=\sum_{j,k}P_{jk}C_{jk}$" +
             f" = {loss:.2f}（本例示意）", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/087-Agent-traf-align/fig-sinkhorn.png",
            dpi=150, bbox_inches="tight")
print("saved fig-sinkhorn.png")
