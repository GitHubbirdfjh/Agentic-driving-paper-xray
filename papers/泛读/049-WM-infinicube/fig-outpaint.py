# -*- coding: utf-8 -*-
"""InfiniCube Fig 1: 无界体素外推 (outpainting) —— RePaint 式潜变量拼接.
新 chunk 与已有场景重叠; 扩散去噪时重叠区潜变量 X_exist 固定 (加噪版本),
只更新新区域 X_new: X_new <- (1-M)*X_hat_new + M*X_hat_exist (§4.1, Eq.1).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BLUE, GREEN, ORANGE, PURPLE = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6a55c2"
GRAY = "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.patch.set_facecolor("white")
fig.suptitle("无界体素世界生成: 重叠区潜变量固定, 新区域才去噪 (§4.1, Eq.1)", fontsize=13,
             weight="bold")

# ---------- 左: 几何示意 ----------
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("外推过程 (俯视)", fontsize=11, weight="bold")

# 已有场景 chunk
ax.add_patch(Rectangle((0.5, 1.0), 5.0, 4.0, fc="#e2eff5", ec=BLUE, lw=2))
ax.text(3.0, 4.5, "已有场景", ha="center", fontsize=10, color=BLUE, weight="bold")
ax.text(3.0, 4.1, "X_exist (已生成)", ha="center", fontsize=9, color=BLUE)
# 新 chunk
ax.add_patch(Rectangle((4.0, 1.0), 5.0, 4.0, fc="#f8ecd2", ec=ORANGE, lw=2, ls="--"))
ax.text(6.5, 4.5, "新 chunk", ha="center", fontsize=10, color=ORANGE, weight="bold")
ax.text(6.5, 4.1, "X_new (待去噪)", ha="center", fontsize=9, color=ORANGE)
# 重叠区
ax.add_patch(Rectangle((4.0, 1.0), 1.5, 4.0, fc="#6a55c2", ec=PURPLE, lw=2, alpha=0.35))
ax.text(4.75, 2.6, "重叠区 M=1", ha="center", fontsize=9, color=PURPLE, weight="bold")
ax.text(4.75, 2.1, "潜变量固定\n只加噪不更新", ha="center", fontsize=8.5, color=PURPLE)
# 网格纹理示意
rng = np.random.default_rng(3)
for (x0, y0, w, h, c) in [(0.5, 1.0, 5.0, 4.0, BLUE), (5.5, 1.0, 3.5, 4.0, ORANGE)]:
    for _ in range(40):
        px, py = rng.uniform(x0, x0 + w), rng.uniform(y0, y0 + h)
        if 4.0 < px < 5.5:
            continue
        ax.plot(px, py, "s", ms=3, color=c, alpha=0.5)
ax.text(5.0, 0.55, "每个 chunk 51.2m×51.2m, 体素 0.2m; 新 chunk 与已有部分充分重叠后逐块外推", ha="center",
        fontsize=9, color="#3d3d3a", style="italic")

# ---------- 右: 方程与去噪步骤 ----------
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("扩散去噪的每一步发生了什么", fontsize=11, weight="bold")

ax.text(5, 5.3, r"$X^{vx}_{new} \leftarrow (1-M)\odot \hat{X}^{vx}_{new} + M\odot \hat{X}^{vx}_{exist}$",
        fontsize=13, ha="center", color="#141413", weight="bold",
        bbox=dict(fc="#f5f4ed", ec="none", pad=8))

items = [
    ("M (重叠掩码)", "重叠区=1, 新区域=0\n逐元素乘 ⊙", "#6a55c2"),
    (r"$\hat{X}_{new}$", "3D U-Net 从学到的\n后验采样 (去噪更新)", ORANGE),
    (r"$\hat{X}_{exist}$", "已有潜变量 X_exist\n加噪到当前 t 的版本\n(固定锚点, 不更新)", BLUE),
]
y = 4.0
for title, body, c in items:
    b = FancyBboxPatch((1.2, y - 0.75), 7.6, 1.05, boxstyle="round,pad=0.02", fc=c, ec="none", alpha=0.9)
    ax.add_patch(b)
    ax.text(2.6, y - 0.22, title, fontsize=10, weight="bold", color="white", va="center")
    ax.text(6.8, y - 0.22, body, fontsize=9, color="white", va="center", linespacing=1.4)
    y -= 1.35

ax.text(5, 0.45, "直觉: 重叠区被钉死, 新区域的去噪被迫与它衔接 —— 训练无关 (training-free), 逐 chunk 无缝拼接。",
        fontsize=9, ha="center", color="#3d3d3a", style="italic",
        bbox=dict(fc="#ebe7f8", ec="none", pad=6))

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/049-WM-infinicube/fig-outpaint.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
