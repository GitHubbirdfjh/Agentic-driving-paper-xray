# -*- coding: utf-8 -*-
"""InfiniCube Fig 2: guidance buffers 如何把 25 帧的 SVD-XT 拉长到 200 帧.
Cvd = {C_Img(4ch 首帧), C_Sem(4ch 语义), C_Crd(4ch 坐标)} 共 12 通道;
自回归: 上一遍最后一帧 latent 复用为条件; 每一遍都有体素渲染的 buffer 做 3D 锚定.
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

fig, ax = plt.subplots(figsize=(13, 6.4))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("世界引导的视频生成: guidance buffers 给每一遍自回归钉上 3D 锚 (§4.2)", fontsize=13,
             weight="bold", pad=12)

# ---- 顶部: 条件结构 ----
ax.text(0.5, 6.5, "视频扩散模型的条件 C_vd (12 通道, 与 latent 同空间尺寸 h×w×T)", fontsize=11,
        weight="bold", color="#141413")
conds = [
    ("C_Img (4ch)", "首帧图像\n重复 T 次后编码", BLUE),
    ("C_Sem (4ch)", "语义 buffer: 体素语义\n渲染成 RGB (实例用\nPuRd 色图区分)", GREEN),
    ("C_Crd (4ch)", "坐标 buffer: 每像素\n射线首交体素的 3D\n坐标, 归一到 [-1,1]", ORANGE),
]
x = 0.5
for title, body, c in conds:
    b = FancyBboxPatch((x, 5.0), 3.6, 1.3, boxstyle="round,pad=0.02", fc=c, ec="none", alpha=0.9)
    ax.add_patch(b)
    ax.text(x + 1.8, 6.05, title, ha="center", fontsize=10, weight="bold", color="white")
    ax.text(x + 1.8, 5.55, body, ha="center", fontsize=8.5, color="white", linespacing=1.35)
    x += 3.9
ax.add_patch(FancyArrowPatch((11.6, 5.65), (12.5, 5.65), arrowstyle="-|>", mutation_scale=14,
                             color=GRAY, lw=1.6))
ax.text(12.1, 5.95, "concat", ha="center", fontsize=8, color=GRAY)

# ---- 中部: 自回归时间轴 ----
ax.text(0.5, 4.5, "自回归长视频生成 (25 帧/遍 × 8 遍 = 200 帧)", fontsize=11, weight="bold",
        color="#141413")
n_pass, frames = 8, 25
x0, bw, gap = 0.5, 1.28, 0.18
for i in range(n_pass):
    xx = x0 + i * (bw + gap)
    fc = "#dcf2e8" if i % 2 == 0 else "#e2eff5"
    ec = GREEN if i % 2 == 0 else BLUE
    ax.add_patch(Rectangle((xx, 2.6), bw, 1.2, fc=fc, ec=ec, lw=1.8))
    ax.text(xx + bw / 2, 3.35, f"第{i+1}遍", ha="center", fontsize=9, weight="bold", color=ec)
    ax.text(xx + bw / 2, 2.95, f"帧 {i*frames+1}–{(i+1)*frames}", ha="center", fontsize=7.5, color="#3d3d3a")
    if i < n_pass - 1:
        ax.add_patch(FancyArrowPatch((xx + bw, 3.2), (xx + bw + gap, 3.2), arrowstyle="-|>",
                                     mutation_scale=12, color="#b34a06", lw=1.8))
        if i == 0:
            ax.text(xx + bw + gap / 2, 3.55, "末帧 latent\n复用为条件", ha="center", fontsize=7,
                    color="#b34a06")

# 3D 锚定条
ax.add_patch(Rectangle((x0, 1.7), n_pass * (bw + gap) - gap, 0.55, fc="#ebe7f8", ec=PURPLE, lw=1.5))
ax.text(x0 + (n_pass * (bw + gap) - gap) / 2, 1.98,
        "每一遍都有体素世界渲染的语义/坐标 buffer 做 3D  grounding —— 抑制自回归累积误差",
        ha="center", fontsize=9.5, color=PURPLE, weight="bold")

# ---- 底部: 坐标 buffer 的关键性质 ----
ax.text(0.5, 1.15, "坐标 buffer 为什么有效: 同一 3D 体素在不同帧投影到不同像素, 但像素值 (3D 坐标) 不变 ——",
        fontsize=10, color="#141413", weight="bold")
ax.text(0.5, 0.75, "跨帧对应关系被写进像素值里, 视频模型不用自己猜运动; 语义 buffer 重复纹理 (如一排相同的窗) 造成歧义时, 坐标 buffer 消歧。",
        fontsize=9.5, color="#3d3d3a", style="italic")
ax.text(0.5, 0.3, "另: 首帧由 FLUX-ControlNet 按语义 buffer + 文本 prompt 生成, 之后才进 SVD-XT (§4.2 末)。",
        fontsize=9, color=GRAY, style="italic")

fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/049-WM-infinicube/fig-buffers.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
