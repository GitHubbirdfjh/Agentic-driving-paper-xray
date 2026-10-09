# -*- coding: utf-8 -*-
"""HERMES Fig 2: BEV-to-Point Render —— 一根 LiDAR ray 从 BEV 到深度的完整路径.
BEV(w,h,c) -> 上采样 -> reshape(w,h,z,c/z) -> 3D卷积 -> volumetric feature
-> ray 采样 n 个点 -> SDF MLP -> alpha/transmittance 权重 -> 深度积分.
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

fig, ax = plt.subplots(figsize=(13, 5.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("BEV-to-Point Render: 一根 LiDAR 射线的深度是怎么算出来的 (§4.1)", fontsize=13,
             weight="bold", pad=14)

def stage(x, y, w, h, title, body, fc):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="none", alpha=0.92)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10,
            weight="bold", color="white")
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.5,
            color="white", linespacing=1.5)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=GRAY, linewidth=1.6))

# 阶段 1: BEV 上采样
stage(0.2, 1.2, 2.0, 3.6, "① BEV 特征", "F ∈ R^{w×h×c}\nw=h=200, c=256\n(论文 S1.1)", BLUE)
arrow(2.2, 3.0, 2.7, 3.0)
ax.text(2.45, 3.35, "最近邻上采样\n+ 卷积", ha="center", fontsize=8, color=GRAY)

# 阶段 2: 加高维
stage(2.7, 1.2, 2.2, 3.6, "② 补高度维", "reshape → R^{w×h×z×(c/z)}\nz=32 (S1.1)\n3D 卷积 → 体特征\nF_vol ∈ R^{w×h×z×c′}", GREEN)
arrow(4.9, 3.0, 5.4, 3.0)

# 阶段 3: ray 采样 — 画示意图
stage(5.4, 0.4, 3.4, 4.4, "③ 沿射线采样 n 个点", "", ORANGE)
# 重画标题区下面画 ray
bx, by = 5.6, 1.0
ax.plot([bx, bx + 3.0], [by + 1.6, by + 1.6], color="#9e5c00", lw=2.5)
ax.plot(bx, by + 1.6, "o", color="#141413", ms=8)
ax.text(bx - 0.15, by + 1.6, "o", fontsize=11, weight="bold", ha="right", va="center")
# 采样点
np.random.seed(0)
ds = np.array([0.35, 0.8, 1.25, 1.7, 2.05, 2.4, 2.7])
sdfs = np.array([0.9, 0.7, 0.45, 0.12, -0.25, -0.5, -0.7])  # 穿过表面: SDF 变号
for d, s in zip(ds, sdfs):
    c = "#0b7a5c" if s < 0 else "#1a6fa3"
    ax.plot(bx + d, by + 1.6, "o", color=c, ms=7)
    ax.text(bx + d, by + 1.25, f"s={s:.2f}", fontsize=7, ha="center", color=c)
ax.text(bx + 1.5, by + 2.05, "p_i = o + d_i·t_k,  三线性插值取特征 f_i", fontsize=8,
        ha="center", color="#141413")
ax.text(bx + 1.5, by + 0.75, "浅层 MLP φ_SDF(p_i, f_i) → SDF 值 s_i\n表面处 s_i 变号 (由正转负)", fontsize=8,
        ha="center", color="#141413", style="italic")
arrow(8.8, 3.0, 9.3, 3.0)

# 阶段 4: 权重与积分
stage(9.3, 1.2, 3.5, 3.6, "④ 加权积分得深度", "", PURPLE)
ax.text(11.05, 4.35, r"$\tilde d(r_k)=\sum_{i=1}^{n} w_i d_i$", fontsize=12, ha="center", color="white",
        weight="bold")
ax.text(11.05, 3.75, r"$w_i = T_i\,\alpha_i$", fontsize=11, ha="center", color="white")
ax.text(11.05, 3.05, "T_i = Π(1−α_j): 光子存活概率\n(遮挡感知: 前面被挡住, 后面权重→0)",
        fontsize=8.5, ha="center", color="white", linespacing=1.4)
ax.text(11.05, 2.15, r"$\alpha_i=\max\left(\frac{\sigma_t(s_i)-\sigma_t(s_{i+1})}{\sigma_t(s_i)},0\right)$",
        fontsize=9.5, ha="center", color="white")
ax.text(11.05, 1.55, "σ_t 为可学习温度 t 的 sigmoid\nSDF 骤降处 α_i 大 → 权重集中在表面",
        fontsize=8.5, ha="center", color="white", style="italic")

# 底部注释: 几何意义
ax.text(6.5, 0.55, "几何直觉: SDF 场把场景表示成隐式曲面; 射线穿过体素时, 在 SDF 变号处 (表面) 分配最大权重,"
        " 前方遮挡会衰减后方点的贡献 —— 这就是 NeuS 式无偏遮挡感知渲染 [48]。", fontsize=9,
        ha="center", color="#3d3d3a", style="italic",
        bbox=dict(fc="#f5f4ed", ec="none", pad=6))

fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/048-WM-hermes/fig-render.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
