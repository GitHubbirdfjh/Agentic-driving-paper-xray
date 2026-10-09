# -*- coding: utf-8 -*-
"""供需感知的核心洞见：Ego 视角的"低前景置信"有三种，但只有两种需要协作。
左：BEV 示意 —— Area 1（近处空旷路面，看得清但无前景）、Area 2（被遮挡）、Area 3（远处稀疏）。
右：掩码逻辑 —— D（Ego 点云稀疏处）∩ S（协作者的前景置信）= 真正要传的区域；
Where2comm 只用 S，会把协作者在 Ego 近处看到的前景也传一遍，浪费带宽。
示意几何，忠实于论文图 1b 与 3.3 节的掩码定义。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge, FancyBboxPatch
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# ============ 左图：三种低置信区域 ============
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a) Ego 视角下\"低前景置信\"的三种成因", fontsize=12, pad=10)

# 路面底色
ax.add_patch(Rectangle((0, 0), 10, 10, fc="#eef1f3", ec="none"))
# 车道线
for y in [3.2, 6.8]:
    for x in np.arange(0.3, 10, 1.2):
        ax.add_patch(Rectangle((x, y), 0.6, 0.08, fc="white", ec="none"))
# Ego 车
ax.add_patch(FancyBboxPatch((1.2, 4.6), 1.4, 0.9, boxstyle="round,pad=0.05",
                            fc="#1a6fa3", ec="#0d3d5c", linewidth=1.5))
ax.text(1.9, 5.05, "Ego", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
# 遮挡物：一辆停着的大货车
ax.add_patch(FancyBboxPatch((4.2, 5.6), 2.2, 1.1, boxstyle="round,pad=0.05",
                            fc="#5a626b", ec="#333a40", linewidth=1.5))
ax.text(5.3, 6.15, "遮挡物", ha="center", va="center", fontsize=9, color="white")
# 遮挡阴影（Area 2）
shadow = Wedge((1.9, 5.05), 7.5, -18, 18, fc="#b34a06", alpha=0.22, ec="none")
ax.add_patch(shadow)
ax.text(7.6, 5.05, "Area 2\n被遮挡\n低置信→需协作", ha="center", va="center", fontsize=9.5,
        color="#7a2f04", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f9e3d6", ec="#b34a06"))
# Area 3：远处稀疏
ax.add_patch(Rectangle((8.3, 1.2), 1.5, 2.2, fc="none", ec="#9e5c00",
                       linewidth=2, linestyle="--"))
ax.text(9.05, 2.3, "Area 3\n远处稀疏\n低置信→需协作", ha="center", va="center", fontsize=9.5,
        color="#7a5200", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8ecd2", ec="#9e5c00"))
# Area 1：近处空旷
ax.add_patch(Rectangle((2.8, 7.4), 2.4, 1.6, fc="none", ec="#0b7a5c",
                       linewidth=2, linestyle="--"))
ax.text(4.0, 8.2, "Area 1\n近处空旷：点云密、无前景\n低置信但→不需协作", ha="center", va="center", fontsize=9.5,
        color="#0b5345", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#dcf2e8", ec="#0b7a5c"))
# 稀疏点示意：远处几个点
rng = np.random.default_rng(0)
for (cx, cy, n) in [(9.05, 2.3, 14), (7.6, 5.05, 6)]:
    xs = cx + rng.normal(0, 0.55, n)
    ys = cy + rng.normal(0, 0.55, n)
    ax.scatter(xs, ys, s=14, c="#37474f", alpha=0.7, zorder=5)

# ============ 右图：掩码交集逻辑 ============
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(b) 掩码逻辑：只传 D ∩ S，Where2comm 只用了 S", fontsize=12, pad=10)

def venn_panel(x0, title, d_rects, s_rects, note, note_c):
    ax.text(x0 + 1.6, 9.3, title, ha="center", fontsize=10.5, fontweight="bold")
    ax.add_patch(Rectangle((x0, 5.6), 3.2, 3.2, fc="#f5f4ed", ec="#6e737b", linewidth=1))
    # D：Ego 需求（点云稀疏处）—— 蓝色斜线区
    for (rx, ry, rw, rh) in d_rects:
        ax.add_patch(Rectangle((x0 + rx, 5.6 + ry), rw, rh, fc="#1a6fa3", alpha=0.35,
                               ec="#1a6fa3", hatch="///", linewidth=0))
    # S：协作者供给（前景置信）—— 紫色区
    for (rx, ry, rw, rh) in s_rects:
        ax.add_patch(Rectangle((x0 + rx, 5.6 + ry), rw, rh, fc="#6a55c2", alpha=0.45,
                               ec="#6a55c2", linewidth=1.2))
    ax.text(x0 + 1.6, 4.9, note, ha="center", fontsize=9, color=note_c,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=note_c))

# D：远处 + 遮挡区稀疏（右半 + 中间一块）
venn_panel(0.4, "D：Ego 的需求掩码",
           [(1.6, 0.4, 1.4, 2.4), (0.9, 1.2, 0.8, 1.0)], [],
           "点云密度 < εa 的 pillar", "#1a6fa3")
# S：协作者看到的前景（含 Ego 近处）
venn_panel(3.9, "S：协作者的供给掩码",
           [], [(0.5, 1.9, 0.7, 0.7), (1.9, 0.7, 0.8, 0.8)],
           "前景置信 > εc 的 pillar", "#6a55c2")
# M = D ∩ S：只有远处那块
venn_panel(7.4, "M = D ⊙ S：实际传输",
           [(1.6, 0.4, 1.4, 2.4), (0.9, 1.2, 0.8, 1.0)],
           [(1.9, 0.7, 0.8, 0.8)],
           "近处那块被 D 滤掉\n→ 省约 10% 带宽", "#2e7d32")
ax.text(5.0, 2.6, "Where2comm 传 S 的全部（含近处）→ 浪费\nCoSDH 传 D∩S → 近处 Ego 自己看得清，不传",
        ha="center", fontsize=10, color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", fc="#faf9f5", ec="#c9c6bb"))
# 图例
ax.text(0.4, 1.2, "■", fontsize=13, color="#1a6fa3", alpha=0.7, va="center")
ax.text(0.75, 1.2, "= D（需求）", fontsize=9.5, va="center")
ax.text(3.2, 1.2, "■", fontsize=13, color="#6a55c2", alpha=0.8, va="center")
ax.text(3.55, 1.2, "= S（供给）", fontsize=9.5, va="center")

fig.suptitle("供需感知的真正含义：把\"Ego 看不清\"和\"协作者看得见\"取交集", fontsize=13,
             fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/084-Agent-cosdh/fig-supply-demand.png",
            dpi=150, bbox_inches="tight")
print("saved fig-supply-demand.png")
