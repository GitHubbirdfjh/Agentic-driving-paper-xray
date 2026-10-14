# -*- coding: utf-8 -*-
"""非对称 aIoU vs 标准 IoU：DriveVLM-Dual 式 (Sec 3.5) 的匹配准则示意。

场景: VLM 给出的关键物体框 b^c(大, 粗定位) 与 3D 检测器投影回来的框 b^{2D}(小, 精确)。
标准 IoU 会因为大框"稀释"分母而偏小; aIoU 只用小框面积做分母, 回答的是
"3D 框有多大比例落进了 VLM 框里"。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 具体数值例子: VLM 框 b^c = (x1,y1,x2,y2), 3D 投影框 b^{2D}
bc = np.array([100, 100, 340, 300])     # VLM 关键物体框(粗)
b2d = np.array([180, 160, 280, 260])    # 3D 检测器投影框(精确, 更小)

def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])

def inter(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

Ai, Ac, A2 = inter(bc, b2d), area(bc), area(b2d)
iou = Ai / (Ac + A2 - Ai)
aiou = Ai / A2  # 论文定义: S_{交集} / S_{b^{2D}}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

for ax, title in zip(axes, ["标准 IoU", "论文的 aIoU (非对称)"]):
    ax.add_patch(Rectangle((bc[0], bc[1]), bc[2] - bc[0], bc[3] - bc[1],
                           facecolor="#b5d8ec", edgecolor="#1a6fa3", lw=2,
                           label="VLM 框 $b^c$ (粗定位)"))
    ax.add_patch(Rectangle((b2d[0], b2d[1]), b2d[2] - b2d[0], b2d[3] - b2d[1],
                           facecolor="none", edgecolor="#9e5c00", lw=2.5, ls="--",
                           label="3D 投影框 $b^{2D}$ (精确)"))
    ix1, iy1 = max(bc[0], b2d[0]), max(bc[1], b2d[1])
    ix2, iy2 = min(bc[2], b2d[2]), min(bc[3], b2d[3])
    ax.add_patch(Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1,
                           facecolor="#f0cf8a", edgecolor="#9e5c00", lw=1.5, alpha=0.85,
                           label="交集"))
    ax.set_xlim(60, 380); ax.set_ylim(60, 340)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc="upper left")

axes[0].text(0.5, -0.13,
             f"IoU = 交集 / 并集 = {Ai} / ({Ac}+{A2}-{Ai}) = {iou:.3f}",
             transform=axes[0].transAxes, ha="center", fontsize=11,
             bbox=dict(boxstyle="round", fc="#f8ecd2", ec="#9e5c00"))
axes[1].text(0.5, -0.13,
             f"aIoU = 交集 / S(b^{{2D}}) = {Ai} / {A2} = {aiou:.3f}",
             transform=axes[1].transAxes, ha="center", fontsize=11,
             bbox=dict(boxstyle="round", fc="#f8ecd2", ec="#9e5c00"))

fig.suptitle("DriveVLM-Dual 的匹配准则: 同一对框, 两种算法给出完全不同的匹配结论",
             fontsize=14, y=1.02)
fig.text(0.5, -0.02,
         f"本例中交集={Ai}, VLM 框面积={Ac}, 3D 框面积={A2}。若阈值 τ=0.5: "
         f"IoU={iou:.3f}<0.5 判为未匹配, aIoU={aiou:.3f}>0.5 判为匹配 —— "
         "aIoU 容忍 VLM 框画得大, 只关心 3D 框是否落进去。",
         ha="center", fontsize=10.5, color="#3d3d3a",
         bbox=dict(boxstyle="round", fc="#f5f4ed", ec="#deddd8"))
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/013-VLA-drivevlm/fig-aiou.png",
            dpi=150, bbox_inches="tight")
print("saved fig-aiou.png")
