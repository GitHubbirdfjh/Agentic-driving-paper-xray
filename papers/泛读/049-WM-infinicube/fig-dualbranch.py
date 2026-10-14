# -*- coding: utf-8 -*-
"""InfiniCube Fig 3: 双分支前馈重建 —— voxel branch 管静态背景, pixel branch 管
mid-ground 与动态物体, 推理时合成 (§4.3).
voxel branch: 每体素 4 个 Gaussian × 14 通道; pixel branch: 每像素 2 个 Gaussian × 12 通道.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BLUE, GREEN, ORANGE, PURPLE = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6a55c2"
GRAY = "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
fig.patch.set_facecolor("white")
fig.suptitle("双分支前馈重建: 静态走体素, 动态与中景走像素, 推理时合成 (§4.3)", fontsize=13,
             weight="bold")

def panel(ax, title, branch_name, rows, color):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, weight="bold", color=color, pad=10)
    y = 9.0
    for head, body in rows:
        b = FancyBboxPatch((0.4, y - 1.55), 9.2, 1.75, boxstyle="round,pad=0.02",
                           fc=color, ec="none", alpha=0.88)
        ax.add_patch(b)
        ax.text(1.0, y - 0.35, head, fontsize=10, weight="bold", color="white", va="center")
        ax.text(1.0, y - 1.0, body, fontsize=8.6, color="white", va="center", linespacing=1.45)
        y -= 2.05
    return y

# ---------- voxel branch ----------
y_end = panel(axes[0], "Voxel Branch —— 静态背景", "voxel", [
    ("输入", "体素世界 + 带位姿视频帧\n动态物体从图像特征中 mask 掉, 只用静态背景体素"),
    ("网络", "图像特征反投影到体素 (max-pool 聚合)\n→ 3D 稀疏卷积 U-Net (base 64, ×[1,2,4])"),
    ("输出", "每体素 4 个 Gaussian × 14 通道\n= RGB(3)+旋转(4)+尺度(3)+不透明度(1)+相对位置(3)\n→ 换算成绝对 3D 中心"),
], BLUE)
axes[0].text(5, y_end + 0.35, "几何分布好 (有体素脚手架), 但体素覆盖不到的中景无能为力",
             ha="center", fontsize=9, color=BLUE, style="italic",
             bbox=dict(fc="#e2eff5", ec="none", pad=5))

# ---------- pixel branch ----------
y_end = panel(axes[1], "Pixel Branch —— 中景 + 动态物体", "pixel", [
    ("输入 (36ch)", "RGB 图像 (3) + 随机 mask 的体素深度 Z~(1)\n+ Depth Anything V2 特征 FDAV2 (32)"),
    ("自监督训练", "训练时随机 mask 掉 16×16 patch (p=0.5)\n模拟\"体素覆盖不到\"的中景; 用完整 Z 监督预测深度\n→ 推理时中景深度合理"),
    ("输出", "每像素 2 个 Gaussian × 12 通道\n= RGB(3)+旋转(4)+尺度(3)+不透明度(1)+深度(1)\n深度→距离→ 3D 中心 (znear=0.5, zfar=300)"),
], GREEN)
axes[1].text(5, y_end + 0.35, "捕捉中景内容与逐帧运动, 但几何不如体素分支稳",
             ha="center", fontsize=9, color=GREEN, style="italic",
             bbox=dict(fc="#dcf2e8", ec="none", pad=5))

fig.text(0.5, 0.02,
         "推理合成: voxel branch 只作用于静态部分; pixel branch 每 4 帧跑一次, 只保留中景像素与动态物体的 Gaussian;\n"
         "动态车辆用语义 buffer 的分割抠出各自的 Gaussian, 按位姿变换聚合进包围盒 —— 改轨迹即改运动, 完全可控。天空用 STORM 式隐式表示 (192 维 sky 向量 + AdaLN 调制 MLP)。",
         ha="center", fontsize=9, color="#3d3d3a", style="italic",
         bbox=dict(fc="#f5f4ed", ec="none", pad=6))

fig.tight_layout(rect=[0, 0.06, 1, 0.93])
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/049-WM-infinicube/fig-dualbranch.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
