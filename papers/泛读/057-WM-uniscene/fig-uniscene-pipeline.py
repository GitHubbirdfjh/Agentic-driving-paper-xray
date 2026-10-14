"""UniScene 双层级生成管线与张量形状（按论文 Fig.2 / Sec.3 与开源代码重构）。
Stage I: BEV布局 -> 语义occupancy；Stage II: occupancy -> 视频 / LiDAR。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FP = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
plt.rcParams["font.family"] = FP.get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, text, fc, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="black", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.6)

def arrow(x1, y1, x2, y2, col="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.5, color=col,
                                 connectionstyle="arc3,rad=0"))

# ===== Stage I =====
ax.text(3.6, 9.5, "Stage I：可控语义 occupancy 生成（Occupancy DiT + VAE）",
        fontsize=12, fontweight="bold", color="#1a6fa3", ha="center")
box(0.3, 7.6, 3.0, 1.3, "BEV布局序列 B\n[T,Cb,200,200]\n下采样→[Cb,h,w]", "#e2eff5")
box(3.7, 7.6, 3.0, 1.3, "噪声体\n[T,Co,h,w]\n(与BEV拼通道)", "#f8ecd2")
box(7.1, 7.6, 3.4, 1.3, "统一patchify\nZcat[T,Co+Cb,h,w]\n→ token [T,L,Ed]", "#dcf2e8")
box(10.9, 7.6, 3.4, 1.3, "OccDiT\n空间块+时间块\n2D位置+1D时间嵌入", "#ebe7f8")
arrow(3.3, 8.25, 3.7, 8.25)
arrow(6.7, 8.25, 7.1, 8.25)
arrow(10.5, 8.25, 10.9, 8.25)

box(5.8, 5.9, 3.4, 1.1, "VAE解码器 Docc\n3D卷积+3D axial注意力\nzseq[T,C,h,w]→Oseq", "#f8f4cf")
arrow(12.6, 7.6, 12.6, 7.0)
arrow(7.5, 7.0, 7.5, 7.0)
ax.annotate("", xy=(7.5, 7.0), xytext=(12.6, 7.0),
            arrowprops=dict(arrowstyle="-|>", lw=1.5))
box(5.8, 4.5, 3.4, 0.9, "语义occupancy Oseq\n[T,200,200,16]，17类\n体素0.5m，范围±50m", "#f9e3d6", fs=10)

# ===== Stage IIa: video =====
ax.text(3.6, 3.9, "Stage IIa：occupancy → 多视角视频", fontsize=12,
        fontweight="bold", color="#9e5c00", ha="center")
box(0.3, 2.2, 3.2, 1.2, "Occ2Gaussian\n每个占据体素→1个高斯\nμ=体素中心，α=1\nΣ=(0.125)²I，旋转=单位", "#ebe7f8", fs=9)
box(3.9, 2.2, 3.2, 1.2, "Tile光栅化渲染\n深度图 D + 语义图 S\n(半分辨率，逐视角)", "#f8ecd2", fs=9)
box(7.5, 2.2, 3.4, 1.2, "ControlNet式编码分支\n语义/深度各：2×ResNet\n+ 零卷积", "#dcf2e8", fs=9)
box(11.3, 2.2, 3.4, 1.2, "视频扩散UNet（SVD初始化）\n+ 文本提示（天气/时段）\n→ 多视角视频", "#e2eff5", fs=9)
arrow(3.5, 2.8, 3.9, 2.8)
arrow(7.1, 2.8, 7.5, 2.8)
arrow(10.9, 2.8, 11.3, 2.8)
arrow(7.5, 4.5, 1.9, 3.4)

# ===== Stage IIb: lidar =====
ax.text(11.5, 3.9, "Stage IIb：occupancy → LiDAR", fontsize=12,
        fontweight="bold", color="#0b7a5c", ha="center")
box(8.2, 0.25, 2.7, 1.35, "SparseUNet\n稀疏体素特征\n(只算占据体素)", "#dcf2e8", fs=9)
box(11.2, 0.25, 3.5, 1.35, "先验引导稀疏采样\n沿ray均匀采样→按占据\nPDF重采样n个点", "#f8f4cf", fs=9)
arrow(10.9, 0.9, 11.2, 0.9)
arrow(9.5, 4.5, 9.5, 1.6)
box(5.2, 0.25, 2.7, 1.35, "LiDAR头 Dlid\nSDF→NeuS权重→深度\n强度头 + ray-drop头", "#f9e3d6", fs=9)
arrow(8.2, 0.9, 7.9, 0.9)

ax.text(7.5, 9.85, "UniScene：先把难的分布拆成 occupancy，再从 occupancy 长出视频和 LiDAR（重构自论文 Fig.2 与代码）",
        ha="center", fontsize=13, fontweight="bold")
ax.text(7.5, 0.02 - 0.35 + 0.0, "", ha="center")
fig.text(0.5, 0.015,
         "核心：BEV布局太粗，直接学 P(视频,LiDAR|BEV) 太难；先学 P(occupancy|BEV)，再学 P(视频,LiDAR|occupancy)。occupancy 自带语义+几何，是天然的中间表示。",
         ha="center", fontsize=10, color="#555555",
         bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.4"))

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/057-WM-uniscene/fig-uniscene-pipeline.png",
            dpi=150, bbox_inches="tight")
print("saved fig-uniscene-pipeline.png")
