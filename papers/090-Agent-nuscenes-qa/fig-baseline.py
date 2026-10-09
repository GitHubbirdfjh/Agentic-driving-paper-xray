# -*- coding: utf-8 -*-
"""NuScenes-QA baseline 框架（论文 Fig.4 重绘）：双路 BEV 特征 → 旋转框裁剪池化 → MCAN。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_IMG = "#e3f2fd"; C_PC = "#fff3e0"; C_BEV = "#e8f5e9"; C_QA = "#f3e5f5"; C_Q = "#ede7f6"

fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.4); ax.axis("off")

def box(x, y, w, h, text, fc, fs=9.5):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                 fc=fc, ec="#555", lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, linespacing=1.55)

def arrow(x1, y1, x2, y2, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#455a64", lw=lw))

def lab(x, y, s, fs=8.5):
    ax.text(x, y, s, fontsize=fs, color="#b34a06", ha="center", style="italic", weight="bold")

ax.text(6.75, 8.05, "NuScenes-QA baseline：检测 backbone 提 BEV 特征 → 旋转框对象池化 → MCAN 答题（论文 Fig.4）",
        fontsize=12.5, weight="bold", ha="center")

# ===== 图像分支 =====
ax.text(1.45, 7.55, "图像分支（BEVDet 式）", fontsize=10.5, weight="bold", ha="center")
box(0.2, 6.1, 2.5, 1.15, "环视多视图图像\n6 视角", C_IMG)
box(0.2, 4.65, 2.5, 1.15, "ResNet + FPN\n多尺度特征", C_IMG)
box(0.2, 3.2, 2.5, 1.15, "深度估计 + 视角变换\n(LSS 式)，Z 轴池化", C_IMG)
box(0.2, 1.75, 2.5, 1.15, "M_I ∈ R^(H×W×d_m)\n图像 BEV 特征图", C_BEV)
for y1, y2 in [(6.1, 5.8), (4.65, 4.35), (3.2, 2.9)]:
    arrow(1.45, y1, 1.45, y2)

# ===== 点云分支 =====
ax.text(4.35, 7.55, "点云分支（CenterPoint 式）", fontsize=10.5, weight="bold", ha="center")
box(3.1, 6.1, 2.5, 1.15, "LiDAR 点云", C_PC)
box(3.1, 4.65, 2.5, 1.15, "体素化\n二值体素网格", C_PC)
box(3.1, 3.2, 2.5, 1.15, "3D 稀疏卷积\nZ 轴池化", C_PC)
box(3.1, 1.75, 2.5, 1.15, "M_P ∈ R^(H×W×d_m)\n点云 BEV 特征图", C_BEV)
for y1, y2 in [(6.1, 5.8), (4.65, 4.35), (3.2, 2.9)]:
    arrow(4.35, y1, 4.35, y2)

# ===== 融合（总线汇入） =====
box(6.5, 1.75, 2.5, 1.15, "融合 BEV\nM ∈ R^(H×W×d_m)", C_BEV)
ax.plot([1.45, 1.45], [1.75, 1.2], color="#455a64", lw=1.5)
ax.plot([4.35, 4.35], [1.75, 1.2], color="#455a64", lw=1.5)
ax.plot([1.45, 7.75], [1.2, 1.2], color="#455a64", lw=1.5)
arrow(7.75, 1.2, 7.75, 1.75, lw=1.5)
ax.text(4.6, 0.98, "聚合（相加/拼接）", fontsize=8.5, ha="center", color="#333")

# ===== 旋转框对象池化 =====
box(6.5, 4.65, 2.5, 2.6, "旋转框对象池化\n（式6–7）\n3D 框 B=[x,y,z,尺寸,φ]\n→ 投影到 BEV 像素\n→ 按 φ 旋转四顶点\n→ 叉积判定像素归属\n→ 框内 mean pooling", C_Q, 8.8)
lab(7.75, 7.42, "式(6): x_m = (x − R_pc)/(F_v·F_o)")
arrow(7.75, 4.65, 7.75, 4.35)
box(6.5, 3.2, 2.5, 1.15, "O ∈ R^(N×d_m)\n对象嵌入", C_Q)
ax.text(9.35, 5.9, "3D 检测框来自\n预训练检测器\n（离线抽取）", fontsize=8.5, ha="left", color="#333")

# ===== 问题分支 =====
box(10.7, 6.1, 2.6, 1.15, "问题 Q\nn_q 个词", C_QA)
box(10.7, 4.65, 2.6, 1.15, "GloVe + 单层 biLSTM\nw_i = [h→;h←] ∈ R^d", C_QA)
arrow(12.0, 6.1, 12.0, 5.8)

# ===== QA 头 =====
box(10.7, 2.6, 2.6, 1.9, "MCAN 答题头\n6 层 encoder-decoder\n自注意力 + 交叉注意力\n→ MLP → 30 类答案\nargmax P(a|S,Q)", C_QA, 9)
arrow(9.0, 3.775, 10.7, 3.775)
arrow(12.0, 4.65, 12.0, 4.5)

ax.text(0.2, 0.28, "关键实现细节：① 检测 backbone 预训练好、离线抽对象特征，只训练答题头；② d_m=512；③ Adam lr=1e-4，每 2 epoch 减半，batch 256，2×RTX3090；\n"
        "④ 答案空间只有 30 类（0–10 计数、yes/no、10 类物体、8 类状态），是闭集分类不是开放生成；⑤ 旋转框裁剪是因为 BEV 上 3D 框是带朝向的矩形，不能直接 RoI Pool。",
        fontsize=8.8, ha="left", color="#333", linespacing=1.7,
        bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.35"))

plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/090-Agent-nuscenes-qa/fig-baseline.png", dpi=150)
print("saved fig-baseline.png")
