"""GEM 三路条件注入方式与张量形状（按论文 Fig.2 + 代码 video_model.py / attention.py / configs 重构）。
关键发现：论文说轨迹用 "LoRA"，代码里 add_lora=False，实际是零初始化全秩线性层加到 K/V 上。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FP = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
plt.rcParams["font.family"] = FP.get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 9.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9.7)
ax.axis("off")

def box(x, y, w, h, text, fc, ec="black", fs=9.5, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.5)

def arrow(x1, y1, x2, y2, col="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.4, color=col, linestyle=ls))

# ---------- 输入行 ----------
box(0.2, 7.3, 2.4, 1.2, "参考帧 x0\n[B,3,576,1024]\nVAE编码→[B,4,72,128]", "#e2eff5")
box(3.0, 7.3, 2.6, 1.2, "带噪隐变量 x\nRGB+Depth+条件\n[B,25,12,72,128]", "#f8ecd2")
box(6.0, 7.3, 2.6, 1.2, "轨迹 c_traj\n8个数(4个未来2D点)\nFourier→[B,1,2432]", "#dcf2e8")
box(9.0, 7.3, 1.9, 1.2, "DINO c_dino\n稀疏token\n768维,≤16/帧", "#ebe7f8")
box(11.1, 7.3, 1.7, 1.2, "骨架 c_pose\n渲染图\nPoseNet", "#f8f4cf")

# ---------- 处理行 ----------
box(0.2, 5.3, 2.4, 1.2, "concat条件帧\n(与x0拼通道)", "#e2eff5")
box(6.0, 5.3, 2.6, 1.4,
    "零初始化 Linear(2432→inner_dim)\nK += A_k(c), V += A_v(c)\n每个block的cross-attn\n(代码add_lora=False,非真LoRA)",
    "#dcf2e8", fs=8.5)
box(9.0, 5.3, 1.9, 1.2, "ObjectNet\n(UNet编码器结构)\n+ identity嵌入", "#ebe7f8")
box(11.1, 5.3, 1.7, 1.2, "1×1卷积\n320→model_ch\n零初始化", "#f8f4cf")

arrow(1.4, 7.3, 1.4, 6.5)
arrow(7.3, 7.3, 7.3, 6.7)
arrow(9.95, 7.3, 9.95, 6.5)
arrow(11.95, 7.3, 11.95, 6.5)

# ---------- UNet 主体 ----------
box(2.6, 2.6, 7.6, 2.0,
    "SVD UNet 去噪器 Dθ（全参数微调）\n"
    "input blocks → middle → output blocks\n"
    "轨迹：cross-attn 的 K/V 上做加法（全局条件）\n"
    "DINO/Pose：加到 input blocks 各层输出上（空间条件）",
    "#ffffff", fs=10)

arrow(1.4, 5.3, 2.6, 4.0)
arrow(7.3, 5.3, 7.3, 4.6)
arrow(9.95, 5.3, 9.0, 4.2)
arrow(11.95, 5.3, 10.0, 4.0)
arrow(4.3, 6.2, 4.3, 4.6)  # noisy latents -> unet

box(3.0, 5.9, 2.6, 0.7, "depth_in 零卷积\n(第0个block后加depth分支)", "#f9e3d6", fs=8.5)

# ---------- 输出头 ----------
box(2.6, 1.0, 3.6, 1.0, "P_image 输出投影\nRGB噪声 [B,25,4,72,128]", "#f8ecd2")
box(6.6, 1.0, 3.6, 1.0, "P_depth 输出投影\nDepth噪声 [B,25,4,72,128]", "#f9e3d6")
arrow(4.4, 2.6, 4.4, 2.0)
arrow(8.4, 2.6, 8.4, 2.0)

box(2.6, 0.05, 7.6, 0.7, "VAE解码 → RGB视频 + Depth视频（同一采样器，天然对齐）",
    "#ffffff", fs=9.5)

ax.text(6.5, 9.35, "GEM 条件注入：三路信号，三种加法（重构自论文 Fig.2 与开源代码）",
        ha="center", fontsize=13, fontweight="bold")
ax.text(6.5, 8.95,
        "共同点：所有新模块都用零初始化（zero_init / zero_module），训练初期恒等，逐步接管",
        ha="center", fontsize=10, color="#555555")

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/056-WM-gem/fig-conditioning.png",
            dpi=150, bbox_inches="tight")
print("saved fig-conditioning.png")
