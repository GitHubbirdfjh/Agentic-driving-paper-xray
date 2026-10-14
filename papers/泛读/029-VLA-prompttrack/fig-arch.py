#!/usr/bin/env python3
"""PromptTrack 数据流与张量形状示意图.
一张图只讲一件事: prompt 的语言信息在哪个位置、以什么形状进入视觉管线,
以及三条推理分支各自输出什么.
形状依据: 论文正文公式 (1)-(6) + 官方代码
(tracker_prompt.py 的 fusion_module, spatial_temporal_prompt_reason.py,
configs/prompttrack/f3_prompttrack.py: hist_len=3, fut_len=8, 500 fixed queries).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager

_HERE = os.path.dirname(os.path.abspath(__file__))
fontManager.addfont(os.path.join(_HERE, "NotoSerifCJKsc-Regular.otf"))

plt.rcParams["font.family"] = "Noto Serif CJK SC"
plt.rcParams["axes.unicode_minus"] = False

# ---------- 配色: 蓝=输入, 绿=带参数变换, 紫=论文新增部件, 橙=输出 ----------
C_IN, C_OP, C_ALT, C_OUT, C_WIRE = "#1a6fa3", "#0b7a5c", "#6a55c2", "#9e5c00", "#6e737b"
F_IN = "#e2eff5"; F_OP = "#dcf2e8"; F_ALT = "#ebe7f8"; F_OUT = "#f8ecd2"

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, title, lines, fc, ec, fs=8.5, tf=9.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=1.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top",
            fontsize=tf, fontweight="bold", color="#141413")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.78 - i * 0.38, ln, ha="center", va="top",
                fontsize=fs, color="#3d3d3a")
    return p

def arrow(x1, y1, x2, y2, label=None, ls=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, linewidth=ls, color=C_WIRE,
                        shrinkA=2, shrinkB=4)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center",
                va="bottom", fontsize=8, color=C_WIRE)

ax.text(7.5, 9.55, "PromptTrack: 语言信息在哪里进入管线 (张量形状)", ha="center",
        fontsize=13, fontweight="bold", color="#141413")

# ---- 左列: 两路输入 ----
box(0.3, 6.6, 3.4, 2.5, "多视角图像", ["[B, 6, 3, H, W]", "6 个相机环视", "图像编码器 (VOV/ResNet)"], F_IN, C_IN)
box(0.3, 3.7, 3.4, 2.5, "语言 prompt", ["RoBERTa-base (冻结)", "[L, 768] → Linear → [L, 256]", "例: “正在超车的车”"], F_ALT, C_ALT)
ax.text(2.0, 3.35, "S: 文本 token 序列", ha="center", fontsize=8.5, color=C_ALT, fontweight="bold")

# ---- 融合: 论文式 (1) ----
box(5.0, 5.0, 4.6, 3.3, "跨模态融合  式(1)", ["Q = 图像 token  (h·w, B·6, 256)", "K,V = 文本 token (L, B·6, 256)", "+ 文本正弦位置编码", "8 头注意力 + 残差相加", "F′t: 被 prompt“染色”的特征"], F_OP, C_OP)
arrow(3.7, 7.6, 5.0, 6.9, "Ft")
arrow(3.7, 4.9, 5.0, 5.9, "S")

# ---- 解码器: 论文式 (2) ----
box(10.6, 5.0, 3.9, 3.3, "PETR 解码器  式(2)", ["Qt = 跟踪 query + 500 固定 query", "query = {特征 fti, 位置 cti}", "+ 3D 位置编码 (PETR)", "输出 QD_t [Nq, 256]", "输出 BD_t [Nq, 10]"], F_IN, C_IN)
arrow(9.6, 6.65, 10.6, 6.65, "F′t")

# ---- 三条推理分支 ----
box(1.2, 0.5, 3.6, 2.4, "过去推理  式(3)", ["跨帧注意力 (τh=3)", "+ 跨目标注意力", "MLP 回归坐标残差", "QR_t, BR_t"], F_OP, C_OP)
box(5.7, 0.5, 3.6, 2.4, "未来推理  式(4)", ["预测 M[t:t+8] [Nq, 8, 3]", "自车运动补偿", "→ 下一帧 Qtrack_t+1", "损失: L1 (权重 0.5)"], F_OP, C_OP)
box(10.2, 0.5, 3.6, 2.4, "Prompt 推理  式(5)", ["MLP: 256→256→256→1", "对每个 track 二分类", "Pt: 是/否 prompt 所指", "损失: Focal (权重 1.0)"], F_ALT, C_ALT)

arrow(10.7, 5.3, 4.6, 2.9)    # 解码器 -> 过去推理
arrow(11.8, 5.0, 7.5, 2.9)    # 解码器 -> 未来推理
arrow(13.5, 5.0, 12.0, 2.9)   # 解码器 -> Prompt 推理
ax.text(7.5, 0.12, "三条分支都从解码器输出出发; prompt 推理分支本身不做跨模态注意力, 语言交互只发生在式(1)的图像特征层",
        ha="center", fontsize=8.5, color=C_WIRE)

# ---- 图例 ----
for i, (lab, fc, ec) in enumerate([("输入", F_IN, C_IN), ("带参数变换", F_OP, C_OP),
                                    ("论文新增", F_ALT, C_ALT)]):
    x = 5.4 + i * 2.2
    ax.add_patch(FancyBboxPatch((x, 9.0), 0.5, 0.3, boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(x + 0.62, 9.15, lab, va="center", fontsize=8.5, color="#3d3d3a")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/029-VLA-prompttrack/fig-arch.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
