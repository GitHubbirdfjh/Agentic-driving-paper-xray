# -*- coding: utf-8 -*-
"""Fig: Dolphins architecture (OpenFlamingo-based, Fig 4 of the paper).
Frozen: CLIP vision encoder, LLM blocks. Trainable: perceiver resampler,
gated cross-attention (GATED XATTN-DENSE), LoRA adapters. Added: learned
temporal position embeddings + media position embeddings for video.
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["font.family"] = ["Noto Serif CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#faf9f5"; INK = "#141413"
C_OP, C_OP_F = "#0b7a5c", "#dcf2e8"      # trainable
C_WIRE = "#6e737b"                        # frozen
C_IN, C_IN_F = "#1a6fa3", "#e2eff5"
C_OUT, C_OUT_F = "#9e5c00", "#f8ecd2"
C_ALT, C_ALT_F = "#6a55c2", "#ebe7f8"

fig, ax = plt.subplots(figsize=(12.5, 5.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.2); ax.axis("off")
ax.set_title("Dolphins 架构：冻住大头，只训“转接头”（绿色=可训练，灰色虚线=冻结）",
             fontsize=12.5, color=INK, pad=12)

def box(x, y, w, h, title, sub, frozen):
    if frozen:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                            fc="#f5f4ed", ec=C_WIRE, lw=1.5, ls=(0, (5, 3))))
        tc = C_WIRE
    else:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                            fc=C_OP_F, ec=C_OP, lw=1.9))
        tc = C_OP
    ax.text(x+w/2, y+h-0.42, title, fontsize=10, color=tc, ha="center", va="top", weight="bold")
    ax.text(x+w/2, y+h/2-0.15, sub, fontsize=8.4, color=INK, ha="center", va="top")

# video frames input
ax.add_patch(patches.FancyBboxPatch((0.3, 1.5), 1.7, 2.2, boxstyle="round,pad=0.04",
                                    fc=C_IN_F, ec=C_IN, lw=1.5))
ax.text(1.15, 3.35, "视频帧\nT frames", fontsize=10, color=C_IN, ha="center", weight="bold")
ax.text(1.15, 2.5, "+ temporal\nposition emb\n（新增，可训练）", fontsize=8, color=C_OP, ha="center")
ax.text(1.15, 1.75, "+ media\nposition emb\n（新增，可训练）", fontsize=8, color=C_OP, ha="center")

ax.annotate("", xy=(2.5, 2.6), xytext=(2.0, 2.6), arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.7))
box(2.5, 1.5, 2.2, 2.2, "CLIP 视觉编码器", "冻结\n（保留预训练知识）", True)
ax.annotate("", xy=(5.2, 2.6), xytext=(4.7, 2.6), arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.7))
box(5.2, 1.5, 2.2, 2.2, "Perceiver\nResampler", "可训练\n（视觉→固定长度 token）", False)

# LLM stack
ax.add_patch(patches.FancyBboxPatch((8.0, 0.7), 3.6, 3.9, boxstyle="round,pad=0.04",
                                    fc="#ffffff", ec=C_WIRE, lw=1.6, ls=(0, (5, 3))))
ax.text(9.8, 4.3, "LLM（LLaMA / MPT，冻结）", fontsize=10, color=C_WIRE, ha="center", weight="bold")
# layers inside
layers = [
    ("第 n 个 Block：Self-Attn（冻）", True),
    ("+ GATED XATTN-DENSE（可训练）", False),
    ("+ LoRA adapter（可训练）", False),
    ("第 1 个 Block：Self-Attn（冻）", True),
]
yy = 3.75
for t, frozen in layers:
    col = C_WIRE if frozen else C_OP
    fcol = "#f5f4ed" if frozen else C_OP_F
    ls = (0, (5, 3)) if frozen else "-"
    ax.add_patch(patches.FancyBboxPatch((8.3, yy-0.42), 3.0, 0.62, boxstyle="round,pad=0.02",
                                        fc=fcol, ec=col, lw=1.4, ls=ls))
    ax.text(9.8, yy-0.11, t, fontsize=8.2, color=col, ha="center", va="center")
    yy -= 0.72
ax.annotate("", xy=(8.0, 2.6), xytext=(7.4, 2.6), arrowprops=dict(arrowstyle="->", color=C_OP, lw=2))
ax.text(7.7, 2.95, "视觉 token\n经门控交叉注意力\n注入每一层", fontsize=8.2, color=C_OP, ha="center",
        bbox=dict(fc="#ffffff", ec=C_OP, alpha=0.95, boxstyle="round,pad=0.3"))

ax.text(11.9, 2.6, "→\n输出\n文本", fontsize=10, color=C_OUT, ha="center", weight="bold",
        bbox=dict(fc=C_OUT_F, ec=C_OUT, boxstyle="round,pad=0.3"))

ax.text(6.25, 0.28, "设计逻辑：OpenFlamingo 本来只吃单图；Dolphins 加了 temporal/media 位置嵌入来吃视频，"
        "其余大头全部冻结——\n训练只动“转接头”（resampler + gated xattn + LoRA），4 张 A100、batch 128、lr 2e-5 即可。",
        fontsize=9, color=INK, ha="center",
        bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.35"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/076-Agent-dolphins/fig-arch.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
