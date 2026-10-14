# -*- coding: utf-8 -*-
"""LMDrive 每帧的 token 预算：406 个视觉 token 经 Q-Former 压到 4 个。
一只讲一件事：为什么 Q-Former 在这里不是可选配件 —— Tmax=40 帧的序列
不压缩根本进不了 LLM 的上下文。
数字来自论文 4.2 节与附录 A（400 BEV + 5 路点 + 1 红绿灯 = 406/帧，M=4）。
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
_font = [f for f in font_manager.fontManager.ttflist if f.name == "Noto Sans CJK JP"]
assert _font, "no CJK font"
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyBboxPatch

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; ALT = "#6a55c2"; WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 5.0); ax.axis("off")
ax.text(5.5, 4.62, "LMDrive 的 token 预算：一帧 406 个视觉 token，不压缩就没有闭环",
        ha="center", fontsize=13, weight="bold", color="#141413")

def box(x, y, w, h, label, fill, edge, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                facecolor=fill, edgecolor=edge, linewidth=1.7))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=edge, weight="bold", linespacing=1.5)

# 一帧的 406 个视觉 token
box(0.3, 2.9, 2.2, 1.0, "BEV token\n400 个", "#e2eff5", IN)
box(2.7, 2.9, 1.5, 1.0, "路点 token\n5 个", "#e2eff5", IN)
box(4.4, 2.9, 1.5, 1.0, "红绿灯 token\n1 个", "#e2eff5", IN)
ax.text(3.1, 2.62, "每帧 406 个视觉 token（论文 4.2 节、附录 A）", ha="center",
        fontsize=10.5, color=IN)

# Q-Former 压缩
ax.annotate("", xy=(7.35, 3.4), xytext=(6.3, 3.4),
            arrowprops=dict(arrowstyle="->", color=OP, lw=2.4))
ax.text(6.82, 3.62, "Q-Former\nM=4 个可学习查询", ha="center", fontsize=10.5,
        color=OP, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#dcf2e8", edgecolor=OP))
box(7.6, 2.9, 1.5, 1.0, "压缩后\n4 个/帧", "#dcf2e8", OP)

# 历史帧堆叠
ax.annotate("", xy=(2.6, 1.7), xytext=(8.35, 2.85),
            arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.6, ls="--",
                            connectionstyle="angle3,angleA=0,angleB=90"))
box(0.6, 0.55, 4.4, 1.05, "Tmax=40 帧历史 → 40×4 = 160 个视觉 token\n+ 指令文本 token → 一起进冻结的 LLaMA",
    "#f5f4ed", WIRE, fs=10.5)
ax.text(7.9, 1.35, "对照：不压缩则为\n40×406 = 16240 个 token\n远超上下文窗口",
        ha="center", fontsize=10.5, color="#b34a06",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9e3d6", edgecolor="#b34a06"))

ax.text(0.3, 0.12, "图：Q-Former 把每帧 406 token 压到 4 个 —— 没有这一步，40 帧历史根本喂不进 LLM",
        fontsize=10, color="#73726c")
plt.tight_layout(pad=0.3)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/074-Agent-lmdrive/fig-token-budget.png",
            dpi=150, bbox_inches="tight")
print("saved fig-token-budget.png")
