"""统一 codebook 示意：语言 token 和 action token 住在同一张表里，
同一个自回归模型处理。对比旧架构（两个头）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import fontManager

fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

IN = "#1a6fa3"
OUT = "#9e5c00"
OP = "#0b7a5c"
ALT = "#6a55c2"
WIRE = "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")

# ---------- 左：旧架构，两个头 ----------
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(a) 旧 VLA：语言和动作各走各的头", fontsize=12)
ax.text(5, 9.2, "vision + instruction", ha="center", color=IN, fontsize=11)
ax.add_patch(mpatches.FancyBboxPatch((3.2, 6.8), 3.6, 1.4, boxstyle="round,pad=0.1",
                                    fc=IN, ec=IN, alpha=0.18))
ax.text(5, 7.5, "VLM backbone", ha="center", color=IN, fontsize=11)
# 语言头
ax.annotate("", xy=(2.5, 5.2), xytext=(4.2, 6.8),
            arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.4))
ax.add_patch(mpatches.FancyBboxPatch((0.6, 3.8), 3.8, 1.4, boxstyle="round,pad=0.1",
                                    fc=OUT, ec=OUT, alpha=0.18))
ax.text(2.5, 4.5, "text head\n\"向左变道\"", ha="center", color=OUT, fontsize=10)
# 动作头
ax.annotate("", xy=(7.5, 5.2), xytext=(5.8, 6.8),
            arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.4))
ax.add_patch(mpatches.FancyBboxPatch((5.6, 3.8), 3.8, 1.4, boxstyle="round,pad=0.1",
                                    fc="#b34a06", ec="#b34a06", alpha=0.18))
ax.text(7.5, 4.5, "action head (MLP)\n直行轨迹", ha="center", color="#b34a06", fontsize=10)
ax.text(5, 2.6, "嘴上说对了，轮子做错——两张皮", ha="center",
        color="#b34a06", fontsize=11, style="italic")
ax.text(5, 1.6, "对齐靠“希望”", ha="center", color=WIRE, fontsize=10)

# ---------- 右：LinkVLA，一个 codebook ----------
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(b) LinkVLA：轨迹也是词，共用一本词典", fontsize=12)
# codebook 表
ax.add_patch(mpatches.FancyBboxPatch((2.5, 5.6), 5, 3.2, boxstyle="round,pad=0.15",
                                    fc=ALT, ec=ALT, alpha=0.12))
ax.text(5, 8.3, "unified codebook C", ha="center", color=ALT, fontsize=11,
        weight="bold")
ax.text(5, 7.5, "K = K_text + 5656", ha="center", color=ALT, fontsize=10)
ax.text(3.6, 6.7, "\"变道\" \"左\"", ha="center", color=IN, fontsize=10)
ax.text(6.6, 6.7, "a_3842 a_1024", ha="center", color=OUT, fontsize=10)
ax.text(3.6, 6.0, "语言词", ha="center", color=IN, fontsize=9)
ax.text(6.6, 6.0, "动作词", ha="center", color=OUT, fontsize=9)
ax.annotate("", xy=(5, 5.6), xytext=(5, 6.4),
            arrowprops=dict(arrowstyle="<->", color=ALT, lw=1.4))
# 同一个模型
ax.add_patch(mpatches.FancyBboxPatch((3.2, 3.2), 3.6, 1.6, boxstyle="round,pad=0.1",
                                    fc=OP, ec=OP, alpha=0.18))
ax.text(5, 4.0, "single VLM\nnext-token prediction", ha="center", color=OP,
        fontsize=10)
ax.annotate("", xy=(5, 4.8), xytext=(5, 5.6),
            arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.4))
ax.text(5, 2.4, "\"向左变道\" ↔ 一串向左偏的轨迹词", ha="center", color=OP,
        fontsize=11)
ax.text(5, 1.5, "同一个 softmax 里竞争，同一张表里表示", ha="center",
        color=WIRE, fontsize=10)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/088-Agent-linkvla/fig-unified-codebook.png",
            dpi=150)
print("saved fig-unified-codebook.png")
