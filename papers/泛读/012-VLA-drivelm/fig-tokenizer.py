#!/usr/bin/env python3
"""轨迹 tokenization 小实例：连续 (x,y) -> 256 区间 -> token 序列 [SOT] ... [EOT]，再解码回去。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

IN   = "#1a6fa3"
OUT  = "#9e5c00"
OP   = "#0b7a5c"
WIRE = "#6e737b"
ALT  = "#6a55c2"
BG   = "#faf9f5"

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor(BG)

# ---- 左：连续轨迹 + x 轴 256 区间 ----
ax = axes[0]
traj = np.array([[0.0, 0.0], [3.25, 0.42], [6.90, 1.05], [10.10, 2.20]])
ax.plot(traj[:, 0], traj[:, 1], color=IN, linewidth=2.2, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=IN, markeredgewidth=2, zorder=3)
for i, (x, y) in enumerate(traj[1:], start=1):
    ax.annotate(f"(x{i}, y{i})", (x, y), xytext=(8, 10), textcoords="offset points",
                fontsize=10, color=IN)
ax.set_xlim(-2, 14)
ax.set_ylim(-1.5, 4.5)
ax.set_xlabel("x (m, BEV 前向)", fontsize=10)
ax.set_ylabel("y (m, BEV 侧向)", fontsize=10)
ax.set_title("1. 连续轨迹 M={(x0,y0),...,(xN,yN)}", fontsize=11, weight="bold", color="#141413")

# x 轴示意 256 个区间（只画出刻度示意）
lo, hi, nbins = -20.0, 20.0, 256
w = (hi - lo) / nbins
x1 = 3.25
b = int((x1 - lo) / w)
ax.axvspan(lo + b * w, lo + (b + 1) * w, color=ALT, alpha=0.25, zorder=1)
ax.text(lo + b * w + w / 2, -1.1, f"bin {b}\n宽 {w:.4f} m", ha="center", fontsize=9, color=ALT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ALT, alpha=0.9))
ax.text(7, 3.6, "x=3.25: (3.25-(-20))/0.15625 = 148.8\n向下取整 -> bin 148", fontsize=9,
        color="#141413", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=WIRE))

# ---- 中：bin -> token 映射 ----
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("2. 每个区间对应一个 token", fontsize=11, weight="bold", color="#141413")
steps = [
    ("bin 148 (x 轴)", "token <num_148>", IN),
    ("bin  57 (y 轴)", "token <num_57>", IN),
    ("...", "...", WIRE),
    ("bin 201 (x 轴)", "token <num_201>", IN),
]
for i, (b, t, c) in enumerate(steps):
    y = 8.2 - i * 1.9
    ax.text(2.2, y, b, fontsize=10.5, color=c, ha="left", va="center")
    arr = FancyArrowPatch((4.6, y), (5.6, y), arrowstyle="-|>", mutation_scale=12, color=OP, linewidth=1.4)
    ax.add_patch(arr)
    ax.text(7.8, y, t, fontsize=10.5, color=OUT, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8ecd2", edgecolor=OUT))
ax.text(5, 0.7, "从 BLIP-2 词表里挑 256 个数字 token\n（跳过个位数），重定义为轨迹 token", fontsize=9,
        color=WIRE, ha="center", style="italic")

# ---- 右：token 序列 + 解码 ----
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("3. LLM 自回归输出轨迹句子", fontsize=11, weight="bold", color="#141413")
seq = ["SOT", "tx1", "ty1", "tx2", "ty2", "tx3", "ty3", "EOT"]
for i, tok in enumerate(seq):
    x = 1.1 + i * 1.08
    col = ALT if tok in ("SOT", "EOT") else OUT
    ax.text(x, 7.2, tok, fontsize=9.5, color=col, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=col, linewidth=1.4))
    if i < len(seq) - 1:
        arr = FancyArrowPatch((x + 0.42, 7.2), (x + 0.66, 7.2), arrowstyle="-|>",
                              mutation_scale=10, color=WIRE, linewidth=1.2)
        ax.add_patch(arr)
ax.text(5, 5.6, "next-token prediction，标准语言模型损失", fontsize=9.5, color=WIRE, ha="center", style="italic")
ax.text(5, 4.2, "解码：token -> 区间中心 -> (x, y)", fontsize=10.5, color=OP, ha="center", weight="bold")
ax.text(5, 3.3, "bin 148 -> 中心 x = -20 + (148+0.5)*0.15625 = 3.203 m", fontsize=9.5,
        color="#141413", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=OP))
ax.text(5, 2.2, "量化误差约半个区间宽（~8 cm），\n换来的是 LLM 能直接\"说\"轨迹", fontsize=9.5,
        color=WIRE, ha="center", style="italic")
ax.text(5, 0.8, "示意数字（论文未公布区间边界，按均匀划分推算）", fontsize=8.5,
        color=WIRE, ha="center")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/012-VLA-drivelm/fig-tokenizer.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
