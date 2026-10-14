# -*- coding: utf-8 -*-
"""HERMES Fig 1: 分离式统一 vs HERMES 的信息流向对比.
左: 分离式统一 (Fig.1c) —— 共享 BEV, 理解与生成两条独立路, 无交互.
右: HERMES (Fig.1d) —— world queries 经 LLM 因果注意力拿到世界知识,
    再经 current-to-future link 把知识注入编码 BEV, 生成未来 BEV.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BLUE, GREEN, ORANGE, PURPLE, GRAY = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6a55c2", "#6e737b"

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
fig.patch.set_facecolor("white")

def box(ax, xy, w, h, text, fc, ec=None, fs=10, sub=None):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.03", fc=fc, ec=ec or "none",
                       alpha=0.9, linewidth=1.5)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2 + (0.06 if sub else 0), text,
            ha="center", va="center", fontsize=fs, color="white" if ec is None else "#141413",
            weight="bold")
    if sub:
        ax.text(xy[0] + w / 2, xy[1] + h / 2 - 0.09, sub,
                ha="center", va="center", fontsize=8, color="white")
    return b

def arrow(ax, p1, p2, color=GRAY, ls="-", label=None):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                        color=color, linewidth=1.8, linestyle=ls)
    ax.add_patch(a)
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.06, label,
                ha="center", va="bottom", fontsize=8, color=color)

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ---------- 左: 分离式统一 ----------
ax = axes[0]
ax.set_title("(a) 分离式统一: 共享特征, 两路独立 (Fig.1c)", fontsize=12, weight="bold", pad=12)
box(ax, (0.35, 0.08), 0.3, 0.14, "BEV Tokenizer", BLUE, sub="多视角图像 → BEV")
arrow(ax, (0.5, 0.22), (0.5, 0.34))
box(ax, (0.28, 0.34), 0.44, 0.12, "Flattened BEV  Ft", ORANGE, fs=9)
arrow(ax, (0.32, 0.46), (0.18, 0.62))
arrow(ax, (0.68, 0.46), (0.82, 0.62))
box(ax, (0.02, 0.62), 0.32, 0.16, "LLM", PURPLE, sub="理解: 问答/描述")
box(ax, (0.66, 0.62), 0.32, 0.16, "Future Generator", GREEN, sub="生成: 未来点云")
arrow(ax, (0.18, 0.78), (0.18, 0.88))
arrow(ax, (0.82, 0.78), (0.82, 0.88))
ax.text(0.18, 0.93, "文本回答", ha="center", fontsize=10, color=PURPLE, weight="bold")
ax.text(0.82, 0.93, "未来点云", ha="center", fontsize=10, color=GREEN, weight="bold")
ax.text(0.5, 0.56, "两条路之间\n没有任何信息交互", ha="center", fontsize=9, color="#b34a06",
        style="italic", bbox=dict(fc="#f9e3d6", ec="none", pad=4))

# ---------- 右: HERMES ----------
ax = axes[1]
ax.set_title("(d) HERMES: world queries 搬运世界知识 (Fig.1d)", fontsize=12, weight="bold", pad=12)
box(ax, (0.35, 0.06), 0.3, 0.12, "BEV Tokenizer", BLUE, sub="It → BEV")
arrow(ax, (0.5, 0.18), (0.5, 0.26))
box(ax, (0.3, 0.26), 0.4, 0.1, "Flattened BEV  Ft", ORANGE, fs=9)
# max pool 分叉
arrow(ax, (0.38, 0.36), (0.2, 0.46), label="展平+MLP")
arrow(ax, (0.62, 0.36), (0.8, 0.46), label="max pool 取峰值")
box(ax, (0.02, 0.52), 0.36, 0.14, "LLM", PURPLE, sub="因果注意力: 理解+知识富集")
box(ax, (0.62, 0.52), 0.36, 0.14, "world queries Qw", "#6a55c2", sub="×Δt 组, 加 ego-motion")
# LLM 内部: BEV 在前, queries 在后才能看到文本知识
ax.text(0.5, 0.44, "queries 排在序列后方 → 能看到前方的文本 token (因果注意力)", ha="center",
        fontsize=8.5, color="#6a55c2", style="italic",
        bbox=dict(fc="#ebe7f8", ec="none", pad=3))
# LLM 输出
box(ax, (0.02, 0.72), 0.36, 0.1, "编码 BEV  Bt", ORANGE, fs=9)
box(ax, (0.62, 0.72), 0.36, 0.1, "富集后 queries", "#6a55c2", fs=9)
arrow(ax, (0.2, 0.66), (0.2, 0.72))
arrow(ax, (0.8, 0.66), (0.8, 0.72))
# current-to-future link
ax.text(0.5, 0.87, "current-to-future link (3× 交叉注意力)", ha="center", fontsize=9.5,
        color=GREEN, weight="bold", bbox=dict(fc="#dcf2e8", ec="none", pad=4))
arrow(ax, (0.2, 0.82), (0.42, 0.85), color=GREEN, label="作 query")
arrow(ax, (0.8, 0.82), (0.58, 0.85), color=GREEN, label="作 key/value")
box(ax, (0.28, 0.9), 0.44, 0.08, "未来 BEV Bt+1..Bt+Δt → Render → 未来点云", GREEN, fs=8.5)

fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/048-WM-hermes/fig-unification.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
