# -*- coding: utf-8 -*-
"""自绘图：WoTE 的轨迹级在线评估 vs RAMP 的中间分布复用 —— 机制对照。

左：WoTE：N 条候选轨迹 × 世界模型递推展开 K 步未来 BEV 状态 → reward 模型打分 → 选最高。
右：RAMP：观测只做一次前向得中间分布 q(z|o)，该分布同时喂给 LLM 调度先验与各候选动作的独立潜空间 rollout，
   风险仲裁器打分、安全盾兜底。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

IN = "#1a6fa3"; IN_FILL = "#e2eff5"
OP = "#0b7a5c"; OP_FILL = "#dcf2e8"
OUT = "#9e5c00"; OUT_FILL = "#f8ecd2"
ALT = "#6a55c2"; ALT_FILL = "#ebe7f8"
WIRE = "#6e737b"

def box(ax, xy, w, h, label, fc, ec, fontsize=9.5):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.6)
    ax.add_patch(b)
    ax.text(xy[0]+w/2, xy[1]+h/2, label, ha="center", va="center", fontsize=fontsize)
    return b

def arr(ax, p1, p2, c=WIRE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=c))

# ---------------- 左：WoTE ----------------
ax = axes[0]
ax.set_title("WoTE: model-based trajectory evaluation", fontsize=12, weight="bold", pad=10)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

box(ax, (3.5, 8.4), 3, 1.0, "BEV encoder\n(o_t -> B_t)", IN_FILL, IN)
box(ax, (3.5, 6.9), 3, 1.0, "trajectory predictor\n(N=256 anchors)", IN_FILL, IN)
box(ax, (3.5, 4.6), 3, 1.5, "BEV world model\n2-layer Transformer\nparallel recurrent\nrollout x K steps, N trajs", OP_FILL, OP)
box(ax, (3.5, 2.6), 3, 1.2, "reward model\n(B_t..B_{t+K}, a_t..a_{t+K})", OUT_FILL, OUT)
box(ax, (3.8, 0.8), 2.4, 1.1, "argmax r_i\n-> final trajectory", ALT_FILL, ALT)

arr(ax, (5.0, 8.4), (5.0, 7.9))
arr(ax, (5.0, 6.9), (5.0, 6.1))
arr(ax, (5.0, 4.6), (5.0, 3.8))
arr(ax, (5.0, 2.6), (5.0, 1.9))

# 每次rollout独立的标注：N 条平行未来
ax.text(8.35, 5.35, "N independent\nBEV-state futures\n(256 x K forward\ntransformer passes)", fontsize=8.5, color=OP,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OP, alpha=0.95))
arr(ax, (8.3, 5.35), (6.5, 5.35), c=OP)
ax.text(5.0, 0.15, "world model cost paid per trajectory", fontsize=9, ha="center", color=WIRE, style="italic")

# ---------------- 右：RAMP ----------------
ax = axes[1]
ax.set_title("RAMP: intermediate-distribution reuse", fontsize=12, weight="bold", pad=10)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

box(ax, (3.5, 8.4), 3, 1.0, "observation encoder\n(o_t -> q(z_t|o_<t))\nONE forward", IN_FILL, IN)
box(ax, (0.6, 6.2), 3.6, 1.3, "LLM scheduler prior\n(high-level decision\nconsumes q(z))", OUT_FILL, OUT)
box(ax, (5.8, 6.2), 3.6, 1.3, "risk arbiter\n+ safety shield\n(scores rollouts)", ALT_FILL, ALT)
box(ax, (2.9, 3.9), 4.2, 1.6, "per-candidate rollouts\nin latent space\nshare initial state q(z_t|o)\neach rolls out independently", OP_FILL, OP)
box(ax, (3.8, 1.5), 2.4, 1.1, "final action\narbitrated", ALT_FILL, ALT)

arr(ax, (3.9, 8.4), (2.4, 7.5), c=IN)
arr(ax, (6.1, 8.4), (7.6, 7.5), c=IN)
arr(ax, (7.6, 6.2), (6.9, 5.5), c=WIRE)
arr(ax, (2.4, 6.2), (4.2, 5.5), c=WIRE)
arr(ax, (5.0, 3.9), (5.0, 2.6), c=ALT)

ax.text(5.0, 0.35, "encoding cost paid once; q(z) has two heterogeneous consumers", fontsize=9, ha="center",
        color=WIRE, style="italic")
ax.text(0.25, 8.9, "shared initial\nlatent state", fontsize=8.5, color=IN, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=IN, alpha=0.95))

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/050-WM-wote/fig-wote-vs-ramp.png", dpi=150)
print("saved fig-wote-vs-ramp.png")
