# -*- coding: utf-8 -*-
"""Fig: Hierarchical vision-language alignment (OpenDriveVLA Stage 1).
Three structured token streams (scene/agent/map) live in different vision
manifolds; each goes through its own 2-layer MLP projector into ONE unified
semantic plane aligned with word embeddings. Frozen LLM generates captions.
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams["font.family"] = ["Noto Serif CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#faf9f5"; INK = "#141413"
C_IN, C_IN_F = "#1a6fa3", "#e2eff5"
C_OP, C_OP_F = "#0b7a5c", "#dcf2e8"
C_ALT, C_ALT_F = "#6a55c2", "#ebe7f8"
C_OUT, C_OUT_F = "#9e5c00", "#f8ecd2"
C_WIRE = "#6e737b"

rng = np.random.default_rng(7)
fig, ax = plt.subplots(figsize=(11, 5.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 11); ax.set_ylim(-0.75, 5.2); ax.axis("off")

def blob(cx, cy, sx, sy, n, color, fill, marker, alpha=0.85, seed=0):
    r = np.random.default_rng(seed)
    x = np.clip(r.normal(cx, sx, n), cx - 2.2*sx, cx + 2.2*sx)
    y = np.clip(r.normal(cy, sy, n), cy - 2.2*sy, cy + 2.2*sy)
    ax.scatter(x, y, s=46, c=fill, edgecolors=color, linewidths=1.4,
               marker=marker, alpha=alpha, zorder=3)

# --- three vision manifolds (left) ---
boxes = [
    (0.4, 0.4, 3.0, 4.4, "scene token  v_scene\n90 tokens, 6视图 2D 池化"),
    (0.4, 0.4, 3.0, 4.4, None),
]
ax.text(1.9, 4.75, "视觉流形（三个来源，形状各异）", fontsize=11, color=INK, ha="center", weight="bold")

# scene: wide flat ellipse
th = np.linspace(0, 2*np.pi, 120)
ax.add_patch(patches.Ellipse((1.9, 3.3), 2.6, 1.1, fill=False, ec=C_IN, lw=1.4, ls="--"))
blob(1.9, 3.3, 0.55, 0.22, 34, C_IN, C_IN_F, "o", seed=1)
ax.text(1.9, 2.55, "v_scene：全局 2D 上下文\n(天气/光照/布局)", fontsize=8.5, color=C_IN, ha="center")

# agent: cluster of small circles (instances)
ax.add_patch(patches.Ellipse((1.9, 1.75), 2.6, 0.75, fill=False, ec=C_ALT, lw=1.4, ls="--"))
blob(1.9, 1.75, 0.5, 0.13, 26, C_ALT, C_ALT_F, "o", seed=2)
ax.text(1.9, 1.22, "v_agent：每个动态目标一个 token\n(900 queries → 置信度过滤)", fontsize=8.5, color=C_ALT, ha="center")

# map: grid squares
ax.add_patch(patches.Rectangle((0.75, 0.15), 2.3, 0.62, fill=False, ec=C_OP, lw=1.4, ls="--"))
mx, my = np.meshgrid(np.linspace(0.95, 2.85, 9), np.linspace(0.25, 0.65, 3))
ax.scatter(mx.ravel(), my.ravel(), s=40, c=C_OP_F, edgecolors=C_OP, linewidths=1.2, marker="s", zorder=3)
ax.text(1.9, 0.98, "v_map：静态结构（车道线/边界/可行驶区）", fontsize=8.5, color=C_OP, ha="center")

# --- projectors (middle) ---
for i, (y, lab, col, fcol) in enumerate([(3.3, "Φ_scene", C_IN, C_IN_F),
                                         (1.75, "Φ_agent", C_ALT, C_ALT_F),
                                         (0.46, "Φ_map", C_OP, C_OP_F)]):
    ax.add_patch(patches.FancyBboxPatch((3.6, y-0.28), 1.5, 0.56, boxstyle="round,pad=0.02",
                                        fc=fcol, ec=col, lw=1.6, zorder=4))
    ax.text(4.35, y, lab, fontsize=10, color=col, ha="center", va="center", weight="bold")
    ax.annotate("", xy=(3.6, y), xytext=(3.15, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    ax.annotate("", xy=(6.4, 2.6), xytext=(5.1, y),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2, ls="--"))
ax.text(4.35, 4.15, "三个独立的两层 MLP 投影器\n(GeLU，Stage 1 唯一可训练参数，3.1M)", fontsize=9, color=INK, ha="center")

# --- unified semantic plane (right) ---
ax.add_patch(patches.Rectangle((6.4, 0.5), 3.0, 4.0, fill=False, ec=C_OUT, lw=2.0))
ax.text(7.9, 4.75, "统一语义平面（词嵌入空间）", fontsize=11, color=INK, ha="center", weight="bold")
# word embeddings: small ticks on a "vocabulary" strip
wx = np.linspace(6.6, 9.2, 26); wy = np.full_like(wx, 4.05)
ax.scatter(wx, wy, s=26, c=C_OUT, marker="s", alpha=0.75, zorder=3)
ax.text(7.9, 3.82, "词嵌入（frozen）", fontsize=8.5, color=C_OUT, ha="center")
# projected tokens land in the plane, mixed together
blob(7.9, 2.9, 0.55, 0.35, 22, C_IN, C_IN_F, "o", seed=3)
blob(7.9, 2.9, 0.5, 0.32, 18, C_ALT, C_ALT_F, "o", seed=4)
blob(7.9, 2.9, 0.52, 0.33, 16, C_OP, C_OP_F, "s", seed=5)
ax.text(7.9, 2.05, "投影后的视觉 token\n与词嵌入同处一平面", fontsize=8.5, color=INK, ha="center")

# --- frozen LLM on top ---
ax.add_patch(patches.FancyBboxPatch((6.4, 0.06), 3.0, 0.5, boxstyle="round,pad=0.02",
                                    fc="#ffffff", ec=C_WIRE, lw=1.4, ls=(0, (4, 3)), zorder=4))
ax.text(7.9, 0.31, "LLM（Stage 1 冻结）→ 生成 caption", fontsize=9, color=C_WIRE, ha="center")
ax.annotate("", xy=(7.9, 0.56), xytext=(7.9, 2.0),
            arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.4))

# per-instance caption note (stacked at bottom-left, clear of the map grid)
ax.text(1.9, -0.08, "agent token：逐个匹配带 BEV 坐标的物体 caption\n（TOD3Cap 后处理）",
        fontsize=8.2, color=C_ALT, ha="center", va="top",
        bbox=dict(fc="#ffffff", ec=C_ALT, alpha=0.95, boxstyle="round,pad=0.3"))
ax.text(1.9, -0.52, "scene / map token：整样本匹配场景级 caption",
        fontsize=8.2, color=C_IN, ha="center", va="top",
        bbox=dict(fc="#ffffff", ec=C_IN, alpha=0.95, boxstyle="round,pad=0.3"))

ax.set_title("Stage 1：分层视觉-语言对齐——三个视觉流形，各自投影，进同一语义平面", fontsize=12, color=INK, pad=14)
plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/075-Agent-opendrivevla/fig-align.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
