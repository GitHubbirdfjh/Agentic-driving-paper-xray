import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["mathtext.fontset"] = "cm"
OUT = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/144-WM-sparseworldtc"

IN, IN_F = "#1a6fa3", "#e2eff5"
OUTC, OUT_F = "#9e5c00", "#f8ecd2"
OP, OP_F = "#0b7a5c", "#dcf2e8"
ALT, ALT_F = "#6a55c2", "#ebe7f8"
WIRE = "#6e737b"
BG = "#faf9f5"

def box(ax, xy, w, h, label, edge, fill, fs=12, sub=None):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=6",
                       ec=edge, fc=fill, lw=1.5)
    ax.add_patch(p)
    dy = 8 if sub else 0
    ax.text(x + w / 2, y + h / 2 + dy, label, ha="center", va="center",
            fontsize=fs, color=edge, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 16, sub, ha="center", va="center",
                fontsize=10, color=WIRE)

def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=WIRE, lw=1.3, shrinkA=2, shrinkB=3))

fig, ax = plt.subplots(figsize=(12.8, 6.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 1280); ax.set_ylim(0, 660); ax.axis("off")

ax.text(640, 620, "SparseWorld-TC 的稀疏 anchor 表示:随机 3D 点被\"洗\"成占据",
        ha="center", fontsize=16, color="#141413", weight="bold")
ax.text(640, 592, "没有 VAE 离散 token,也没有 BEV 网格;每个 anchor = M 个 3D 点 + 一条特征向量 f_i",
        ha="center", fontsize=11, color=WIRE)

rng = np.random.default_rng(7)
# random points cloud
px = rng.normal(230, 60, 60); py = rng.normal(300, 60, 60)
ax.scatter(px, py, s=26, color=IN, alpha=0.5)
ax.text(230, 170, "t+1 帧: N=600 个 anchor\n每 anchor M=128 个随机 3D 点\nf_i 全零初始化", ha="center", fontsize=11, color=IN,
        bbox=dict(boxstyle="round,pad=0.4", fc=IN_F, ec=IN))

arrow(ax, (360, 300), (470, 300))
box(ax, (480, 230), 320, 140, "attention 融合", OP, OP_F, 13,
    sub="anchor 特征 ← 图像证据(deformable)\nanchor 特征 ← 轨迹 embedding")
ax.text(640, 210, "逐层迭代 L 次", ha="center", fontsize=10, color=WIRE)
arrow(ax, (800, 300), (910, 300))

# denoised: two car-like clusters
cx1 = rng.normal(1010, 28, 45); cy1 = rng.normal(330, 26, 45)
cx2 = rng.normal(1160, 30, 40); cy2 = rng.normal(240, 26, 40)
ax.scatter(cx1, cy1, s=26, color=OUTC, alpha=0.75)
ax.scatter(cx2, cy2, s=26, color=ALT, alpha=0.75)
ax.text(1085, 170, "解码:两个 MLP\nΔp_{i,j}: 点偏移(洗干净)\ns_{i,j}: 语义类别\nChamfer + Focal 监督", ha="center", fontsize=11, color=OUTC,
        bbox=dict(boxstyle="round,pad=0.4", fc=OUT_F, ec=OUTC))

ax.text(640, 60, "对比 VAE 路线: token 表容量有限;对比 BEV 路线:几何先验写死。\n这里随机点+特征直接端到端,代价是点太多(Small 版每帧 600×128=76800 个点)",
        ha="center", fontsize=11, color="#141413")

fig.savefig(f"{OUT}/fig-anchor.png", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/fig-anchor.png")
