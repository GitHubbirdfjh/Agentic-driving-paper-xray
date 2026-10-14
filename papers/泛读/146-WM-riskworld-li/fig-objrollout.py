import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["mathtext.fontset"] = "cm"
OUT = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/146-WM-riskworld-li"

IN, IN_F = "#1a6fa3", "#e2eff5"
OUTC, OUT_F = "#9e5c00", "#f8ecd2"
OP, OP_F = "#0b7a5c", "#dcf2e8"
ALT, ALT_F = "#6a55c2", "#ebe7f8"
LOSS, LOSS_F = "#b34a06", "#f9e3d6"
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

fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor(BG)

# ---- left: pipeline ----
ax = axes[0]
ax.set_xlim(0, 640); ax.set_ylim(0, 620); ax.axis("off")
ax.text(320, 590, "RiskWorld-Li:逐对象 latent 滚动找风险源", ha="center",
        fontsize=15, color="#141413", weight="bold")
box(ax, (40, 470), 240, 80, "冻结 V-JEPA2 编码", IN, IN_F, 12, sub="16 帧 RGB clip")
box(ax, (360, 470), 240, 80, "结构化状态编码", IN, IN_F, 12, sub="MLP + masked GRU")
arrow(ax, (280, 510), (360, 510))
box(ax, (120, 350), 400, 80, "关系注意力 RelAttn", OP, OP_F, 12,
    sub="ego–对象、对象–对象交互")
arrow(ax, (160, 470), (200, 430)); arrow(ax, (480, 470), (440, 430))
box(ax, (120, 220), 400, 90, "RSSM 式滚动 H=60 步(3s)", ALT, ALT_F, 12,
    sub="z~pθ(·|h), h=GRU(z,h); 每个候选独立滚")
arrow(ax, (320, 350), (320, 310))
box(ax, (40, 90), 250, 100, "风险分数 r_i", OUTC, OUT_F, 12, sub="f_pool(均值未来态)")
box(ax, (350, 90), 250, 100, "风险曲线/最小距离/TTR", OUTC, OUT_F, 12,
    sub="f_step 逐帧解码")
arrow(ax, (200, 220), (165, 190)); arrow(ax, (440, 220), (475, 190))

# ---- right: risk curves ----
ax = axes[1]
ax.set_xlim(0, 60); ax.set_ylim(0, 1.15)
ax.set_xlabel("未来时间步 k (共60步,3s)", fontsize=11)
ax.set_ylabel("预测风险概率", fontsize=11)
ax.set_title("风险源 vs 干扰项:rollout 让曲线提前分开", fontsize=13, weight="bold")
x = np.linspace(0, 60, 61)
ax.plot(x, 1 / (1 + np.exp(-(x - 30) / 6)), color=LOSS, lw=2.4, label="风险源对象 O3")
ax.plot(x, 0.08 + 0.05 * np.sin(x / 8), color=IN, lw=2, label="干扰对象 O1/O2")
ax.axhline(0.5, color=WIRE, ls="--", lw=1)
ax.text(46, 0.53, "阈值 γ", fontsize=10, color=WIRE)
ax.annotate("交叉点=TTR", xy=(30, 0.5), xytext=(8, 0.85),
            arrowprops=dict(arrowstyle="->", color=WIRE), fontsize=10, color="#141413")
ax.legend(fontsize=10, loc="upper left")
ax.text(30, -0.18, "RiskBench: 总体 F1 63.0%, FA 2.1%\n消融最大跌幅是去掉世界特征(63.0→50.6)",
        ha="center", fontsize=10, color=WIRE)

fig.tight_layout()
fig.savefig(f"{OUT}/fig-objrollout.png", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/fig-objrollout.png")
