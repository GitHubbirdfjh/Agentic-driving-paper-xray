import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["mathtext.fontset"] = "cm"
OUT = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/147-WM-riskworld-zeng"

IN, IN_F = "#1a6fa3", "#e2eff5"
OUTC, OUT_F = "#9e5c00", "#f8ecd2"
OP, OP_F = "#0b7a5c", "#dcf2e8"
ALT, ALT_F = "#6a55c2", "#ebe7f8"
LOSS, LOSS_F = "#b34a06", "#f9e3d6"
WIRE = "#6e737b"
BG = "#faf9f5"

def box(ax, xy, w, h, label, edge, fill, fs=12, sub=None, fss=10):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=6",
                       ec=edge, fc=fill, lw=1.5)
    ax.add_patch(p)
    dy = 10 if sub else 0
    ax.text(x + w / 2, y + h / 2 + dy, label, ha="center", va="center",
            fontsize=fs, color=edge, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 20, sub, ha="center", va="center",
                fontsize=fss, color=WIRE)

def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=WIRE, lw=1.3, shrinkA=2, shrinkB=3))

fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.6))
fig.patch.set_facecolor(BG)

# ============ 左: RiskWorld-Zeng ============
ax = axes[0]
ax.set_facecolor(BG)
ax.set_xlim(0, 680); ax.set_ylim(0, 760); ax.axis("off")
ax.text(340, 730, "RiskWorld-Zeng:一次预测,供 12 条候选轨迹复用", ha="center",
        fontsize=15, color="#141413", weight="bold")
ax.text(340, 700, "同构候选复用:所有消费者做同一件事(查未来风险、选轨迹)",
        ha="center", fontsize=11, color=WIRE)

box(ax, (180, 590), 320, 80, "流引导占据演化(一次前向)", OP, OP_F, 13,
    sub="warp 传输 + 符号修正,{(O_t,S_t)},T=6")
arrow(ax, (340, 590), (340, 545))
box(ax, (180, 455), 320, 70, "共享未来预测(冻结)", ALT, ALT_F, 13,
    sub="每个规划步只算一次")

# 12 candidates as small rows
ax.text(90, 400, "12 条候选轨迹\n(码本 10 + 2 运动学)", ha="center", fontsize=10, color=IN)
for i in range(6):
    y = 380 - i * 26
    ax.add_patch(FancyBboxPatch((200, y), 260, 20,
                 boxstyle="round,pad=0.01,rounding_size=4", ec=IN, fc=IN_F, lw=1))
for i in range(6):
    y = 380 - i * 26
    ax.add_patch(FancyBboxPatch((490, y), 130, 20,
                 boxstyle="round,pad=0.01,rounding_size=4", ec=IN, fc=IN_F, lw=1))
arrow(ax, (460, 300), (555, 300))
ax.text(340, 268, "每个候选查预测 + 持久性参照\n对比→非负碰撞分修正", ha="center",
        fontsize=10, color=WIRE)
box(ax, (180, 130), 320, 100, "选择(不改几何)", OUTC, OUT_F, 13,
    sub="anchor 保留; 触发干预且约束满足\n才换候选")
arrow(ax, (340, 370), (340, 230))
ax.text(340, 60, "K=1→24 时共享预测耗时 13.76→13.87ms\n边际成本≈0:这就是它的\"复用\"",
        ha="center", fontsize=11, color=OP)

# ============ 右: RAMP ============
ax = axes[1]
ax.set_facecolor(BG)
ax.set_xlim(0, 680); ax.set_ylim(0, 760); ax.axis("off")
ax.text(340, 730, "RAMP 创新点 B:单次前向,中间分布供两种消费者", ha="center",
        fontsize=15, color="#141413", weight="bold")
ax.text(340, 700, "异构双消费者复用:同一份中间分布,两种不同用法",
        ha="center", fontsize=11, color=WIRE)

box(ax, (180, 590), 320, 80, "世界模型单次前向", OP, OP_F, 13,
    sub="Bench2Drive 闭环, 多规划器调度")
arrow(ax, (340, 590), (340, 545))
box(ax, (160, 455), 360, 70, "中间分布(一次算出,不再重算)", ALT, ALT_F, 13,
    sub="未来状态/风险的分布表示")

# two consumers
arrow(ax, (260, 455), (160, 400))
arrow(ax, (420, 455), (520, 400))
box(ax, (30, 250), 260, 120, "消费者 1:调度先验", IN, IN_F, 12,
    sub="LLM 调度器用它\n给多规划器打分/排序")
box(ax, (390, 250), 260, 120, "消费者 2:风险校验", LOSS, LOSS_F, 12,
    sub="闭环风险校验/\n冲突仲裁用它")
ax.text(340, 130, "划界动词:\n147 是\"一次预测 → 多个候选\"(同构)\nRAMP 是\"一次前向 → 两种不同用途\"(异构)",
        ha="center", fontsize=11, color="#141413",
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec=WIRE))
ax.text(340, 40, "147 的预测是动作无关的(候选只决定查询位置);\nRAMP 的中间分布要同时承载调度与校验两种语义",
        ha="center", fontsize=11, color=OP)

fig.tight_layout()
fig.savefig(f"{OUT}/fig-reuse-compare.png", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/fig-reuse-compare.png")
