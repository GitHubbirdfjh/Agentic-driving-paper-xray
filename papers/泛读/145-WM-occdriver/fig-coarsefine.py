import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["mathtext.fontset"] = "cm"
OUT = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/145-WM-occdriver"

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

fig, ax = plt.subplots(figsize=(12.8, 7.0))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 1280); ax.set_ylim(0, 700); ax.axis("off")

ax.text(640, 668, "OccDriver:粗→细迭代解码,占据世界模型夹在中间", ha="center",
        fontsize=16, color="#141413", weight="bold")
ax.text(640, 640, "Q_c = D_c(Q_vec; F_occ),  Q_s = D_s(F_occ; Q_c),  Q_f = D_f(Q_c; Q_s)",
        ha="center", fontsize=12, color=WIRE)

box(ax, (60, 520), 220, 80, "向量化分支输入", IN, IN_F, 12, sub="ego/车辆/静态/map")
box(ax, (60, 400), 220, 80, "栅格化分支输入", IN, IN_F, 12, sub="占据栅格+反向流")
arrow(ax, (280, 560), (380, 560)); arrow(ax, (280, 440), (380, 440))

box(ax, (400, 520), 240, 80, "D_c 粗轨迹解码器", OP, OP_F, 13, sub="M 个模态的粗轨迹 Y_c")
box(ax, (400, 400), 240, 80, "D_s 未来场景解码器", ALT, ALT_F, 13,
    sub="以 Q_c 为条件预测\n每模态的占据 O_e,O_a 与流")
arrow(ax, (640, 560), (740, 560))
box(ax, (760, 520), 240, 80, "D_f 精轨迹解码器", OP, OP_F, 13, sub="用 Q_s 精修→精轨迹 Y")
arrow(ax, (1000, 560), (1100, 560))
box(ax, (1100, 520), 140, 80, "输出 Y", OUTC, OUT_F, 13, sub="M 模态+置信度")

# cross arrow: D_c -> D_s
arrow(ax, (640, 480), (520, 440))
ax.text(560, 490, "每条粗轨迹都触发\n一次占据预测", ha="center", fontsize=10, color=ALT)
# loss boxes
box(ax, (400, 240), 220, 90, "L_oi 占据互斥", LOSS, LOSS_F, 12,
    sub="ego 占据 vs 车辆 GT 占据")
box(ax, (660, 240), 220, 90, "L_align / L_collision", LOSS, LOSS_F, 12,
    sub="轨迹点落在占据格子上\n打分/惩罚")
box(ax, (920, 240), 220, 90, "contingency", LOSS, LOSS_F, 12,
    sub="短期 marginal + 长期 joint\n占据混合, max 操作")
ax.text(640, 150, "代价:占据预测是 per-modality 的——M 条粗轨迹就要做 M 次条件预测,\n和 RiskWorld-Zeng 的\"一次预测全候选共享\"正好相反",
        ha="center", fontsize=11, color=LOSS,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=LOSS))
ax.text(640, 60, "nuPlan Val14: NR-S 0.896 / R-S 0.838,碰撞 0.971,TTC 0.938;推理 23.03ms(Test14-Hard)",
        ha="center", fontsize=11, color=WIRE)

fig.savefig(f"{OUT}/fig-coarsefine.png", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/fig-coarsefine.png")
