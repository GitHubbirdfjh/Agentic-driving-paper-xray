import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["mathtext.fontset"] = "cm"
OUT = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/143-WM-piwm"

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

fig, ax = plt.subplots(figsize=(12.8, 7.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 1280); ax.set_ylim(0, 720); ax.axis("off")

ax.text(300, 690, "Dreamer:场景级世界模型", ha="center", fontsize=17,
        color="#141413", weight="bold")
ax.text(300, 662, "所有车辆拼成一根向量,信息混在一起",
        ha="center", fontsize=11, color=WIRE)
ax.text(980, 690, "PIWM:分支个体级世界模型", ha="center", fontsize=17,
        color="#141413", weight="bold")
ax.text(980, 662, "每辆车独立分支+self-attention 建交互+轨迹预测代替重建",
        ha="center", fontsize=11, color=WIRE)

# ---- left: Dreamer ----
box(ax, (60, 500), 480, 70, "观测 x_t: ego+VDI+VPI 全部拼接成一根向量", IN, IN_F, 12)
box(ax, (110, 370), 380, 90, "单个 RSSM", OP, OP_F, 14,
    sub="h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})")
box(ax, (150, 230), 300, 80, "重建损失: 解码回整幅观测", OUTC, OUT_F, 12,
    sub="−ln p(x_t | s_t)")
arrow(ax, (300, 500), (300, 460)); arrow(ax, (300, 370), (300, 310))
ax.text(300, 150, "问题: 一辆车的意图被淹没在场景里;\n重建损失逼它记住背景,不逼它预测意图",
        ha="center", fontsize=11, color=OUTC)

# ---- right: PIWM ----
box(ax, (700, 500), 560, 70, "观测:共享轨迹编码器 Enc_φ 后分三路", IN, IN_F, 12)
box(ax, (700, 390), 170, 80, "ego 分支", OP, OP_F, 12, sub="Enc_θ1")
box(ax, (895, 390), 170, 80, "VDI 分支", OP, OP_F, 12, sub="Enc_θ2")
box(ax, (1090, 390), 170, 80, "VPI 分支", OP, OP_F, 12, sub="Enc_θ3")
box(ax, (780, 265), 400, 80, "transition 里的 self-attention", ALT, ALT_F, 12,
    sub="self_att = Self_Attention(h_ego, h_VDI, h_VPI)")
box(ax, (700, 130), 260, 80, "ego+VDI 轨迹预测", OUTC, OUT_F, 12,
    sub="ŷ→y_hat = Dec(s_t), 2s 未来")
box(ax, (1000, 130), 260, 80, "reward / continuation", OUTC, OUT_F, 12,
    sub="cross-attention 融合 ego+VDI")
for x0 in (785, 980, 1175):
    arrow(ax, (x0, 500), (x0, 470))
arrow(ax, (890, 390), (890, 345)); arrow(ax, (1090, 390), (1090, 345))
arrow(ax, (880, 265), (800, 210)); arrow(ax, (1080, 265), (1160, 210))
ax.text(980, 60, "承重设计: 个体分支降低样本复杂度;\n轨迹预测代替重建,把意图逼进隐状态",
        ha="center", fontsize=11, color=OP)

ax.text(640, 300, "vs", ha="center", fontsize=26, color=WIRE, weight="bold")
fig.savefig(f"{OUT}/fig-branched.png", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/fig-branched.png")
