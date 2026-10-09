import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["mathtext.fontset"] = "cm"

IN, IN_F = "#1a6fa3", "#e2eff5"
OUT, OUT_F = "#9e5c00", "#f8ecd2"
OP, OP_F = "#0b7a5c", "#dcf2e8"
ALT, ALT_F = "#6a55c2", "#ebe7f8"
WIRE = "#6e737b"
BG = "#faf9f5"

def box(ax, xy, w, h, label, edge, fill, fs=13, sub=None):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=6",
                       ec=edge, fc=fill, lw=1.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2 + (4 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=edge, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 12, sub, ha="center", va="center",
                fontsize=10, color=WIRE)
    return (x + w, y + h / 2)

def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=WIRE, lw=1.3, shrinkA=1, shrinkB=2))

# ---------- 图1: 交错序列 ----------
fig, ax = plt.subplots(figsize=(12.6, 4.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 1260); ax.set_ylim(0, 460); ax.axis("off")

ax.text(630, 430, "交错 vision-action 序列:一张图、一个动作、一张图、一个动作……",
        ha="center", fontsize=16, color="#141413", weight="bold")

x, y0, bh = 30, 150, 130
end = box(ax, (x, y0), 60, bh, "<SYS>", WIRE, "#f2f0e8", fs=11); x = end[0] - 65 + 75
# 三个历史对
for i, t in enumerate(["t-3", "t-2", "t-1"]):
    arrow(ax, (x, y0 + bh / 2), (x + 14, y0 + bh / 2))
    x += 14
    e1 = box(ax, (x, y0), 120, bh, r"$\mathtt{<img>}$", IN, IN_F, fs=13, sub="576 tokens")
    x = e1[0]
    arrow(ax, (x, y0 + bh / 2), (x + 14, y0 + bh / 2)); x += 14
    e2 = box(ax, (x, y0), 110, bh, r"$\mathtt{<act>}$", OUT, OUT_F, fs=13, sub="数字词")
    x = e2[0]
    ax.text(x - 55, y0 - 22, f"第{i+1}对 · {t}", ha="center", fontsize=11, color=WIRE)
# 当前帧
arrow(ax, (x, y0 + bh / 2), (x + 14, y0 + bh / 2)); x += 14
ec = box(ax, (x, y0), 120, bh, r"$\mathtt{<img>}$", IN, IN_F, fs=13, sub="当前帧 t")
x = ec[0]
ax.text(x - 60, y0 - 22, "当前帧", ha="center", fontsize=11, color=WIRE)
# MLLM
arrow(ax, (x, y0 + bh / 2), (x + 16, y0 + bh / 2)); x += 16
em = box(ax, (x, y0), 120, bh, "MLLM", OP, OP_F, fs=14, sub="Vicuna-7B")
x = em[0]
arrow(ax, (x, y0 + bh / 2), (x + 16, y0 + bh / 2)); x += 16
ea = box(ax, (x, y0), 110, bh, r"$\mathtt{<act>}$", OUT, OUT_F, fs=13, sub="预测 A't")
ax.text(x + 55, y0 - 22, "输出", ha="center", fontsize=11, color=WIRE)

# 动作编码放大
ax.text(630, 95, "一个动作词长什么样:  steer = -0.091 rad  →  ×1000 取整  →  "
        r"$\mathtt{<num\_start>}$ - 0 9 1 $\mathtt{<num\_end>}$",
        ha="center", fontsize=13, color=OUT,
        bbox=dict(boxstyle="round,pad=0.4", fc=OUT_F, ec=OUT, lw=1.2))
ax.text(630, 45, "速度同理:12.345 m/s → 12345。数字变成词,LLM 才能用交叉熵直接学",
        ha="center", fontsize=12, color=WIRE)
fig.savefig("fig-interleave.png", dpi=150, bbox_inches="tight", facecolor=BG)

# ---------- 图2: 无限循环 ----------
fig, ax = plt.subplots(figsize=(12.6, 6.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 1260); ax.set_ylim(0, 620); ax.axis("off")

ax.text(630, 585, "在自己创造的世界里开车:动作 → 未来帧 → 动作,无限循环",
        ha="center", fontsize=16, color="#141413", weight="bold")

y1, bh = 380, 120
x = 40
e_h = box(ax, (x, y1), 170, bh, "历史 (I,A) 对", IN, IN_F, fs=12, sub="t-3..t-1")
x = e_h[0]
arrow(ax, (x, y1 + bh / 2), (x + 18, y1 + bh / 2)); x += 18
e_i = box(ax, (x, y1), 130, bh, "当前帧 It", IN, IN_F, fs=13)
x = e_i[0]
arrow(ax, (x, y1 + bh / 2), (x + 18, y1 + bh / 2)); x += 18
e_m = box(ax, (x, y1), 130, bh, "MLLM", OP, OP_F, fs=14, sub="猜动作")
x = e_m[0]
arrow(ax, (x, y1 + bh / 2), (x + 18, y1 + bh / 2)); x += 18
e_a = box(ax, (x, y1), 140, bh, "控制信号 A't", OUT, OUT_F, fs=13, sub="速度+转角")
x = e_a[0]

y2 = 130
# GPT3.5 motion prompt
ax.annotate("", xy=(x + 40, y1), xytext=(x + 40, y2 + bh),
            arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.3,
                            connectionstyle="angle3,angleA=0,angleB=90"))
e_g = box(ax, (x - 55, y2), 190, bh, "GPT-3.5", ALT, ALT_F, fs=13, sub="数字→运动描述")
gx = x - 55 + 190
arrow(ax, (gx, y2 + bh / 2), (gx + 16, y2 + bh / 2)); gx += 16
e_v = box(ax, (gx, y2), 190, bh, "VDM 扩散模型", OP, OP_F, fs=13, sub="按动作画未来帧")
gx = gx + 190
arrow(ax, (gx, y2 + bh / 2), (gx + 16, y2 + bh / 2)); gx += 16
e_f = box(ax, (gx, y2), 200, bh, "未来四帧", IN, IN_F, fs=13, sub="取 I't+1 继续开")
fx_cx, fx_top = gx + 100, y2 + bh
m_cx, m_top = e_m[0] - 65, y1 + bh
# 循环箭头:从未来帧框顶边出发,绕到 MLLM 框顶边
ax.annotate("", xy=(m_cx, m_top + 4), xytext=(fx_cx, fx_top + 4),
            arrowprops=dict(arrowstyle="-|>", color=OUT, lw=1.6,
                            connectionstyle="arc3,rad=0.28", shrinkA=2, shrinkB=2))
ax.text(1030, 480, "把预测帧当作下一帧喂回去", ha="center", fontsize=12, color=OUT,
        bbox=dict(boxstyle="round,pad=0.35", fc=OUT_F, ec=OUT, lw=1.2))

ax.text(630, 60, "训练时 MLLM 与 VDM 分开训练;推理时才把两环扣在一起。VDM 画歪一帧,下一轮的动作就跟着歪——作者在 Discussion 里承认了这一点",
        ha="center", fontsize=12, color=WIRE,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f9e3d6", ec="#b34a06", lw=1.2))
fig.savefig("fig-loop.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("done")
