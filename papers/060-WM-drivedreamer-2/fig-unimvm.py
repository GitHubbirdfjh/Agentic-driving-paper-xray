# -*- coding: utf-8 -*-
"""Self-drawn figure: UniMVM panorama + mask-unified generation tasks.

Left: six views {FL, F, FR, BR, B, BL} concatenated in width into one panorama
x' in R^{T x 3 x H x 6W} (paper eq.3, Fig.5). Right: the same mask mechanism
drives three tasks (unconditional / first-frame / front-video).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (ax1, ax2):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

ANN = "#6e737b"
VIEWS = ["FL", "F", "FR", "BR", "B", "BL"]
VC = ["#dcebf5", "#e2eff5", "#dcebf5", "#f8ecd2", "#f8ecd2", "#f8ecd2"]

ax1.text(5, 9.5, "UniMVM: 六个视图拼成一条全景带, 自注意力直接看到视图边界",
         ha="center", fontsize=12.5, weight="bold")
ax1.text(5, 9.05, "x′ ∈ R^{T×3×H×6W}, 拼接顺序 {FL, F, FR, BR, B, BL} (论文 3.2 节)",
         ha="center", fontsize=9.5, color=ANN)

# panorama strip
px0, pw, ph = 0.7, 8.6, 2.6
py0 = 5.4
for i, (v, c) in enumerate(zip(VIEWS, VC)):
    x = px0 + i * pw / 6
    ax1.add_patch(Rectangle((x, py0), pw / 6, ph, facecolor=c,
                            edgecolor="#1a6fa3", lw=1.4))
    ax1.text(x + pw / 12, py0 + ph / 2, v, ha="center", va="center",
             fontsize=13, weight="bold")
ax1.text(px0 + pw / 2, py0 - 0.45, "H × 6W 全景帧 (256×2688)", ha="center",
         fontsize=10, color=ANN)
# seam highlight between F and FR
sx = px0 + 2 * pw / 6
ax1.annotate("", xy=(sx, py0), xytext=(sx, py0 - 1.15),
             arrowprops=dict(arrowstyle="-|>", color="#b34a06", lw=1.6))
ax1.text(sx + 0.25, py0 - 0.75, "视图接缝:\n同一辆车跨过这里时\n必须连续",
         ha="left", va="center", fontsize=8.5, color="#b34a06")

ax1.text(0.7, 4.35,
         "旧做法 (DriveDreamer): 每个视图独立走 UNet,\n"
         "视图之间只靠 cross-view attention 软约束 ——\n"
         "加参数, 且不保证像素级对齐。\n"
         "UniMVM: 拼好之后就是一张普通全景图,\n"
         "时空注意力的感受野天然跨过接缝,\n"
         "不新增任何跨视图参数 (论文表 4 消融)。",
         fontsize=9.5, color=ANN, va="top",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f5f4ed", ec="#deddd8"))

ax1.text(0.7, 1.7,
         "代价: 全景图 256×2688 进 SVD 的 VAE,\n"
         "latent 很大; 但一次前向出六个视图,\n"
         "相对 Drive-WM 的逐视图自回归更快。",
         fontsize=9.5, color=ANN, va="top")

# ---- right: mask tasks ----
ax2.text(5, 9.5, "同一个 mask 机制, 三种生成任务",
         ha="center", fontsize=12.5, weight="bold")
ax2.text(5, 9.05, "公式 (3): p(x′) = p(x′·m) p(x′·(1−m) | x′·m)",
         ha="center", fontsize=9.5, color=ANN)

tasks = [
    ("无图像条件", "m 遮住全部帧\n纯生成, 多样性最高", "#b5d8ec"),
    ("首帧多视图", "m 只遮住第 2..T 帧\n首帧给定, 预测未来", "#a9e3cc"),
    ("前视视频", "m 遮住除 F 视图外全部\n单目视频外推全景", "#f0cf8a"),
]
for i, (title, desc, mc) in enumerate(tasks):
    y = 7.6 - i * 2.35
    ax2.text(0.4, y + 0.55, title, fontsize=11, weight="bold", va="center")
    # mini grid: 6 views x 4 frames
    gx0 = 3.6
    for f in range(4):
        for v in range(6):
            masked = (i == 0) or (i == 1 and f > 0) or (i == 2 and v != 1)
            col = "#9aa0a8" if masked else mc
            ec = "#6e737b"
            ax2.add_patch(Rectangle((gx0 + v * 0.85, y - f * 0.42), 0.78, 0.36,
                                    facecolor=col, edgecolor=ec, lw=0.7))
    ax2.text(3.6, y + 0.05, "帧→", fontsize=7.5, color=ANN)
    ax2.text(gx0 + 6 * 0.85 + 0.25, y - 0.6, desc, fontsize=8.5, color=ANN, va="top")

ax2.text(0.4, 0.75,
         "灰色 = 被 mask (要生成), 彩色 = 给定条件。\n"
         "训练时随机 mask, 推理时按任务选 mask ——\n"
         "一个模型覆盖三种条件设置 (论文表 1)。",
         fontsize=9, color=ANN, va="top",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ed", ec="#deddd8"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/060-WM-drivedreamer-2/fig-unimvm.png"
plt.savefig(out, dpi=150)
print("saved", out)
