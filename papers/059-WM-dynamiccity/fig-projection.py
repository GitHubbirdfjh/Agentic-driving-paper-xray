# -*- coding: utf-8 -*-
"""Self-drawn figure: the Projection Module (one plane, Pxy) with concrete shapes.

Follows code: dynamic_city/vae/encoder.py TrEncoder.process_plane.
Concrete numbers: CarlaSC, sequence T=16; after 3D conv encoder (2x spatial down):
B x C x T x X x Y x Z with T=16, X=64, Y=64, Z=4, C=16.
For Pxy: keep (x, y) -> (B*X*Y) groups, each with T*Z tokens + 1 learnable token.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.5, 6.4))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 6.6)
ax.axis("off")

ANN = "#6e737b"
BLUE, BLUE_E = "#dcebf5", "#1a6fa3"
GREEN, GREEN_E = "#dcf2e8", "#0b7a5c"
PURP, PURP_E = "#ebe7f8", "#6a55c2"
ORNG, ORNG_E = "#f8ecd2", "#9e5c00"

ax.text(6.75, 6.25, "Projection Module: 以 Pxy 为例, 4D 体素如何被压成一张平面",
        ha="center", fontsize=14.5, weight="bold")
ax.text(6.75, 5.92, "code: TrEncoder.process_plane  |  每一步形状都按 CarlaSC 配置写出 (B=1, T=16, X=Y=64, Z=4, C=16)",
        ha="center", fontsize=10, color=ANN)

# step 1: volume
r1 = Rectangle((0.3, 2.9), 2.2, 1.7, facecolor=BLUE, edgecolor=BLUE_E, lw=1.8)
ax.add_patch(r1)
ax.text(1.4, 4.15, "Xtxyz", ha="center", fontsize=13, weight="bold")
ax.text(1.4, 3.72, "B×C×T×X×Y×Z", ha="center", fontsize=10)
ax.text(1.4, 3.30, "= 1×16×16×64×64×4", ha="center", fontsize=9, color=ANN)
ax.text(1.4, 2.55, "① 按 (x, y) 分组", ha="center", fontsize=9.5, color=BLUE_E, weight="bold")

# step 2: tokens
r2 = Rectangle((3.6, 2.9), 2.6, 1.7, facecolor=PURP, edgecolor=PURP_E, lw=1.8)
ax.add_patch(r2)
ax.text(4.9, 4.15, "(B·X·Y) 组 token", ha="center", fontsize=12, weight="bold")
ax.text(4.9, 3.72, "每组: T·Z + 1 = 65 个 token", ha="center", fontsize=10)
ax.text(4.9, 3.30, "4096 组 × 65 × 16 维", ha="center", fontsize=9, color=ANN)
ax.text(4.9, 2.55, "② +可学习 token, 2 层 transformer", ha="center", fontsize=9.5,
        color=PURP_E, weight="bold")
# token dots
for i, xx in enumerate([4.15, 4.45, 4.75, 5.35, 5.65]):
    c = ORNG_E if i == 3 else PURP_E
    ax.add_patch(Rectangle((xx, 3.95), 0.18, 0.12, facecolor=c, edgecolor=c))
ax.text(5.05, 3.98, "…", fontsize=12, color=ANN)

# step 3: take token 0
r3 = Rectangle((7.3, 2.9), 2.0, 1.7, facecolor=GREEN, edgecolor=GREEN_E, lw=1.8)
ax.add_patch(r3)
ax.text(8.3, 4.15, "取第 0 个 token", ha="center", fontsize=12, weight="bold")
ax.text(8.3, 3.72, "(B·X·Y) × C", ha="center", fontsize=10)
ax.text(8.3, 3.30, "= 4096 × 16", ha="center", fontsize=9, color=ANN)
ax.text(8.3, 2.55, "③ reshape 回平面", ha="center", fontsize=9.5, color=GREEN_E, weight="bold")

# step 4: plane + conv
r4 = Rectangle((10.4, 2.9), 2.5, 1.7, facecolor=ORNG, edgecolor=ORNG_E, lw=1.8)
ax.add_patch(r4)
ax.text(11.65, 4.15, "Pxy 平面", ha="center", fontsize=13, weight="bold")
ax.text(11.65, 3.72, "B×C×X×Y = 1×16×64×64", ha="center", fontsize=10)
ax.text(11.65, 3.30, "+ 两层 1×1 卷积 + Tanh", ha="center", fontsize=9, color=ANN)
ax.text(11.65, 2.55, "④ 六张平面拼成 HexPlane", ha="center", fontsize=9.5,
        color=ORNG_E, weight="bold")

for x0, x1 in [(2.5, 3.6), (6.2, 7.3), (9.3, 10.4)]:
    ax.annotate("", xy=(x1, 3.75), xytext=(x0, 3.75),
                arrowprops=dict(arrowstyle="-|>", color="#3d3d3a", lw=1.6))

ax.text(0.3, 1.75,
        "为什么不用平均池化? 平均池化对 T·Z 个 token 一视同仁, \n"
        "一辆快速穿过的车只占其中几个 token, 会被背景淹没。\n"
        "transformer 的自注意力让 (x, y) 位置自己决定\n"
        "该听哪几个时刻、哪个高度的 token —— 表 3 里\n"
        "Projection 相对 Average Pooling 高出 12.56 个 mIoU 点\n"
        "(CarlaSC, Query 解码器下 60.97 → 68.73)。",
        fontsize=9.5, color=ANN, va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f5f4ed", ec="#deddd8"))
ax.text(0.3, 0.35,
        "空间平面 (xy/xz/yz) 先过 ht 把 T 压掉再投影; 时间平面 (tx/ty/tz) 直接从 Xtxyz 投影, 一次压掉两个空间维 (code: separate_t_reducer)。",
        fontsize=9, color=ANN, va="top")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/059-WM-dynamiccity/fig-projection.png"
plt.savefig(out, dpi=150)
print("saved", out)
