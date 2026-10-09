"""C2F 两步解码：(1) 预测终点；(2) 线性插值成粗轨迹，并行细化。
对比自回归的 T 步串行。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager

fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

IN = "#1a6fa3"
OUT = "#9e5c00"
OP = "#0b7a5c"
ALT = "#6a55c2"
WIRE = "#6e737b"

T = 8
w0 = np.array([0.0, 0.0])
wT = np.array([18.0, -3.5])
# 粗轨迹：线性插值
coarse = np.array([w0 + (i + 1) / T * (wT - w0) for i in range(T)])
# 精轨迹：加一点曲率（示意细化）
fine = coarse.copy()
fine[:, 1] += 1.2 * np.sin(np.linspace(0, np.pi, T)) * -1

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")

# ---------- 左：自回归 ----------
ax = axes[0]
ax.set_xlim(-2, 22)
ax.set_ylim(-8, 4)
ax.set_aspect("equal")
ax.set_title("(a) 自回归：T 步串行，361ms", fontsize=12)
pts = np.array([w0 + (i + 1) / T * (wT - w0) for i in range(T)])
pts[:, 1] += 1.2 * np.sin(np.linspace(0, np.pi, T)) * -1
prev = w0
ax.plot(w0[0], w0[1], "s", color=IN, ms=9)
ax.text(w0[0] - 0.6, w0[1] + 0.8, "w0", color=IN, fontsize=10)
for i, p in enumerate(pts):
    ax.annotate("", xy=p, xytext=prev,
                arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.4))
    ax.plot(p[0], p[1], "o", color=OUT, ms=6, alpha=0.5 + 0.5 * i / T)
    prev = p
ax.text(10, 2.2, "一步错，步步错\n(exposure bias)", ha="center", color="#b34a06",
        fontsize=10, style="italic")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# ---------- 右：C2F ----------
ax = axes[1]
ax.set_xlim(-2, 22)
ax.set_ylim(-8, 4)
ax.set_aspect("equal")
ax.set_title("(b) C2F：2 次前向，48ms", fontsize=12)
# 障碍物
ax.add_patch(plt.Circle((9, -1.2), 1.6, fc=WIRE, alpha=0.25, ec=WIRE))
ax.text(9, -1.2, "obs", ha="center", va="center", color=WIRE, fontsize=9)
# step1: 终点
ax.plot(w0[0], w0[1], "s", color=IN, ms=9)
ax.plot(wT[0], wT[1], "*", color=ALT, ms=15, mec="white")
ax.text(wT[0] + 0.8, wT[1] + 0.6, "step1: 终点", color=ALT, fontsize=10)
ax.annotate("", xy=wT, xytext=(14, 1.5),
            arrowprops=dict(arrowstyle="->", color=ALT, lw=1.4, ls="--"))
ax.text(13.2, 2.0, "1 次前向", color=ALT, fontsize=9)
# step2: 粗轨迹
ax.plot(coarse[:, 0], coarse[:, 1], "o--", color=WIRE, ms=5, alpha=0.8,
        label="coarse")
# 精轨迹
ax.plot(fine[:, 0], fine[:, 1], "o-", color=OP, ms=6, lw=2, label="fine")
for c, f in zip(coarse, fine):
    ax.annotate("", xy=f, xytext=c,
                arrowprops=dict(arrowstyle="->", color=OP, lw=1.0, alpha=0.7))
ax.text(4, -5.6, "step2: 线性插值→并行细化 (1 次前向)", color=OP, fontsize=10)
ax.legend(loc="upper left", fontsize=9)
ax.set_xlabel("x (m)")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/088-Agent-linkvla/fig-c2f.png",
            dpi=150)
print("saved fig-c2f.png")
