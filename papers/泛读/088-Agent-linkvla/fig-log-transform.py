"""Log 坐标变换：z' = sign(z) log(1 + 5|z|)。
上：变换曲线与导数（局部密度）；下：均匀网格 vs log 网格在 BEV 上的对比。
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
WIRE = "#6e737b"

k = 5
z = np.linspace(0, 50, 400)
zp = np.log(1 + k * z)
dzp = k / (1 + k * z)  # 导数 = 局部分格密度

fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white",
                         gridspec_kw={"height_ratios": [1, 1.15]})

# ---------- 左上：变换曲线 ----------
ax = axes[0][0]
ax.plot(z, zp, color=IN, lw=2)
ax.set_xlabel("z (m)")
ax.set_ylabel("z' (变换后)")
ax.set_title(r"(a) $z'=\log(1+5z)$：远处被压扁", fontsize=12)
ax.grid(alpha=0.25)
for zv in [1, 2, 40, 41]:
    ax.plot(zv, np.log(1 + k * zv), "o", color=OUT, ms=6)
ax.text(2, 3.4, "1→2m：走 0.61\n(6 个格子)", color=OUT, fontsize=10)
ax.text(30, 4.6, "40→41m：走 0.03\n(不到 1 个格子)", color=OUT, fontsize=10)

# ---------- 右上：导数（密度） ----------
ax = axes[0][1]
ax.plot(z, dzp, color=IN, lw=2)
ax.set_xlabel("z (m)")
ax.set_ylabel("dz'/dz（格子/米）")
ax.set_title("(b) 局部分格密度：近处是远处的 250 倍", fontsize=12)
ax.grid(alpha=0.25)
ax.set_ylim(0, 5.5)

# ---------- 左下：均匀网格 ----------
ax = axes[1][0]
ax.set_xlim(0, 50)
ax.set_ylim(-30, 30)
ax.set_aspect("equal")
for gx in np.arange(0, 51, 5):
    ax.axvline(gx, color=WIRE, lw=0.7, alpha=0.7)
for gy in np.arange(-30, 31, 5):
    ax.axhline(gy, color=WIRE, lw=0.7, alpha=0.7)
ax.add_patch(plt.Rectangle((0, 0), 4, 3, fc="none", ec=OUT, lw=2))
ax.text(2, -4, "ego", ha="center", color=OUT, fontsize=10)
ax.set_title("(c) 均匀网格：近处不够细", fontsize=12)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# ---------- 右下：log 网格 ----------
ax = axes[1][1]
ax.set_xlim(0, 5.6)
ax.set_ylim(-5.05, 5.05)
ax.set_aspect("equal")
# 变换后的均匀格子，反解回物理坐标画出来
for i in range(57):
    x_t = i * 0.1
    x_p = (np.exp(x_t) - 1) / k
    ax.axvline(x_t, color=WIRE, lw=0.7, alpha=0.7)
for j in range(-50, 52):
    y_t = j * 0.1
    ax.axhline(y_t, color=WIRE, lw=0.4, alpha=0.35)
ax.text(2.8, 5.6, "变换后空间的均匀格子", ha="center", color=IN, fontsize=10)
ax.set_title("(d) log 网格：物理上近密远疏", fontsize=12)
ax.set_xlabel("x' (变换后)")
ax.set_ylabel("y' (变换后)")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/088-Agent-linkvla/fig-log-transform.png",
            dpi=150)
print("saved fig-log-transform.png")
