"""轨迹场示意：低维轨迹（position field heatmap + orientation field 箭头）
引导高维特征的注意力采样。左：position field；右：orientation field。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
from matplotlib.font_manager import fontManager
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

IN = "#1a6fa3"
OUT = "#9e5c00"
OP = "#0b7a5c"
WIRE = "#6e737b"
ALT = "#6a55c2"

# 构造一条轨迹：从 (10, 25) 到 (40, 25)，轻微弯曲
t = np.linspace(0, 1, 60)
xs = 10 + 30 * t
ys = 25 + 4 * np.sin(t * np.pi * 0.8)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")

# ---------- 左：position field heatmap ----------
ax = axes[0]
gx, gy = np.meshgrid(np.linspace(0, 50, 100), np.linspace(10, 40, 60))
heat = np.zeros_like(gx)
for x0, y0 in zip(xs, ys):
    heat += np.exp(-((gx - x0) ** 2 + (gy - y0) ** 2) / 6.0)
heat /= heat.max()
im = ax.imshow(heat, origin="lower", extent=(0, 50, 10, 40),
               cmap="YlOrBr", vmin=0, vmax=1, aspect="auto")
ax.plot(xs, ys, color=ALT, lw=1.5, ls="--", alpha=0.9)
# 历史观测位置（路侧在 s 时刻看到的）
for i, (x0, y0) in enumerate(zip(xs[::15], ys[::15])):
    ax.plot(x0, y0, "o", color=IN, ms=7)
    ax.text(x0, y0 - 2.2, f"s-{3-i}", ha="center", color=IN, fontsize=9)
# 预测的当前位置
ax.plot(xs[-1], ys[-1], "*", color=OUT, ms=14, mec="white")
ax.text(xs[-1] + 1.5, ys[-1] + 1.0, "predicted @ t", color=OUT, fontsize=10)
ax.set_title("(a) position field：沿轨迹的 heatmap 峰", fontsize=12)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# ---------- 右：orientation field 箭头 ----------
ax = axes[1]
ax.imshow(heat, origin="lower", extent=(0, 50, 10, 40),
          cmap="YlOrBr", vmin=0, vmax=1, alpha=0.35, aspect="auto")
# 方向向量
idx = np.arange(0, 60, 6)
dx = np.gradient(xs)[idx]
dy = np.gradient(ys)[idx]
nrm = np.hypot(dx, dy) + 1e-9
ax.quiver(xs[idx], ys[idx], dx / nrm, dy / nrm, color=OP, scale=4.0, width=0.006,
          headlength=4, headwidth=3)
# query 与 attention 采样点
qx, qy = xs[-1] - 1.0, ys[-1]
ax.plot(qx, qy, "s", color="#b34a06", ms=9)
for x0, y0 in zip(xs[idx][:-1], ys[idx][:-1]):
    ax.plot(x0, y0, "o", color="#b34a06", ms=5, alpha=0.8)
ax.text(qx + 1.2, qy + 1.2, "query @ t", color="#b34a06", fontsize=10)
ax.text(xs[12], ys[12] - 2.6, "18 个 attention 点沿轨迹排开",
        ha="center", color="#b34a06", fontsize=10)
ax.set_title("(b) orientation field + attention 采样点", fontsize=12)
ax.set_xlabel("x (m)")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/087-Agent-traf-align/fig-trajectory-field.png",
            dpi=150)
print("saved fig-trajectory-field.png")
