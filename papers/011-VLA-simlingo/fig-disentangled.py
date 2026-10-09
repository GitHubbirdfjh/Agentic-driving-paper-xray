#!/usr/bin/env python3
"""解耦输出表示：几何路径点（每 1 米，红）vs 时序速度点（每 0.25 秒，绿）；差分预测 + 累加。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

IN   = "#1a6fa3"
OUT  = "#9e5c00"
OP   = "#0b7a5c"
LOSS = "#b34a06"
WIRE = "#6e737b"
ALT  = "#6a55c2"
BG   = "#faf9f5"
RED  = "#c0392b"
GREEN = "#1e8449"

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
fig.patch.set_facecolor(BG)

# ---- 左：左转弯道上的两种 waypoint ----
ax = axes[0]
ax.set_facecolor(BG)
t = np.linspace(0, 1, 200)
# 一条左转曲线：x 前向，y 侧向
cx = 12 * t
cy = 6 * (1 - np.cos(t * np.pi / 2)) ** 1.2 * 2
cx = 14 * t
cy = 10 * t ** 2
ax.plot(cx, cy, color=WIRE, linewidth=1.5, linestyle="--", alpha=0.6, label="真实未来轨迹（示意）")

# 几何路径点：每 1 米一个（共 8 个）
arc = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
s = np.concatenate([[0], np.cumsum(arc)])
path_s = np.arange(1, 9) * 1.0
px = np.interp(path_s, s, cx)
py = np.interp(path_s, s, cy)
ax.scatter(px, py, s=90, color=RED, zorder=5, edgecolors="white", linewidths=1.2, label="path 点 p（每 1 米）")
for i, (x, y) in enumerate(zip(px, py), start=1):
    ax.annotate(f"p{i}", (x, y), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=RED)

# 时序速度点：每 0.25 秒一个，假设车速先快后慢（进弯减速）
speeds = [8, 8, 6, 5, 4, 4, 4, 5]  # m/s，每 0.25s 一段
dists = np.array(speeds) * 0.25
cumd = np.cumsum(dists)
wx = np.interp(cumd, s, cx)
wy = np.interp(cumd, s, cy)
ax.scatter(wx, wy, s=70, color=GREEN, marker="s", zorder=5, edgecolors="white", linewidths=1.2,
           label="speed 点 w（每 0.25 秒）")
ax.annotate("w 点间距随车速变化\n（快时稀、慢时密）", (wx[4], wy[4]), xytext=(30, 30),
            textcoords="offset points", fontsize=9, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.2))
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 11)
ax.set_aspect("equal")
ax.set_xlabel("x（前向，米）", fontsize=10)
ax.set_ylabel("y（侧向，米）", fontsize=10)
ax.set_title("解耦输出：几何路径 p（管方向）+ 时序速度 w（管快慢）", fontsize=12, weight="bold", color="#141413")
ax.legend(fontsize=9.5, loc="upper left", frameon=True, facecolor="white", edgecolor=WIRE)

# 从 w 取目标速度、从 p 取目标角度 -> 两个 PID
ax.text(13.2, 9.5, "w -> 目标速度\np -> 目标角度", fontsize=9.5, color="#141413", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=WIRE))
ax.text(13.2, 7.6, "纵向 PID -> 油门/刹车\n横向 PID -> 方向盘", fontsize=9.5, color=OP, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=OP))

# ---- 右：差分预测 + 累加 ----
ax = axes[1]
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("MLP 输出差分，累加得到路径点", fontsize=12, weight="bold", color="#141413")
diffs = [(0.9, 0.2), (0.85, 0.35), (0.70, 0.50)]
cum = np.cumsum(diffs, axis=0)
ax.text(5, 9.0, "20 个 path query -> MLP -> Δp1, Δp2, ...", fontsize=10, color=ALT, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ALT))
y = 7.6
for i, (d, c) in enumerate(zip(diffs, cum)):
    ax.text(2.2, y, f"Δp{i+1} = ({d[0]:.2f}, {d[1]:.2f})", fontsize=10.5, color=IN, ha="left", va="center")
    arr = FancyArrowPatch((4.9, y), (5.9, y), arrowstyle="-|>", mutation_scale=12, color=OP, linewidth=1.4)
    ax.add_patch(arr)
    ax.text(8.0, y, f"p{i+1} = ({c[0]:.2f}, {c[1]:.2f})", fontsize=10.5, color=RED, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=RED))
    y -= 1.5
ax.text(5, 2.4, "p_k = Σ_{i≤k} Δp_i（累加和）", fontsize=11, color="#141413", ha="center", weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f4ed", edgecolor=WIRE))
ax.text(5, 1.3, "预测差分而非绝对坐标：差分量级小、零均值附近，好学；\n累加天然保证路径连续（[推断]，论文未解释此动机）",
        fontsize=9, color=WIRE, ha="center", style="italic")
ax.text(5, 0.3, "代码确认：全量 SimLingo 也是 20 个 path query / 10 个 speed query（adaptors.py）", fontsize=8.5, color=WIRE, ha="center")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/011-VLA-simlingo/fig-disentangled.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
