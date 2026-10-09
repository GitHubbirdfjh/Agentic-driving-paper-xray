#!/usr/bin/env python3
"""表 5：Action Dreaming 各类别成功率 —— 不加 dream 数据 vs SimLingo。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

IN   = "#1a6fa3"
OUT  = "#9e5c00"
WIRE = "#6e737b"
BG   = "#faf9f5"

cats = ["Faster\n加速", "Slower\n减速", "Target Speed\n目标速度", "Lane Change\n变道", "Objects\n朝物体开", "Avg\n平均"]
wo   = [56.45, 22.58, 19.35, 3.23, 20.97, 24.52]
sim  = [92.45, 84.91, 86.79, 83.02, 58.49, 81.13]

x = np.arange(len(cats))
wdt = 0.34
fig, ax = plt.subplots(figsize=(11.5, 5.6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

b1 = ax.bar(x - wdt / 2, wo, wdt, color=WIRE, alpha=0.75, edgecolor=WIRE, label="w/o Dream 数据（只用 CARLA 原生指令训）")
b2 = ax.bar(x + wdt / 2, sim, wdt, color=OUT, alpha=0.9, edgecolor=OUT, label="SimLingo（含 Dream 数据）")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.2f}",
                ha="center", fontsize=9.5, color="#141413")
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10.5)
ax.set_ylabel("成功率 SR (%)", fontsize=11)
ax.set_ylim(0, 108)
ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor=WIRE, loc="upper left")
ax.set_title("Action Dreaming：同一视觉上下文配多条指令，模型才开始听指令（表 5）", fontsize=13,
             weight="bold", color="#141413", pad=12)
ax.text(2.5, 12, "注意：正文写\"28.22 -> 72.96 SR\"，与表 5 的 24.52 -> 81.13 对不上（[未核实]，疑为旧版本数字）",
        fontsize=9.5, color="#b34a06", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#b34a06", alpha=0.95))
ax.grid(axis="y", color=WIRE, alpha=0.25, linestyle="--")
plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/011-VLA-simlingo/fig-dreaming-sr.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
