# -*- coding: utf-8 -*-
"""Tab.A3：图像权重 λ1 的 synergy-conflict 曲线。
数据来自论文附录 D 表 A3（Dev10）。
主实验用 λ1=1.0（代码 im_weight 默认 1.0），但消融里 0.5 更好。
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
_font = [f for f in font_manager.fontManager.ttflist if f.name == "Noto Sans CJK JP"]
assert _font, "no CJK font"
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

LOSS = "#b34a06"; OUT = "#9e5c00"; WIRE = "#6e737b"; ALT = "#6a55c2"

w = np.array([0, 0.25, 0.5, 1, 2])
ds = np.array([53.97, 56.96, 61.13, 57.83, 54.44])
is_ = np.array([0.76, 0.68, 0.75, 0.65, 0.66])

fig, ax1 = plt.subplots(figsize=(8.5, 5.6))
ax1.set_title("NSP 与文本推理的 synergy–conflict：图像损失权重 lambda1 的影响（Dev10，论文表 A3）",
              fontsize=12.5, weight="bold", pad=14)
ax1.set_xlabel("图像损失权重 lambda1")
ax1.set_ylabel("Driving Score ↑", color=OUT, fontsize=12)
l1, = ax1.plot(w, ds, "o-", color=OUT, lw=2.4, ms=9, label="DS")
for x, y in zip(w, ds):
    ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=10, color=OUT, weight="bold")
ax1.tick_params(axis="y", labelcolor=OUT)

ax2 = ax1.twinx()
ax2.set_ylabel("Infraction Score ↑", color=WIRE, fontsize=12)
l2, = ax2.plot(w, is_, "s--", color=WIRE, lw=1.8, ms=8, label="IS")
for x, y in zip(w, is_):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -18),
                 ha="center", fontsize=10, color=WIRE)
ax2.tick_params(axis="y", labelcolor=WIRE)

ax1.axvline(0.5, color=ALT, ls=":", lw=1.8)
ax1.annotate("消融最优点 lambda1=0.5\nDS 61.13", xy=(0.5, 61.13), xytext=(1.35, 62.5),
             fontsize=11, color=ALT,
             arrowprops=dict(arrowstyle="->", color=ALT, lw=1.6))
ax1.axvline(1.0, color=LOSS, ls="-", lw=1.6)
ax1.annotate("主实验实际用的值 lambda1=1.0\n（代码 im_weight 默认值）\nDS 57.83，比最优点低 3.3",
             xy=(1.0, 57.83), xytext=(1.05, 55.2), fontsize=10.5, color=LOSS,
             arrowprops=dict(arrowstyle="->", color=LOSS, lw=1.6))

ax1.grid(True, alpha=0.25, axis="y")
fig.legend(handles=[l1, l2], loc="upper left", bbox_to_anchor=(0.1, 0.94))
ax1.text(0.02, -0.02, "图：lambda1 太小 NSP 信号可忽略，太大则视觉预测挤压文本推理；注意主实验取值并非消融最优点",
         transform=ax1.transAxes, fontsize=10, color="#73726c")
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/073-Agent-reasonplan/fig-weight-tradeoff.png",
            dpi=150, bbox_inches="tight")
print("saved fig-weight-tradeoff.png")
