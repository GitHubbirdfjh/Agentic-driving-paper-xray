# -*- coding: utf-8 -*-
"""LMDrive 附录表 7：notice 指令使用率消融。
只讲一件事：训练时只保留 25% 的 notice 指令最好 —— 全部保留（100%）
反而让车开得过度保守（RC 从 60.0 掉到 58.2），全部去掉（0%）则
违章变多（IS 0.68）。数据来自附录 B 表 7，LangAuto-Short。
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

OUT = "#9e5c00"; IN = "#1a6fa3"; WIRE = "#6e737b"; ALT = "#6a55c2"

pct = np.array([0, 25, 100])
ds = np.array([45.2, 50.6, 49.1])
rc = np.array([67.1, 60.0, 58.2])
is_ = np.array([0.68, 0.84, 0.83])

fig, ax1 = plt.subplots(figsize=(8.5, 5.4))
ax1.set_title("notice 指令用多少：25% 是甜点，100% 让车变怂（附录表 7，LangAuto-Short）",
              fontsize=12.5, weight="bold", pad=14)
ax1.set_xlabel("训练时保留的 notice 指令比例 (%)")
ax1.set_ylabel("Driving Score / Route Completion", color=OUT, fontsize=12)
l1, = ax1.plot(pct, ds, "o-", color=OUT, lw=2.4, ms=9, label="DS")
l2, = ax1.plot(pct, rc, "D--", color=IN, lw=1.8, ms=8, label="RC")
for x, y in zip(pct, ds):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=10, color=OUT, weight="bold")
for x, y in zip(pct, rc):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -18),
                 ha="center", fontsize=10, color=IN)
ax1.tick_params(axis="y", labelcolor=OUT)

ax2 = ax1.twinx()
ax2.set_ylabel("Infraction Score", color=WIRE, fontsize=12)
l3, = ax2.plot(pct, is_, "s:", color=WIRE, lw=1.8, ms=8, label="IS")
for x, y in zip(pct, is_):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(12, 4),
                 ha="left", fontsize=10, color=WIRE)
ax2.tick_params(axis="y", labelcolor=WIRE)

ax1.axvline(25, color=ALT, ls=":", lw=1.8)
ax1.annotate("25%：DS 50.6 最好\nIS 0.84", xy=(25, 50.6), xytext=(45, 52.5),
             fontsize=10.5, color=ALT,
             arrowprops=dict(arrowstyle="->", color=ALT, lw=1.6))
ax1.annotate("0%：IS 掉到 0.68\n（不听劝告，违章多）", xy=(0, 45.2), xytext=(10, 41.5),
             fontsize=10, color=WIRE,
             arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.4))
ax1.annotate("100%：RC 58.2\n（过度保守）", xy=(100, 49.1), xytext=(72, 47.0),
             fontsize=10, color=WIRE,
             arrowprops=dict(arrowstyle="->", color=WIRE, lw=1.4))
ax1.grid(True, alpha=0.25, axis="y")
fig.legend(handles=[l1, l2, l3], loc="upper left", bbox_to_anchor=(0.1, 0.93))
ax1.text(0.02, -0.015, "图：notice 指令保留比例的权衡 —— 听劝和不怂之间有一个甜点",
         transform=ax1.transAxes, fontsize=10, color="#73726c")
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/074-Agent-lmdrive/fig-notice-ablation.png",
            dpi=150, bbox_inches="tight")
print("saved fig-notice-ablation.png")
