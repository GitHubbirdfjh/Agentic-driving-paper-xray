"""表 1 的真实数字：AP 随延迟的衰减曲线。
V2V4Real AP50：TraF-Align vs ERMVP vs V2VNet vs 单车。
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

lat = np.array([0, 100, 200, 300, 400])
# 表 1 V2V4Real AP50 真实数字
traf = np.array([74.28, 73.04, 71.51, 70.07, 69.41])
ermvp = np.array([70.74, 65.53, 61.63, 59.24, 57.26])
v2vnet = np.array([67.11, 61.97, 58.09, 56.26, 54.62])
single = np.array([48.75] * 5)

fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white")
ax.plot(lat, traf, "o-", color="#6a55c2", lw=2.2, ms=7, label="TraF-Align")
ax.plot(lat, ermvp, "s--", color="#1a6fa3", lw=1.6, ms=6, label="ERMVP")
ax.plot(lat, v2vnet, "d--", color="#6e737b", lw=1.4, ms=6, label="V2VNet")
ax.plot(lat, single, ":", color="#9e5c00", lw=1.4, label="Single Agent")
# 标注衰减量
ax.annotate("", xy=(400, 69.41), xytext=(400, 74.28),
            arrowprops=dict(arrowstyle="<->", color="#6a55c2", lw=1.4))
ax.text(415, 71.8, "-4.87", color="#6a55c2", fontsize=11, va="center")
ax.annotate("", xy=(400, 57.26), xytext=(400, 70.74),
            arrowprops=dict(arrowstyle="<->", color="#1a6fa3", lw=1.4))
ax.text(415, 64.0, "-13.48", color="#1a6fa3", fontsize=11, va="center")

ax.set_xlabel("latency (ms)")
ax.set_ylabel("AP50 (%) on V2V4Real")
ax.set_title("延迟 0→400ms：TraF-Align 只掉 4.87，ERMVP 掉 13.48（表 1 真实数字）",
             fontsize=12)
ax.set_xticks(lat)
ax.set_ylim(45, 78)
ax.legend(loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/087-Agent-traf-align/fig-latency-robustness.png",
            dpi=150)
print("saved fig-latency-robustness.png")
