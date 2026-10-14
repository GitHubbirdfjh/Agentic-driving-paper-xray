# -*- coding: utf-8 -*-
"""080 Fig3: 论文图7的近似重绘 —— LLM 延迟容忍：REA 缓降，安全指标持平。数值从论文图7近似读出。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 从论文 Fig.7 曲线近似读出（已标注近似）
lat = np.array([0, 1, 2, 3, 4])
rea       = np.array([0.84, 0.82, 0.79, 0.75, 0.69])   # instruction realization
collision = np.array([0.99, 0.99, 0.98, 0.98, 0.97])   # collision avoidance
ttc       = np.array([0.88, 0.87, 0.87, 0.86, 0.85])   # TTC

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(lat, rea, "o-", color="#9e5c00", lw=2.2, ms=7, label="Instruction Realization (REA)")
ax.plot(lat, collision, "s--", color="#0b7a5c", lw=1.6, ms=6, label="Collision Avoidance")
ax.plot(lat, ttc, "^--", color="#1a6fa3", lw=1.6, ms=6, label="TTC")
ax.set_xlim(-0.2, 4.2); ax.set_ylim(0.55, 1.05)
ax.set_xlabel("LLM inference latency (s)"); ax.set_ylabel("score")
ax.set_title("Fig.7 of the paper (approximate redraw): REA degrades gradually, safety metrics stay flat",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.25)
ax.text(4.05, 0.62, "values read approximately\nfrom Fig.7", fontsize=8, color="#6e737b", ha="right",
        bbox=dict(fc="white", ec="#6e737b", lw=0.6))

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/080-Agent-open-ended-scheduling/fig-latency.png",
            dpi=160, bbox_inches="tight", facecolor="white")
print("saved fig-latency.png")
