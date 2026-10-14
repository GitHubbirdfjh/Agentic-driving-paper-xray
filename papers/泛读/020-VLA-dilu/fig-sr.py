# -*- coding: utf-8 -*-
"""DiLu vs GRAD 成功率对比 (论文图7a 的文字数字, 非从图中估读):
lane-4-density-2 / lane-5-density-2.5 / lane-5-density-3
DiLu: 70% / 65% / 35% ; GRAD: 69% / 47% / 10%
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

envs = ["lane-4-density-2\n(train env)", "lane-5-density-2.5", "lane-5-density-3"]
dilu = [70, 65, 35]
grad = [69, 47, 10]

x = np.arange(len(envs))
w = 0.36
fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar(x - w / 2, dilu, w, label="DiLu (40 memory items)", color="#1a6fa3")
b2 = ax.bar(x + w / 2, grad, w, label="GRAD (600k action steps)", color="#b34a06")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{int(b.get_height())}%", ha="center", fontsize=11)

ax.set_xticks(x); ax.set_xticklabels(envs, fontsize=10)
ax.set_ylabel("success rate (30 collision-free frames)", fontsize=11)
ax.set_title("DiLu vs GRAD: success rate across environments\n"
             "(both only optimized on lane-4-density-2; N=10 scenarios each)",
             fontsize=12)
ax.set_ylim(0, 88)
ax.legend(fontsize=10)
ax.text(2, 20, "GRAD relative drop:\n(69-10)/69 = 85.5%", fontsize=10,
        color="#b34a06", ha="center",
        bbox=dict(boxstyle="round", fc="#f9e3d6", ec="#b34a06"))
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/"
            "020-VLA-dilu/fig-sr.png", dpi=150, bbox_inches="tight")
print("saved fig-sr.png")
