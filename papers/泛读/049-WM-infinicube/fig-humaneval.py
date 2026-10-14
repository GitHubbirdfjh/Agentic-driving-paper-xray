# -*- coding: utf-8 -*-
"""InfiniCube Fig 4: 人工评测 —— 生成视频与 HD map 对齐的正率 (论文表 2).
180 样本/帧索引; ours vs Panacea (共享 ControlNet 首帧).
Panacea 随帧数衰减, ours 基本持平 —— guidance buffer 抑制了长程漂移.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

frames = ["第 40 帧", "第 80 帧", "第 120 帧"]
ours = [84.6, 83.9, 84.8]
panacea = [76.8, 54.0, 53.4]

x = np.arange(len(frames))
w = 0.34

fig, ax = plt.subplots(figsize=(8.5, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

b1 = ax.bar(x - w / 2, ours, w, color="#0b7a5c", label="InfiniCube (ours)")
b2 = ax.bar(x + w / 2, panacea, w, color="#b34a06", label="Panacea [59]")
ax.set_xticks(x)
ax.set_xticklabels(frames, fontsize=11)
ax.set_ylabel("与 HD map 对齐的正率 (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.set_title("人工评测: 视频内容与 HD map 对齐率 (表 2, 180 样本/帧)", fontsize=12,
             weight="bold", pad=12)
ax.legend(fontsize=10, frameon=False)
ax.grid(True, axis="y", alpha=0.25)

for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.1f}%", ha="center", fontsize=10, weight="bold")

ax.text(1, 96, "Panacea 在 80 帧后明显漂移,\nours 几乎不衰减", ha="center", fontsize=9.5,
        color="#0b7a5c", style="italic", bbox=dict(fc="#dcf2e8", ec="none", pad=5))

fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/049-WM-infinicube/fig-humaneval.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
