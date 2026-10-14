# -*- coding: utf-8 -*-
"""DriveGPT4-V2: 文本解码 vs DeciHead 的 token 经济学 + 推理速度对比.
数据来源: 论文 4.6 节正文("~160 tokens")与表 6 (FPS 1.4 vs 8.1).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Serif CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# (a) 每个前向输出的 token 数
ax = axes[0]
labels = ["文本解码\n(38个数×~4token)", "DeciHead\n(4个决策token)"]
vals = [160, 4]
colors = ["#b34a06", "#0b7a5c"]
bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white")
ax.set_ylabel("输出 token 数 / 前向")
ax.set_title("(a) 输出 4 组决策变量需要的 token 数", fontsize=11)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}", ha="center", fontsize=13, fontweight="bold")
ax.set_ylim(0, 185)
ax.text(0.5, 120, "约 40 倍", ha="center", fontsize=12, color="#73726c",
        transform=ax.transData)

# (b) 推理速度
ax = axes[1]
labels2 = ["逐点预测\n(8+10 个 token)", "DeciHead\n(4 个 token)"]
fps = [1.4, 8.1]
bars = ax.bar(labels2, fps, color=["#b34a06", "#0b7a5c"], width=0.55, edgecolor="white")
ax.set_ylabel("FPS (单张 A800, FP16)")
ax.set_title("(b) 两种输出头设计的推理速度", fontsize=11)
for b, v in zip(bars, fps):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}", ha="center", fontsize=13, fontweight="bold")
ax.set_ylim(0, 9.5)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("DriveGPT4-V2：为什么数值输出不能走文本解码", fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/015-VLA-drivegpt4-v2/fig-tokens.png", dpi=150)
print("saved fig-tokens.png")
