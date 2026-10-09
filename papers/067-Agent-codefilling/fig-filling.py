# -*- coding: utf-8 -*-
"""图2: 信息填充的几何动作 —— "拼图"
一维示意: ego 在各位置的信息现状 Cj, 两个 supporter 的信息分数,
填充优化把三者叠起来, 但在需求上界 u 处截断 —— 吃饱就停,
重叠区只被填一次, 避免多方同时发送同一块信息。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]

x = np.arange(12)
# ego 自己的信息分数: 位置 3-6 被遮挡, 分数低
Cj = np.array([0.9, 0.8, 0.7, 0.15, 0.1, 0.2, 0.1, 0.75, 0.85, 0.8, 0.7, 0.6])
# supporter1: 能看到位置 3-5
C1 = np.array([0.1, 0.1, 0.2, 0.85, 0.9, 0.8, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])
# supporter2: 能看到位置 5-7, 与 supporter1 在位置 5 重叠
C2 = np.array([0.1, 0.1, 0.1, 0.15, 0.2, 0.85, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1])

u = 1.0
# 无协调: 两家都发自己高分的位置 -> 位置 5 被发两次 (溢出)
# 有填充优化: 在每个位置上, 只选到把累计填到 u 为止
sel1 = np.zeros(12)
sel2 = np.zeros(12)
acc = Cj.copy()
# supporter1 先填
for i in range(12):
    if acc[i] < u and C1[i] > 0.5:
        sel1[i] = 1
        acc[i] = min(u, acc[i] + C1[i])
# supporter2 再填: 位置 5 已经被填满, 不再发
for i in range(12):
    if acc[i] < u and C2[i] > 0.5:
        sel2[i] = 1
        acc[i] = min(u, acc[i] + C2[i])

filled = np.minimum(Cj + sel1 * C1 + sel2 * C2, u)

fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), dpi=150,
                         gridspec_kw={"height_ratios": [1, 1]})
fig.patch.set_facecolor("white")
w = 0.8

def bars(ax, vals, color, label):
    ax.bar(x, vals, width=w, color=color, edgecolor="white", linewidth=0.8,
           label=label)
    ax.set_xlim(-0.7, 11.7)
    ax.set_ylim(0, 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"pos{i}" for i in x], rotation=0, fontsize=8)
    ax.axhline(u, color="#b34a06", ls="--", lw=1.2)
    ax.text(11.55, u + 0.03, "u = 1.0 (信息需求上界)", color="#b34a06",
            fontsize=9, ha="right", va="bottom")

# 上: 无协调 —— 两家都发
ax = axes[0]
ax.bar(x, Cj, width=w, color="#1a6fa3", alpha=0.9, label="ego Cj (自身信息)")
ax.bar(x, C1, width=w, bottom=Cj, color="#0b7a5c", alpha=0.85,
       label="supporter1 全发")
ax.bar(x, C2, width=w, bottom=Cj + C1, color="#6a55c2", alpha=0.85,
       label="supporter2 全发")
ax.axhline(u, color="#b34a06", ls="--", lw=1.2)
ax.text(11.55, u + 0.03, "u = 1.0", color="#b34a06", fontsize=9,
        ha="right", va="bottom")
ax.set_xlim(-0.7, 11.7)
ax.set_ylim(0, 2.6)
ax.set_xticks(x)
ax.set_xticklabels([f"pos{i}" for i in x], fontsize=8)
ax.set_ylabel("累积信息分数")
ax.set_title("无协调: 每个 supporter 独立做选择 —— 位置 5 被两家重复发送(溢出)",
             fontsize=12, pad=8)
ax.legend(loc="upper left", fontsize=9, ncol=3)
# 标出溢出
ax.annotate("溢出: 两家都发了 pos5", xy=(5, 1.9), xytext=(8.2, 2.25),
            fontsize=9, color="#b34a06",
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.2))

# 下: 信息填充 —— 截断在 u
ax = axes[1]
bars(ax, Cj, "#1a6fa3", "ego Cj")
ax.bar(x, sel1 * C1, width=w, bottom=Cj, color="#0b7a5c", alpha=0.9,
       label="supporter1 选中发送")
ax.bar(x, sel2 * C2, width=w, bottom=Cj + sel1 * C1, color="#6a55c2",
       alpha=0.9, label="supporter2 选中发送")
# 截断线: 填充后的有效值
ax.plot(x, filled, "o-", color="#b34a06", ms=4, lw=1.4,
        label="填充后有效信息 min(·, u)")
ax.set_ylabel("累积信息分数")
ax.set_title("信息填充: 累计到 u 就停 —— pos5 只由一家发, 另一家省下带宽",
             fontsize=12, pad=8)
ax.legend(loc="upper left", fontsize=9, ncol=4)

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/codefilling-xray/fig-filling.png",
            bbox_inches="tight")
print("saved fig-filling.png")
print("sel1:", sel1.astype(int).tolist())
print("sel2:", sel2.astype(int).tolist())
