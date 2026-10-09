# -*- coding: utf-8 -*-
"""One-hop 问答走查：场景图 → 找参照物 → 算关系边 → 过滤候选 → 读状态。
问题："What is the status of the moving car to the front left of the stopped bus?"
并展示检测框噪声如何让 30° 边界附近的物体掉出候选集。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.6), gridspec_kw={"width_ratios": [1, 1.25]})
fig.suptitle("走查一道 one-hop 题：场景图推理的四步 + 它在哪里会断", fontsize=13, weight="bold", y=0.98)

# ---------- 左：BEV 场景 ----------
ax1.set_xlim(-9, 9); ax1.set_ylim(-9, 12); ax1.set_aspect("equal")
ax1.set_xlabel("x"); ax1.set_ylabel("y（ego 前进方向）")
ax1.set_title("场景（BEV，ego 在原点朝 +y）", fontsize=11)
ax1.axhline(0, color="#999", lw=0.8, ls="--"); ax1.axvline(0, color="#999", lw=0.8, ls="--")
ax1.add_patch(mpatches.Rectangle((-1, -2), 2, 4, fc="#333", ec="k"))
ax1.text(0, 0, "me\n(moving)", color="white", ha="center", va="center", fontsize=8, weight="bold")

objs = {
    "bus": ((6, 4), "stopped", "#c62828"),
    "car1": ((2, 9), "moving", "#1a6fa3"),
    "car2": ((-5, 2), "parked", "#777"),
    "ped1": ((7, -2), "standing", "#2e7d32"),
    "cone": ((-2, -6), "-", "#e69f00"),
}
for name, (pos, attr, c) in objs.items():
    ax1.plot(pos[0], pos[1], "o", color=c, ms=11)
    ax1.text(pos[0] + 0.4, pos[1] + 0.5, f"{name}\n({attr})", fontsize=8, color=c)
# 画出 car1-bus 连线与角度
ax1.annotate("", xy=(2, 9), xytext=(6, 4), arrowprops=dict(arrowstyle="->", color="#1a6fa3", lw=1.6, ls="--"))
ax1.text(4.6, 7.2, "θ≈38.7°\n→ front left", fontsize=9, color="#1a6fa3", ha="center",
         bbox=dict(fc="white", ec="#1a6fa3", boxstyle="round,pad=0.25"))
# bus 的 30° 边界线示意（ego 前进方向为基准）
for ang, ls in [(30, "-"), (-30, "-")]:
    rad = np.deg2rad(ang)
    dx, dy = -np.sin(rad), np.cos(rad)
    ax1.plot([6, 6 + dx * 7], [4, 4 + dy * 7], color="#c62828", lw=1, ls=(0, (4, 3)))
ax1.text(6.5, 10.6, "bus 的 front 扇区\n±30° 边界（虚线）", fontsize=8, color="#c62828")

# ---------- 右：四步推理 ----------
ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.set_title("四步推理（答案：moving）", fontsize=11)
steps = [
    ("① 找参照物", "category=bus 且 attribute=stopped\n→ 场景图中唯一节点 bus@(6,4)"),
    ("② 算关系边", "θ=arccos(((B_obj−B_bus)·V_ego)/(范数乘积))\ncar1: (−4,5)→θ≈+38.7°→front left ✓\ncar2: θ≈+100.3°→back left ✗；ped1: θ≈−170.5°→back ✗；cone: θ≈+141.3°→back left ✗"),
    ("③ 过滤候选", "在 front-left-of-bus 的集合 {car1} 里\n找 category=car 且 attribute=moving → car1"),
    ("④ 读状态", "car1 节点的 attribute = moving\n→ 输出答案 moving（30 类闭集之一）"),
]
y0 = 8.9
for i, (t, body) in enumerate(steps):
    y = y0 - i * 1.95
    ax2.add_patch(mpatches.Rectangle((0.2, y - 0.8), 9.6, 1.75, fc="#f5f5f5" if i % 2 == 0 else "white",
                                     ec="#bbb", lw=0.8))
    ax2.text(0.5, y + 0.45, t, fontsize=10.5, weight="bold", ha="left", va="center")
    ax2.text(0.5, y - 0.25, body, fontsize=8.8, ha="left", va="center", linespacing=1.6)

ax2.text(5, 0.55, "断裂点：若检测器把 bus 中心估计偏到 (4.5, 4)，\n"
                  "car1 的 θ 变成 26.6° → 落入 front 扇区 → 候选集变空 → 答错。\n"
                  "关系是硬分箱，30° 边界附近的物体对检测噪声零容忍；\n"
                  "而 one-hop 推理链没有任何回退机制——这就是 GT 上界\n"
                  "（GroundTruth+MCAN H1 88.2%）与检测 baseline（H1 50.6%）\n"
                  "之间 37 个点的鸿沟里，检测误差之外的另一部分。",
         fontsize=9, ha="center", color="#b34a06", linespacing=1.65,
         bbox=dict(fc="#fff8e1", ec="#e0c36a", boxstyle="round,pad=0.4"))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/090-Agent-nuscenes-qa/fig-onehop.png", dpi=150)
print("saved fig-onehop.png")
