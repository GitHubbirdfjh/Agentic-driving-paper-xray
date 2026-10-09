#!/usr/bin/env python3
"""GVQA 有向无环图示意：感知->预测->规划->行为->轨迹，边上是 "Context:" 前缀拼接的父问答。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

IN   = "#1a6fa3"  # 输入
OUT  = "#9e5c00"  # 输出/预测
OP   = "#0b7a5c"  # 带参数的变换
WIRE = "#6e737b"
BG   = "#faf9f5"

fig, ax = plt.subplots(figsize=(12, 5.2))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.2)
ax.axis("off")

stages = [
    ("P1\nPerception", 1.2,  "What are the\nimportant objects?", IN),
    ("P2\nPrediction", 3.7,  "Where will the\npedestrian go?", IN),
    ("P3\nPlanning",   6.2,  "What are safe\nactions?", IN),
    ("B\nBehavior",    8.5,  "(Bsp, Bst)\ngo straight + slow", OUT),
    ("M\nMotion",     10.6,  "{(x1,y1)...}\ntrajectory tokens", OUT),
]

for title, x, desc, color in stages:
    box = FancyBboxPatch((x - 0.85, 2.2), 1.7, 1.6, boxstyle="round,pad=0.08,rounding_size=0.25",
                         facecolor=color, edgecolor=color, linewidth=1.5, alpha=0.16)
    ax.add_patch(box)
    ax.text(x, 3.35, title, ha="center", va="center", fontsize=12, weight="bold", color=color)
    ax.text(x, 2.62, desc, ha="center", va="center", fontsize=9.5, color="#141413", linespacing=1.35)

# 任务级边：上一阶段的 QA 作为 Context: 拼进下一阶段的问题
for i in range(len(stages) - 1):
    x0 = stages[i][1] + 0.85
    x1 = stages[i + 1][1] - 0.85
    arr = FancyArrowPatch((x0, 3.0), (x1, 3.0), arrowstyle="-|>", mutation_scale=14,
                          color=WIRE, linewidth=1.3, shrinkA=0, shrinkB=2)
    ax.add_patch(arr)
    ax.text((x0 + x1) / 2, 3.32, "Context: <parent QA>", ha="center", va="bottom",
            fontsize=9, style="italic", color=WIRE)

# 对象级边示例：行人的感知节点 -> 轿车的规划节点
ax.annotate("", xy=(5.55, 2.2), xytext=(3.0, 1.55),
            arrowprops=dict(arrowstyle="-|>", color=OP, linewidth=1.3,
                            linestyle=(0, (4, 3)), mutation_scale=14, shrinkB=2))
ax.text(4.3, 1.45, "object-level edge: pedestrian P1 -> sedan P3", fontsize=9,
        color=OP, ha="center", style="italic")
ax.text(4.3, 0.95, "对象级边：不同对象的问答节点之间也有依赖（图 1 例子）",
        fontsize=9.5, color="#141413", ha="center")

ax.text(6, 4.55, "GVQA: 每个节点是一个问答对 v=(q, a)，边是逻辑依赖", fontsize=13,
        weight="bold", color="#141413", ha="center")
ax.text(6, 4.15, "G=(V, E) 有向无环图；推理时按拓扑序 P1->P2->P3->B->M 逐轮问答", fontsize=10.5,
        color=WIRE, ha="center")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/012-VLA-drivelm/fig-graph.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
