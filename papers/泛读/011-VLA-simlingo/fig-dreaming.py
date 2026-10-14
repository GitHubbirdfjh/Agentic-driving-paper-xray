#!/usr/bin/env python3
"""Action Dreaming 数据生成：同一视觉上下文 -> world-on-rails 离线仿真 -> 多条指令-动作对（safe/unsafe 旗标）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]

IN   = "#1a6fa3"
OUT  = "#9e5c00"
OP   = "#0b7a5c"
LOSS = "#b34a06"
WIRE = "#6e737b"
ALT  = "#6a55c2"
BG   = "#faf9f5"

fig, ax = plt.subplots(figsize=(13.5, 6.4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 6.4)
ax.axis("off")

def box(x, y, w, h, title, lines, color):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25",
                       facecolor="white", edgecolor=color, linewidth=1.6)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.75 - i * 0.38, ln, ha="center", va="center", fontsize=9, color="#141413")

def arrow(x0, y0, x1, y1, label=None, color=WIRE, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                        color=color, linewidth=1.4, linestyle=ls, shrinkB=2)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, label, ha="center", fontsize=8.5, color=color, style="italic")

ax.text(6.75, 6.0, "Action Dreaming：让模型不得不听指令的数据工厂", fontsize=14, weight="bold",
        color="#141413", ha="center")

# 左：输入
box(0.3, 2.6, 3.0, 2.6, "保存的模拟器状态", ["自车位姿、速度", "动态物体未来 10 步轨迹", "专家路径（每 10cm 一点）", "同一视觉上下文"], IN)
# 中：world-on-rails
box(4.3, 2.6, 3.4, 2.6, "world-on-rails 仿真", ["其他 actor 按回放走（无响应）", "自车：运动学自行车模型", "+ PDM-lite 的 PID 控制器", "逐步前向展开备选轨迹"], OP)
# 右：备选模式
box(8.7, 2.6, 4.5, 2.6, "五种备选轨迹模式", ["Objects：朝物体开（可能碰撞）", "Faster：油门 >50% / Slower：刹车", "随机目标速度（0~35 m/s）", "变道（含上人行道/停车道）"], ALT)

arrow(3.3, 3.9, 4.3, 3.9)
arrow(7.7, 3.9, 8.7, 3.9)

# 下：检查与标注
box(1.5, 0.3, 4.2, 1.7, "可行性检查", ["与回放 actor 做碰撞检查", "交规检查（红灯/停车线等）", "-> safe / unsafe 旗标"], LOSS)
box(7.7, 0.3, 4.3, 1.7, "指令-动作对 + 语言标签", ["\"Crash into the cone\" -> 轨迹", "\"可执行/不可执行 + 原因\"", "同一图像对应多条不同指令"], OUT)

arrow(6.0, 2.6, 3.6, 2.0, color=LOSS)
arrow(11.0, 2.6, 9.9, 2.0, color=LOSS)
arrow(5.7, 1.15, 7.7, 1.15, color=WIRE)

# 训练用法
ax.text(6.75, 5.45, "关键：事后从专家动作反推指令 -> 指令可从图像直接猜出 -> 模型不听指令；同一上下文配多条指令 -> 必须听", fontsize=10,
        color=LOSS, ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9e3d6", edgecolor=LOSS, alpha=0.9))

# 训练时的 Dreamer flag
ax.text(6.75, 0.05, "训练：Dreamer flag 开 -> 学\"按指令预测动作\"；flag 关 -> 学\"拒绝不安全指令\"（各 50%）", fontsize=9.5,
        color=WIRE, ha="center", style="italic")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/011-VLA-simlingo/fig-dreaming.png"
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
print("saved", out)
