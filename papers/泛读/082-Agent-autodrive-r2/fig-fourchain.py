"""AutoDrive-R2: 四步逻辑链 + 自反思回路（论文图1管线的理解重绘）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2, C3, C4 = "#1a6fa3", "#0b7a5c", "#9e5c00", "#6a55c2"
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc=color, ec=color, alpha=0.22, lw=1.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color="#141413", weight="bold")

def arrow(x1, y1, x2, y2, color="#3d3d3a", ls="-", lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 color=color, lw=lw, linestyle=ls, mutation_scale=15))

steps = [
    (0.3, "1. Observation\n观察", "图像里有什么：\n车道、障碍物、信号灯", C1),
    (3.5, "2. Calculation\n计算", "运动学方程算未来位置：\nx(t+1)=x+v·δt+½a·δt²", C2),
    (6.7, "3. Logic\n逻辑", "交规综合：红灯？\n行人？碰撞？", C3),
    (9.9, "4. Reflection\n自反思", "回查：假设成立吗？\n速度可达吗？要修正吗？", C4),
]
for x, title, desc, c in steps:
    box(x, 2.6, 2.9, 1.9, title + "\n\n" + desc, c, fs=9.5)
arrow(3.2, 3.55, 3.5, 3.55); arrow(6.4, 3.55, 6.7, 3.55); arrow(9.6, 3.55, 9.9, 3.55)

# 自反思回路：从 Reflection 指回 Calculation/Logic
ax.annotate("", xy=(7.6, 2.55), xytext=(11.2, 2.55),
            arrowprops=dict(arrowstyle="-|>", color=C4, lw=2.2,
                            linestyle=(0, (4, 3)), mutation_scale=15,
                            connectionstyle="arc3,rad=0.35"))
ax.text(9.4, 1.55, "发现矛盾 → 回去重算", fontsize=10, color=C4, weight="bold",
        ha="center", style="italic")

box(0.3, 0.3, 6.1, 1.1, "输入：前视图像 + 过去3秒自车状态（位置/速度/加速度/方向盘转角）",
    "#6e737b", fs=10)
box(6.7, 0.3, 6.0, 1.1, "输出：<think>四步链</think><answer>(x1,y1),…,(x6,y6)</answer>",
    "#6e737b", fs=10)
arrow(6.4, 0.85, 6.7, 0.85)

ax.set_title("R² 的第二个 R：Reflection（自反思）——想完之后再回头检查一遍",
             fontsize=13, weight="bold", pad=14)
fig.text(0.5, -0.01, "单轮推理是“想到就说”；这里的第四步强制模型做 backward-check："
         "预测位置需要的平均速度历史加速度给得了吗？前面那堵墙是真的过不去吗？\n"
         "附录的 Aha Moment 例子显示，模型会在这一步自己推翻前三步的结论（“Oh, I see…”）。",
         ha="center", fontsize=9.5, color="#3d3d3a")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/082-Agent-autodrive-r2/fig-fourchain.png",
            dpi=150, bbox_inches="tight")
print("saved fig-fourchain.png")
