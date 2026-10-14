# -*- coding: utf-8 -*-
"""Self-drawn figure: DriveDreamer-2 customized traffic simulation pipeline.

Data flow: user text -> finetuned LLM + function library -> python script ->
trajectory arrays -> HDMap generator (ControlNet diffusion conditioned on the
trajectory map) -> BEV HDMap -> project to image plane -> structured
conditions (HDMap seq + projected 3D boxes) + CLIP text -> UniMVM video gen.
The key design: background is generated CONDITIONED ON foreground trajectories.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.5, 7.0))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 7.4)
ax.axis("off")

ANN = "#6e737b"
C1, E1 = "#ebe7f8", "#6a55c2"   # LLM stage
C2, E2 = "#f8ecd2", "#9e5c00"   # HDMap stage
C3, E3 = "#dcf2e8", "#0b7a5c"   # video stage
C0, E0 = "#dcebf5", "#1a6fa3"   # input

ax.text(6.75, 7.0, "DriveDreamer-2 定制交通仿真: 先前景轨迹, 再以轨迹为条件生成背景 HDMap",
        ha="center", fontsize=13.5, weight="bold")

def box(x, y, w, h, fc, ec, title, lines, tsize=12):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.8))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", fontsize=tsize, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.25, lines, ha="center", va="center",
            fontsize=8.8, color="#3d3d3a")

def arrow(x0, y0, x1, y1, c="#3d3d3a"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))

# 1. user query
box(0.3, 4.3, 2.2, 1.6, C0, E0, "用户输入",
    "“雨天, 有辆车加塞”\n纯自然语言", 11)
# 2. LLM
box(3.1, 4.3, 3.0, 1.6, C1, E1, "微调 LLM (GPT-3.5)",
    "函数库 18 个原语:\ncut_in / forward / braking …\n输出 Python 脚本", 11)
arrow(2.5, 5.1, 3.1, 5.1)
# 3. trajectories
box(6.7, 4.3, 2.6, 1.6, C1, E1, "轨迹数组",
    "ego_trajs, obj_trajs\n每条 NUM_POINT 个 (x, y, v)\n+ 碰撞检查 safe_dis", 11)
arrow(6.1, 5.1, 6.7, 5.1)

# 4. HDMap generator
box(9.9, 4.3, 3.3, 1.6, C2, E2, "HDMap 生成器",
    "SD2.1 + ControlNet\n条件: BEV 轨迹图 Tb\n目标: 三通道 HDMap\n车道线 / 分隔线 / 斑马线", 11)
arrow(9.3, 5.1, 9.9, 5.1, E2)

# key annotation: conditioning direction
ax.text(11.55, 3.95, "背景以前景为条件生成:\n同一组轨迹可采样出多张\n合法 HDMap (论文图 4)",
        ha="center", fontsize=8.5, color=E2,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f9f3e8", ec=E2))

# 5. projection + conditions
box(3.1, 1.55, 3.0, 1.6, C2, E2, "投影到图像平面",
    "BEV HDMap → 六视图 HDMap 序列\n轨迹 → 3D 框 → 投影成框图\n二值化 + 骨架提取", 11)
arrow(11.55, 4.3, 11.55, 3.6, E2)
arrow(9.9, 3.6, 4.6, 3.6, E2)
arrow(4.6, 3.6, 4.6, 3.15, E2)

# 6. UniMVM
box(6.7, 1.55, 3.4, 1.6, C3, E3, "UniMVM 视频生成 (SVD)",
    "条件: HDMap 序列 + 3D 框图 + CLIP 文本\n输出: 六视图全景视频\nT=8 帧, 256×448×6", 11)
arrow(6.1, 2.35, 6.7, 2.35)

# 7. output
box(10.7, 1.55, 2.5, 1.6, C3, E3, "多视角视频",
    "FID 11.2 / FVD 55.7\n(首帧条件, nuScenes)", 11)
arrow(10.1, 2.35, 10.7, 2.35)

ax.text(0.3, 1.0,
        "承重设计: 轨迹与 HDMap 分开生成, 但 HDMap 生成时以轨迹为条件 ——\n"
        "避免“车开在车道线外面”这类前后矛盾。代价: 误差级联,\n"
        "轨迹不合理时 HDMap 生成器没有纠错机制 (论文未讨论)。",
        fontsize=9.2, color=ANN, va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f5f4ed", ec="#deddd8"))
ax.text(0.3, 0.28, "注意: 官方仓库只放出了视频生成推理代码, LLM 函数库与 HDMap 生成器均未开源。",
        fontsize=9, color="#b34a06", va="top")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/060-WM-drivedreamer-2/fig-pipeline.png"
plt.savefig(out, dpi=150)
print("saved", out)
