# -*- coding: utf-8 -*-
"""两阶段记忆检索示意:向量KNN粗排 -> LLM模糊重排;以及自反思优化式(1)的几何含义."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- 左图:两阶段检索 ----
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("两阶段经验记忆检索", fontsize=12, weight="bold", pad=10)

def box(ax, x, y, w, h, text, fc, fs=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec="#333", lw=1.1)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1a1a1a", linespacing=1.35)

box(ax, 0.3, 5.3, 3.4, 1.1, "当前场景\n(检测/预测/地图/自车状态)", "#dcebf5")
box(ax, 6.3, 5.3, 3.4, 1.1, "经验库: N条\n(环境信息, 历史轨迹)", "#efe6f7")
ax.text(5.0, 6.35, "各自编码\n成向量", ha="center", fontsize=8.5, color="#555")
a = FancyArrowPatch((3.7, 5.85), (4.6, 5.85), arrowstyle="-|>", mutation_scale=12, color="#333"); ax.add_patch(a)
a = FancyArrowPatch((6.3, 5.85), (5.4, 5.85), arrowstyle="-|>", mutation_scale=12, color="#333"); ax.add_patch(a)

box(ax, 3.1, 3.6, 3.8, 1.0, "Stage-1: 向量空间KNN\n取Top-K (快,但编码器泛化弱)", "#fef3d8", fs=8.5)
a = FancyArrowPatch((5.0, 5.3), (5.0, 4.6), arrowstyle="-|>", mutation_scale=12, color="#333"); ax.add_patch(a)

for i, t in enumerate(["候选(i)", "候选(j)", "候选(k)"]):
    box(ax, 0.6 + i * 3.1, 2.1, 2.7, 0.9, t, "#f5f5f5", fs=8.5)
a = FancyArrowPatch((5.0, 3.6), (5.0, 3.0), arrowstyle="-|>", mutation_scale=12, color="#333"); ax.add_patch(a)

box(ax, 2.3, 0.5, 5.4, 1.2, "Stage-2: LLM模糊重排\n把K个候选+当前场景一起喂给LLM,\n让它按\"相关性\"排序,取第1名", "#e7f4e4", fs=8.5)
a = FancyArrowPatch((5.0, 2.1), (5.0, 1.7), arrowstyle="-|>", mutation_scale=12, color="#333"); ax.add_patch(a)
ax.text(5.0, 0.18, "输出: 最相似历史场景 + 它的轨迹(相似度分数,如0.77)", ha="center", fontsize=8.5, color="#333")

# ---- 右图:自反思优化 ----
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("自反思: 碰撞检查 + 轨迹重优化 (式1)", fontsize=12, weight="bold", pad=10)

# road
ax.add_patch(plt.Rectangle((0.5, 2.2), 9, 2.6, fc="#f0f0f0", ec="#999"))
ax.plot([0.5, 9.5], [3.5, 3.5], "--", color="#bbb", lw=1)
# ego start
ax.add_patch(Circle((1.5, 3.5), 0.35, fc="#1a6fa3", ec="#333"))
ax.text(1.5, 2.75, "自车", ha="center", fontsize=9, color="#1a6fa3")
# obstacle
ax.add_patch(plt.Rectangle((5.6, 3.1), 1.4, 0.8, fc="#b34a06", ec="#333"))
ax.text(6.3, 2.75, "障碍物", ha="center", fontsize=9, color="#b34a06")

# planned trajectory (colliding) - dashed orange through obstacle
tx = np.linspace(1.5, 8.8, 60)
ty = 3.5 + 0.15 * np.sin((tx - 1.5) * 1.2)
ax.plot(tx, ty, "--", color="#9e5c00", lw=2)
ax.plot(tx[::10], ty[::10], "o", color="#9e5c00", ms=5)
ax.text(8.9, 3.9, "τ0: LLM初版轨迹\n(穿过障碍物)", fontsize=8.5, color="#9e5c00")

# refined trajectory - green solid curving around
tx2 = np.linspace(1.5, 8.8, 60)
ty2 = 3.5 + 0.9 * np.exp(-((tx2 - 6.3) ** 2) / 2.5)
ax.plot(tx2, ty2, "-", color="#0b7a5c", lw=2.2)
ax.plot(tx2[::10], ty2[::10], "s", color="#0b7a5c", ms=5)
ax.text(8.9, 4.9, "τ*: 重优化轨迹\n(绕开,贴近τ0)", fontsize=8.5, color="#0b7a5c")

ax.text(5.0, 0.6, "min  λ1||τ-τ0||² + λ2·Fcol(τ)\n第一项: 别偏离原轨迹太远   第二项: 碰撞代价压到零",
        ha="center", fontsize=9.5, color="#333",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fef3d8", ec="#999"))
ax.text(5.0, 1.75, "collision_check(τ0) 发现碰撞 → 触发优化", ha="center", fontsize=9, color="#b34a06",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f9e3d6", ec="#b34a06"))

plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/113-Agent-agentdriver/fig-memory-reflection.png", dpi=150, bbox_inches="tight")
print("saved fig-memory-reflection.png")
