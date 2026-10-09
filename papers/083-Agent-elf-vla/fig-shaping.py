# -*- coding: utf-8 -*-
"""Policy Shaping f(x) = x / (x + γ)，γ = 0.1 的几何形状。
关键点：它是凹的，在 x≈0 处最陡 —— 越是低概率 token，权重被抬得越多。
标注几个具体数值，让读者看到"低概率但正确"的 token 如何被从梯度里捞回来。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

gamma = 0.1
x = np.linspace(0, 1, 400)
f = x / (x + gamma)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(x, f, color="#0b7a5c", linewidth=2.5, label=r"$f(x)=\frac{x}{x+\gamma},\ \gamma=0.1$")
ax.plot(x, x, color="#9aa0a8", linewidth=1.4, linestyle="--", label="恒等线（无 shaping）")
ax.fill_between(x, x, f, where=(f > x), color="#0b7a5c", alpha=0.12)

# 标注几个关键点
for px in [0.02, 0.1, 0.5]:
    py = px / (px + gamma)
    ax.plot(px, py, "o", color="#0b7a5c", markersize=7, zorder=5)
    ax.annotate(f"x={px}\nf={py:.2f}\n放大 {py/px:.1f}×",
                xy=(px, py), xytext=(px + 0.16, py + 0.13),
                fontsize=9.5, color="#0b5345",
                arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="#dcf2e8", ec="#0b7a5c"))

ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.set_xlabel("token 在当前策略下的概率 $x = \\pi_\\theta(o^{fb}|q_{base})$", fontsize=11.5)
ax.set_ylabel("进入目标函数的权重 f(x)", fontsize=11.5)
ax.set_title("Policy Shaping 的几何动作：凹函数把低概率 token 往上托", fontsize=13, pad=12)
ax.legend(fontsize=10.5, loc="lower right")

ax.text(0.62, 0.30,
        "读法：f 在 0 附近斜率 = 1/γ = 10，\n"
        "概率 0.02 的 token 权重被抬到 0.17；\n"
        "概率 0.5 的 token 只从 0.5 抬到 0.83。\n"
        "修正样本里那些\"策略本来想不到、\n"
        "但老师证明是对的\"token，\n"
        "就靠这段陡坡留在梯度里，不被冲掉。",
        fontsize=10.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", fc="#faf9f5", ec="#c9c6bb"))

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/083-Agent-elf-vla/fig-shaping.png",
            dpi=150, bbox_inches="tight")
print("saved fig-shaping.png")
