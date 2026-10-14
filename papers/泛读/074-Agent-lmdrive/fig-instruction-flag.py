# -*- coding: utf-8 -*-
"""LMDrive 拒绝误导指令的机制：指令完成标志 + 1 秒标注技巧。
只讲一件事：模型如何学会对不合理指令说"不" —— 不是靠推理，
而是靠把误导指令发生后约 1 秒的数据标成 completed，让模型
学会"指令完成后继续安全驾驶"这个模式。
数字来自论文 4.3 节与附录 D。
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
_font = [f for f in font_manager.fontManager.ttflist if f.name == "Noto Sans CJK JP"]
assert _font, "no CJK font"
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

IN = "#1a6fa3"; OUT = "#9e5c00"; OP = "#0b7a5c"; WIRE = "#6e737b"; LOSS = "#b34a06"

fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
ax.text(5, 4.25, "误导指令是怎么被拒绝的：没有推理器，只有一个二分类标志",
        ha="center", fontsize=13, weight="bold", color="#141413")

# 时间线
t = np.linspace(0.5, 9.5, 400)
ax.plot([0.5, 9.5], [2.6, 2.6], color=WIRE, lw=2.5)
for x, lab in [(0.5, "t=0"), (9.5, "t")]:
    ax.text(x, 2.35, lab, ha="center", fontsize=10, color=WIRE)

# 误导指令区间
ax.axvspan(3.0, 4.6, ymin=0.52, ymax=0.72, color="#f9e3d6", alpha=0.9)
ax.text(3.8, 3.62, "误导指令持续 1–2 秒\n（如单车道上说\"换到左车道\"）",
        ha="center", fontsize=10.5, color=LOSS, weight="bold")

# 标注：误导指令后约 1 秒标 completed
ax.plot([3.0, 4.0], [3.02, 3.02], color=LOSS, lw=5)
ax.text(3.5, 2.82, "训练标注：completed=1", ha="center", fontsize=9.5, color=LOSS)

# 模型输出：路点 + 完成标志
ax.text(6.9, 3.55, "模型在每帧同时输出：\n① 未来路点（l1 监督）\n② 指令完成标志（交叉熵监督）",
        ha="left", fontsize=11, color=OUT,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8ecd2", edgecolor=OUT))
ax.annotate("", xy=(6.2, 3.0), xytext=(4.6, 2.9),
            arrowprops=dict(arrowstyle="->", color=OUT, lw=1.8))

ax.text(5, 1.85,
        "学到的行为：看到与场景矛盾的指令 → 把完成标志置 1 → 按场景继续安全驾驶\n"
        "直到下一条正确指令到来。这不是语义理解误导，而是学会了\"忽略并继续\"的模式。",
        ha="center", fontsize=11, color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#faf9f5", edgecolor="#deddd8"))

ax.text(5, 0.62,
        "代价：误导指令在 LangAuto 里只占约 5% 且随机出现 —— 论文没有报告\n"
        "模型是否会把正常指令也误判为完成（误拒率），这是评估里缺的一块。",
        ha="center", fontsize=10.5, color="#3d3d3a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f4ed", edgecolor=WIRE))

ax.text(0.5, 0.12, "图：LMDrive 拒绝误导指令的机制 —— 1 秒标注 + 完成标志二分类（论文 4.3 节）",
        fontsize=10, color="#73726c")
plt.tight_layout(pad=0.3)
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/074-Agent-lmdrive/fig-instruction-flag.png",
            dpi=150, bbox_inches="tight")
print("saved fig-instruction-flag.png")
