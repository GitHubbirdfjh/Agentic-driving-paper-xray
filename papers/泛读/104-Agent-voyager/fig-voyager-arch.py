# -*- coding: utf-8 -*-
"""Voyager 三组件循环架构示意图:自动课程 → 技能检索 → 迭代代码生成 → 自我验证 → 技能库."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_BG = "#f7f5ef"
C_CUR = "#3b7dd8"
C_LIB = "#6a55c2"
C_ACT = "#d97b2e"
C_CRI = "#2e9e6b"

fig, ax = plt.subplots(figsize=(13.5, 8.5))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 9.5)
ax.axis("off"); fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)

def box(x, y, w, h, text, color, fs=9.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=color,
                       ec="none", alpha=0.93)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color="white", weight="bold", linespacing=1.5,
            fontfamily="Noto Sans CJK JP")

def arrow(x1, y1, x2, y2, label="", color="#555", rad=0.0, fs=8.5,
          lx=None, ly=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                        color=color, linewidth=1.7,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2 if lx is None else lx,
                (y1+y2)/2 + 0.28 if ly is None else ly,
                label, ha="center", fontsize=fs, color=color,
                fontfamily="Noto Sans CJK JP",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85))

ax.text(6.75, 9.0, "Voyager: 三组件终身学习循环", ha="center", fontsize=15,
        weight="bold", fontfamily="Noto Sans CJK JP")
ax.text(6.75, 8.5, "动作空间 = 可执行代码(Mineflayer JS);LLM 全程黑盒调用,不做梯度训练",
        ha="center", fontsize=10, color="#444", fontfamily="Noto Sans CJK JP")

# layout: curriculum top, library left, iterative prompting center-right
box(4.6, 6.6, 4.3, 1.3, "① 自动课程 (GPT-4)\n看状态提新任务:由易到难\n温度 0.1 保多样性", C_CUR)
box(0.5, 3.4, 3.6, 2.2, "② 技能库 (向量数据库)\nkey: 技能描述 embedding\nvalue: 可执行代码\n只增不减、组合复用", C_LIB, fs=9)
box(5.4, 3.4, 3.6, 2.2, "③ 迭代代码生成 (GPT-4)\n检索 top-5 相关技能\n→ 生成代码 → 执行\n→ 环境反馈/报错 → 修正", C_ACT, fs=9)
box(10.0, 3.4, 3.0, 2.2, "自我验证 (GPT-4\n当裁判)\n做成了吗?\n没成给 critique", C_CRI, fs=9)

# arrows forming the loop (labels placed in empty space, never inside boxes)
arrow(6.75, 6.6, 7.05, 5.6, "新任务", C_CUR, lx=6.05, ly=6.1)      # curriculum -> action
arrow(4.1, 4.5, 5.4, 4.5, "检索 top-5", C_LIB, ly=4.85)            # library -> action
arrow(9.0, 4.9, 10.0, 4.9, "执行代码", C_ACT, ly=5.2)             # action -> critic
arrow(11.3, 5.6, 8.2, 5.6, "失败 critique → 修正重来", C_CRI, rad=0.3,
      lx=9.75, ly=6.35)                                            # critic -> action (fail loop)
arrow(11.5, 3.4, 11.5, 2.5, "通过:入库", C_CRI, lx=12.45, ly=2.95)  # critic down
arrow(11.5, 2.2, 2.3, 2.2, "技能入库,回课程要新任务", C_LIB, ly=2.5)  # bottom bus
arrow(2.3, 2.2, 2.3, 3.4, "", C_LIB)                              # up into library
arrow(5.8, 5.6, 5.8, 6.6, "卡住 4 轮 → 换任务", "#b34a06",
      lx=4.75, ly=6.15)                                            # stuck -> new task

# bottom notes
ax.text(6.75, 1.15, "三类反馈:①环境反馈(缺 2 块木板) ②执行报错(acacia_axe 不存在) ③自我验证的 critique",
        ha="center", fontsize=9.5, color="#333", fontfamily="Noto Sans CJK JP",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb"))
ax.text(6.75, 0.45, "消融结论:去课程发现物 -93% / 去自验证 -73% / GPT-3.5 换 GPT-4 代码能力差 5.7×",
        ha="center", fontsize=9.5, color="#333", fontfamily="Noto Sans CJK JP",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/104-agent-voyager/fig-voyager-arch.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print("saved", out)
