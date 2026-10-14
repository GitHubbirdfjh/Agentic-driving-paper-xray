# -*- coding: utf-8 -*-
"""ReAct 核心机制示意图:增广动作空间 A^ = A ∪ L,交错 Thought/Action/Observation 轨迹.
一条 Mini 轨迹走完 ReAct 循环,左侧 Reason-only / Act-only 在哪一步失败一并标出."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_THOUGHT = "#3b7dd8"   # 蓝:推理
C_ACT     = "#d97b2e"   # 橙:动作
C_OBS     = "#2e9e6b"   # 绿:观测
C_BG      = "#f7f5ef"

fig, ax = plt.subplots(figsize=(13, 9.6))
ax.set_xlim(0, 13); ax.set_ylim(-3.2, 10)
ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)

def box(x, y, w, h, text, color, fs=9.5, tc="white", style="round,pad=0.25"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=color, ec="none",
                       alpha=0.95)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight="bold", linespacing=1.45,
            fontfamily="Noto Sans CJK JP")
    return b

def arrow(x1, y1, x2, y2, label=None, color="#555555", ls="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        color=color, linewidth=1.6, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, ha="center", fontsize=8,
                color=color, fontfamily="Noto Sans CJK JP")

ax.text(6.5, 9.55, "ReAct: 交错的 Thought / Action / Observation 轨迹",
        ha="center", fontsize=15, weight="bold", fontfamily="Noto Sans CJK JP")
ax.text(6.5, 9.05, "动作空间被增广为 Â = A ∪ L:语言思考(Thought)成为一种不触动环境、只改写上下文的动作",
        ha="center", fontsize=10, color="#444444", fontfamily="Noto Sans CJK JP")

# ---------- main vertical trajectory ----------
traj = [
    ("Thought 1", "分解问题:\n\"Milhouse 以谁命名?\n只需搜 Milhouse\"", C_THOUGHT),
    ("Action 1", "Search[Milhouse]", C_ACT),
    ("Obs 1", "Milhouse 是辛普森\n的人物……", C_OBS),
    ("Thought 2", "段落没说命名由来\n→ 换关键词查", C_THOUGHT),
    ("Action 2", "Lookup[以命名]", C_ACT),
    ("Obs 2", "以尼克松\n(中间名 Milhous)命名", C_OBS),
    ("Thought 3", "答案是尼克松", C_THOUGHT),
    ("Action 3", "Finish[尼克松]", C_ACT),
]
x0, y0, bw, bh, gap = 5.1, 7.9, 2.8, 0.78, 0.28
for i, (tag, txt, col) in enumerate(traj):
    y = y0 - i * (bh + gap)
    box(x0, y, bw, bh, f"{tag}\n{txt}", col, fs=8.6)
    if i > 0:
        arrow(x0 + bw/2, y + bh, x0 + bw/2, y + bh + gap, color="#777777")

# context band (left): context grows
ax.text(3.9, 7.9, "上下文 c_t 随轮次\n单调增长\n(只追加,不改写)", ha="center", va="top",
        fontsize=8.5, color="#333333", fontfamily="Noto Sans CJK JP",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbbbbb"))
ax.text(3.9, 4.9, "c_{t+1} = (c_t ; â_t)\nThought 不产生\n环境观测", ha="center", va="top",
        fontsize=8.5, color="#333333", fontfamily="Noto Sans CJK JP",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbbbbb"))

# ---------- right side: the two synergies ----------
box(8.7, 6.9, 3.6, 1.15, "reason to act\n推理→指引下一步动作\n(分解目标/异常处理)", "#6a55c2", fs=9)
box(8.7, 5.3, 3.6, 1.15, "act to reason\n动作→为推理取回事实\n(搜维基/看环境)", "#0b7a5c", fs=9)
arrow(7.9, 7.45, 8.7, 7.45, color="#6a55c2", rad=0.15)
arrow(8.7, 5.6, 7.9, 5.1, color="#0b7a5c", rad=0.15)

# ---------- bottom: where baselines break ----------
ax.text(6.5, -1.25, "基线在同一问题上的断点", ha="center", fontsize=11, weight="bold",
        fontfamily="Noto Sans CJK JP")
box(0.4, -2.85, 5.9, 1.3, "CoT (Reason-only): 全程闭眼推理\n无环境交互 → 事实幻觉\n(56% 的失败是幻觉,表 2)", "#b34a06", fs=8.8)
box(6.7, -2.85, 5.9, 1.3, "Act (Act-only): 只会机械搜词条\n不会把\"搜不到\"翻成新策略\n(推理错误率高于 ReAct)", "#9e5c00", fs=8.8)

plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/102-agent-react/fig-react-loop.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print("saved", out)
