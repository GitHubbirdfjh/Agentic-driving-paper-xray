# -*- coding: utf-8 -*-
"""SayCan 核心机制:任务 grounding × 世界 grounding 的双路打分相乘选技能.
示意性小例子(数值经构造,仅用于展示机制形状):指令"帮我拿可乐",
LLM 最想选的技能被 affordance 一票否决."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_BG = "#f7f5ef"
C_SAY = "#3b7dd8"
C_CAN = "#2e9e6b"
C_BOTH = "#6a55c2"

skills = ["去商店买可乐", "找可乐罐", "拿起可乐罐", "把可乐给我", "用吸尘器打扫"]
p_say = np.array([0.55, 0.80, 0.75, 0.70, 0.10])   # LLM: 对指令有用程度
p_can = np.array([0.02, 0.90, 0.05, 0.60, 0.01])   # affordance: 当前状态下能做成
p_both = p_say * p_can

fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), sharey=True)
fig.patch.set_facecolor(C_BG)
titles = ["Say: p(ℓ|i)\nLLM 打分:对指令有用吗", "Can: p(c|s,ℓ)\naffordance:现在做得成吗",
          "相乘选最大\nπ = argmax p(c|s,ℓ)·p(ℓ|i)"]
colors = [C_SAY, C_CAN, C_BOTH]
vals = [p_say, p_can, p_both]

for ax, title, color, v in zip(axes, titles, colors, vals):
    ax.set_facecolor(C_BG)
    y = np.arange(len(skills))
    bars = ax.barh(y, v, color=color, alpha=0.85, edgecolor="none", height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(skills, fontsize=10.5)
    ax.set_xlim(0, 1.02); ax.set_title(title, fontsize=11.5, weight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    for b, val in zip(bars, v):
        ax.text(val + 0.02, b.get_y() + b.get_height()/2, f"{val:.2f}",
                va="center", fontsize=9.5, color="#333")
    ax.invert_yaxis()

# 标注关键 veto
axes[0].annotate("LLM 最想去商店买\n(合理但不可执行)",
                 xy=(0.55, 0), xytext=(0.72, 1.6),
                 fontsize=9, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
axes[1].annotate("场景里没商店/没吸尘器\n→ affordance 接近 0",
                 xy=(0.02, 0), xytext=(0.30, 2.2),
                 fontsize=9, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
axes[2].text(0.36, 1, "胜出:找可乐罐\n0.80 × 0.90", color="white", weight="bold",
             ha="center", va="center", fontsize=10)

fig.suptitle("SayCan: 双路打分相乘选技能(示意数值,展示机制形状)",
             fontsize=14, weight="bold", y=1.02)
fig.text(0.5, -0.01,
         "LLM 只管\"对指令有用\",affordance(value 函数)只管\"当前状态做得成\";两者相乘才是\"有用且做得成\"。被一票否决的是 LLM 的幻觉,不是执行失败。",
         ha="center", fontsize=10, color="#555")
plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/103-agent-saycan/fig-saycan-factorize.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print("saved", out)
