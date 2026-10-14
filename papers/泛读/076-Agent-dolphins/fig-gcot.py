# -*- coding: utf-8 -*-
"""Fig: Dolphins GCoT (Grounded Chain of Thought) three-step pipeline.
Concrete example from the paper's Appendix Table 2 (surfboard / 1926).
Step 1: describe image. Step 2: pin the question object to its bounding box.
Step 3: reason. Then append "So the answer is {answer}".
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams["font.family"] = ["Noto Serif CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#faf9f5"; INK = "#141413"
C_IN, C_IN_F = "#1a6fa3", "#e2eff5"
C_OP, C_OP_F = "#0b7a5c", "#dcf2e8"
C_OUT, C_OUT_F = "#9e5c00", "#f8ecd2"
C_ALT, C_ALT_F = "#6a55c2", "#ebe7f8"
C_WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(12.5, 5.4))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.4); ax.axis("off")
ax.set_title("GCoT：三步把“推理”钉在图像坐标上（例子取自论文附录 Table 2，真实例子）",
             fontsize=12.5, color=INK, pad=12)

# ---- input: image with bbox ----
ax.add_patch(patches.Rectangle((0.3, 0.9), 2.6, 2.6, fc="#e8e4d8", ec=C_WIRE, lw=1.4))
ax.text(1.6, 3.15, "海滩 / 冲浪者", fontsize=9, color=C_WIRE, ha="center")
# surfboard bbox
ax.add_patch(patches.Rectangle((0.7, 1.15), 1.8, 0.9, fill=False, ec="#b34a06", lw=2))
ax.text(1.6, 1.5, "surfboard\n[0.388,0.418,1.254,0.977]", fontsize=8, color="#b34a06", ha="center", weight="bold")
# person bbox
ax.add_patch(patches.Rectangle((1.15, 1.9), 1.1, 1.35, fill=False, ec=C_ALT, lw=1.6, ls="--"))
ax.text(1.7, 2.95, "person", fontsize=7.5, color=C_ALT, ha="center")
ax.text(1.6, 0.55, "输入：图像 + captions\n+ 物体标注（含 bbox）\n问题：这件运动器材何时发明？",
        fontsize=8.8, color=INK, ha="center",
        bbox=dict(fc="#ffffff", ec=C_WIRE, boxstyle="round,pad=0.3"))

steps = [
    (4.35, "第 1 步：描述图像", "“画面里一个穿黑色潜水服的人\n抱着冲浪板穿过沙滩。”",
     "用 caption 概括全局，\n给 LLM 一个场景锚点", C_IN, C_IN_F),
    (7.0, "第 2 步：定位问题对象", "“问题里的‘运动器材’指冲浪板，\n它的 bbox 是 [0.388, 0.418, 1.254, 0.977]。”",
     "关键一步：把文本指称\n“钉”到图像坐标上", C_ALT, C_ALT_F),
    (9.65, "第 3 步：推理", "“1926 年美国冲浪者 Tom Blake\n发明了第一块空心冲浪板。”",
     "只有需要推理的题才有这步；\n直接看图能答的题跳过", C_OP, C_OP_F),
]
for x, title, body, note, col, fcol in steps:
    ax.add_patch(patches.FancyBboxPatch((x-1.15, 0.7), 2.3, 3.6, boxstyle="round,pad=0.04",
                                        fc=fcol, ec=col, lw=1.7))
    ax.text(x, 3.95, title, fontsize=10, color=col, ha="center", weight="bold")
    ax.text(x, 3.35, body, fontsize=8.6, color=INK, ha="center", va="top")
    ax.text(x, 1.55, note, fontsize=8.2, color=C_WIRE, ha="center", va="top")
    ax.annotate("", xy=(x-1.25, 2.5), xytext=(x-1.75, 2.5),
                arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.7))

ax.annotate("", xy=(11.05, 2.5), xytext=(10.8, 2.5),
            arrowprops=dict(arrowstyle="->", color=C_WIRE, lw=1.7))
ax.text(11.55, 2.5, "…So the answer is 1926", fontsize=9.5, color=C_OUT, ha="center", weight="bold",
        bbox=dict(fc=C_OUT_F, ec=C_OUT, boxstyle="round,pad=0.35"))

ax.text(6.4, 0.22, "为什么叫 Grounded：CoT 的每一步都被迫引用图像上的具体位置，模型不能凭空推理——"
        "这是给 VLM 补“细粒度视觉理解”课的手段。", fontsize=9, color=INK, ha="center",
        bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.3"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/076-Agent-dolphins/fig-gcot.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
