# -*- coding: utf-8 -*-
"""Fig: RICES retrieval-based in-context example selection (Dolphins Sec 3.3).
Each video-instruction-answer triplet is embedded by VideoMAE (video) and
text-embedding-ada-002 (text); top-k=3 nearest neighbors by cosine similarity
in TEXT space become in-context examples (paper: text retrieval worked better).
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)

# ---------- left: embedding space retrieval ----------
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")
ax1.set_title("在文本 embedding 空间里按余弦相似度检索（k=3）", fontsize=11.5, color=INK, pad=10)

rng = np.random.default_rng(11)
# background triplets (gray)
bx, by = rng.uniform(0.5, 9.5, 40), rng.uniform(0.5, 9.5, 40)
ax1.scatter(bx, by, s=55, c="#e8e6df", edgecolors=C_WIRE, linewidths=1.0, alpha=0.9, zorder=2)
# current instance (star, center)
ax1.scatter([5.0], [5.0], s=340, c=C_OUT, marker="*", edgecolors="#7a4a00", linewidths=1.2, zorder=5)
ax1.text(5.0, 4.15, "当前样本 Zi\n（视频+指令+答案）", fontsize=9, color=C_OUT, ha="center", weight="bold")
# 3 nearest in text space
nb = [(6.7, 6.4), (3.6, 6.1), (5.9, 3.2)]
for i, (nx, ny) in enumerate(nb):
    ax1.scatter([nx], [ny], s=200, c=C_IN_F, edgecolors=C_IN, linewidths=2, zorder=4)
    ax1.text(nx, ny, f"{i+1}", fontsize=10, color=C_IN, ha="center", va="center", weight="bold")
    ax1.annotate("", xy=(nx, ny), xytext=(5.0, 5.0),
                arrowprops=dict(arrowstyle="-", color=C_IN, lw=1.3, ls=(0, (3, 2))))
ax1.text(7.9, 8.7, "选中的 3 个 in-context 样本：\n与当前样本“行为相似”\n（如都是跟车/让行）",
         fontsize=9, color=C_IN, ha="center",
         bbox=dict(fc="#ffffff", ec=C_IN, boxstyle="round,pad=0.35"))
ax1.text(5.0, 0.55, "R(Zi)：VideoMAE 编视频、text-embedding-ada-002 编文本，\n"
         "分别取 top-k；论文发现按文本相似度检索效果更好，\n实际只用了文本检索的 k=3 个样本。",
         fontsize=9, color=INK, ha="center",
         bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.35"))

# ---------- right: prompt template ----------
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
ax2.set_title("拼成 few-shot prompt：只有绿色 token 算 loss", fontsize=11.5, color=INK, pad=10)

y = 8.6
blocks = [
    ("任务定义（task-level instruction）", C_WIRE, "#f5f4ed", False),
    ("样本 1：<image> 视频1 <endofchunk> 指令1 <answer> 答案1", C_IN, C_IN_F, False),
    ("样本 2：<image> 视频2 <endofchunk> 指令2 <answer> 答案2", C_IN, C_IN_F, False),
    ("样本 3：<image> 视频3 <endofchunk> 指令3 <answer> 答案3", C_IN, C_IN_F, False),
    ("当前实例：<image> 视频 <endofchunk> 指令 <answer> …", C_OP, C_OP_F, True),
]
for i, (t, col, fcol, is_cur) in enumerate(blocks):
    yy = y - i*1.28
    ax2.add_patch(patches.FancyBboxPatch((0.5, yy-0.55), 9.0, 1.05, boxstyle="round,pad=0.03",
                                        fc=fcol, ec=col, lw=1.7 if is_cur else 1.3))
    ax2.text(5, yy, t, fontsize=9.2, color=col, ha="center", va="center",
             weight="bold" if is_cur else "normal")
ax2.annotate("", xy=(5, 2.2), xytext=(5, 2.9),
             arrowprops=dict(arrowstyle="->", color=C_OP, lw=2))
ax2.text(5, 1.75, "交叉熵 loss 只算 <answer> 之后\n当前实例答案的 token（绿色）",
         fontsize=9.5, color=C_OP, ha="center", weight="bold",
         bbox=dict(fc="#ffffff", ec=C_OP, boxstyle="round,pad=0.35"))
ax2.text(5, 0.55, "几何动作：把“找相似历史”这件事，\n翻译成“在 prompt 里预置 3 个最像的问答对”——\n模型不用更新权重，靠上下文学会新指令的格式。",
         fontsize=9, color=INK, ha="center",
         bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.35"))

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/076-Agent-dolphins/fig-rices.png"
plt.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
