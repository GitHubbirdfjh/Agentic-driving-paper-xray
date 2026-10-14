# -*- coding: utf-8 -*-
"""NaviLLM schema token layout for one navigation decision step (minimal labels)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#faf9f5"
INK = "#141413"
GREY = "#6e737b"
C_TASK = "#1a6fa3"
C_HIST = "#6a55c2"
C_OBS = "#0b7a5c"
C_HINT = "#9e5c00"
C_LLM = "#a8457c"
C_OUT = "#b34a06"
FILL = {C_TASK: "#e2eff5", C_HIST: "#ebe7f8", C_OBS: "#dcf2e8",
        C_HINT: "#f8ecd2", C_LLM: "#f6e3ec", C_OUT: "#f9e3d6"}

fig, ax = plt.subplots(figsize=(12, 6.8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1200)
ax.set_ylim(0, 680)
ax.axis("off")

def box(x, y, w, h, color, title, sub):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=10,rounding_size=12",
                       facecolor=FILL[color], edgecolor=color, linewidth=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 34, title, ha="center", va="top",
            fontsize=18, color=color, weight="bold")
    ax.text(x + w / 2, y + h - 66, sub, ha="center", va="top",
            fontsize=13.5, color=INK, family="monospace")

def arrow(x1, y1, x2, y2, color=GREY, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=lw, color=color,
                                 shrinkA=2, shrinkB=5))

BX, BW, BH = 30, 560, 118
box(BX, 520, BW, BH, C_TASK, "Task", "dialog / instruction text")
box(BX, 384, BW, BH, C_HIST, "History", "[Emb(1), h1, ... Emb(t), ht]")
box(BX, 248, BW, BH, C_OBS, "Observation", "[Emb(0), s0, ... Emb(3), s3]")
box(BX, 112, BW, BH, C_HINT, "Output Hint", "how to answer (text)")

ax.text(310, 662, "input token stream: text tokens + visual tokens",
        ha="center", fontsize=14, color=GREY, style="italic")

for yy in (579, 443, 307, 171):
    ax.plot([BX + BW, BX + BW + 30], [yy, yy], color=GREY, lw=1.5)
ax.plot([BX + BW + 30, BX + BW + 30], [171, 579], color=GREY, lw=1.5)
arrow(BX + BW + 30, 375, 700, 375, GREY, 2)

llm = FancyBboxPatch((700, 312), 250, 126,
                     boxstyle="round,pad=10,rounding_size=14",
                     facecolor=FILL[C_LLM], edgecolor=C_LLM, linewidth=2.4)
ax.add_patch(llm)
ax.text(825, 406, "Vicuna-7B", ha="center", va="center",
        fontsize=20, color=C_LLM, weight="bold")
ax.text(825, 364, "fine-tuned", ha="center", va="center",
        fontsize=13, color=GREY, style="italic")
ax.text(825, 340, "autoregressive", ha="center", va="center",
        fontsize=13, color=GREY, style="italic")

arrow(956, 375, 1020, 375, C_OUT, 2)
ax.text(1055, 400, '" 2 "', ha="center", va="center", fontsize=24,
        color=C_OUT, weight="bold", family="monospace")
ax.text(1055, 360, "one number token", ha="center", va="center",
        fontsize=12.5, color=GREY, style="italic")
ax.text(1055, 322, "-> move to viewpoint 2", ha="center", fontsize=13.5,
        color=INK, va="center")
ax.text(1055, 292, "history appends s2", ha="center", fontsize=13.5,
        color=C_HIST, va="center", family="monospace")

ax.text(30, 40, "s_i: R1024 -> R4096 (LLM dim)   Emb(k): R4096 learned",
        fontsize=12.5, color=GREY, family="monospace")
ax.text(30, 14, "n = 36 views;  t = history rounds (15/15/30/20)",
        fontsize=12.5, color=GREY, family="monospace")

plt.tight_layout(pad=0.8)
out_path = "/home/hatch/workspace/your_files/paper-xray-40/vln/navillm-xray/fig-schema.png"
plt.savefig(out_path, dpi=150, facecolor=BG)
print("saved:", out_path)
