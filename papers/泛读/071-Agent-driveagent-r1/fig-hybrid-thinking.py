# -*- coding: utf-8 -*-
"""fig-hybrid-thinking.py: DriveAgent-R1 hybrid-thinking architecture.
Top path: M_text (efficient text-only reasoning). Bottom path: M_tool
(active perception: interleave thoughts with tool calls into the Vision Toolkit).
One idea per figure: the agent first picks a thinking mode, then either goes
straight to actions or loops through evidence gathering.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IN, OUT, OP, ALT, WIRE = "#1a6fa3", "#9e5c00", "#0b7a5c", "#6a55c2", "#6e737b"
IN_F, OUT_F, OP_F, ALT_F = "#e2eff5", "#f8ecd2", "#dcf2e8", "#ebe7f8"

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.4); ax.axis("off")

def box(x, y, w, h, face, edge, txt, fs=9.5, tcol="black", lw=1.6, ls="-"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tcol, weight="bold" if lw >= 1.6 else "normal", zorder=5)

def arrow(x1, y1, x2, y2, c=WIRE, w=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=w))

# ---- input block ----
box(0.3, 3.0, 2.2, 1.5, IN_F, IN, "Multimodal input\nI0 (front view) + T0\n(speed, navigation)")
arrow(2.5, 3.75, 3.05, 3.75)

# ---- mode selector ----
box(3.05, 2.9, 1.75, 1.7, "#f5f4ed", "#73726c",
    "Mode select\n<think_text>\n  or\n<think_tool>", fs=9)
arrow(4.8, 4.35, 4.8, 5.9); arrow(4.8, 3.15, 4.8, 1.9)

# ---- top: M_text path ----
ax.text(4.4, 6.45, "M_text  (routine scene)", fontsize=11, color=IN, weight="bold")
box(5.15, 5.35, 1.7, 1.0, IN_F, IN, "Text M-CoT\nT1: describe →\nreason → predict", fs=8.5)
arrow(6.85, 5.85, 7.35, 5.85)
box(7.35, 5.35, 2.4, 1.0, OUT_F, OUT, "A = (a1..a4)\n8s meta-action seq", fs=9)

# ---- bottom: M_tool path ----
ax.text(4.4, 2.55, "M_tool  (complex / uncertain scene)", fontsize=11, color=ALT, weight="bold")
box(5.15, 0.45, 1.9, 1.15, ALT_F, ALT, "LM Decoder\nthought Tk\n+ tool call Tk", fs=8.5)
arrow(7.05, 1.02, 7.6, 1.02)
box(7.6, 0.45, 2.2, 1.15, OP_F, OP, "Execute Tk →\nnew visual evidence Ik", fs=8.5)
# vision toolkit strip
box(10.05, 0.2, 2.55, 1.65, "#ffffff", OP,
    "Vision Toolkit\n\n1 Retrieve View\n2 RoI Inspection\n3 Depth Estimation\n4 3D Object Detection", fs=8, lw=1.2)
# loop back arrow: Ik -> LM decoder
ax.annotate("", xy=(6.1, 1.62), xytext=(8.75, 1.62),
            arrowprops=dict(arrowstyle="->", color=ALT, lw=1.8,
                            connectionstyle="arc3,rad=0.55"))
ax.text(7.4, 2.15, "Ik re-encoded,\nappended to Hk", fontsize=8, color=ALT, ha="center")
arrow(9.8, 1.02, 10.05, 1.02, c=OP)
ax.text(11.32, 1.05, "Hk = Hk-1 ⊕ Tk ⊕ Ik\nk < K", fontsize=8, color=OP, ha="center")
# exit to actions
box(7.35, 3.3, 2.4, 0.9, OUT_F, OUT, "A = (a1..a4)\n8s meta-action seq", fs=9)
arrow(6.5, 1.6, 7.9, 3.3, c=ALT)
ax.text(6.6, 2.62, "done /\nmax K=3", fontsize=8, color=ALT)

# ---- meta action legend ----
box(0.3, 0.45, 2.2, 1.35, "#ffffff", WIRE,
    "meta-action at =\n(speed, traj)\nVs: Stop / Keep /\nAccel / Decel ×\nStraight / Left / Right", fs=8, lw=1.2)

ax.text(6.5, 7.05, "DriveAgent-R1 hybrid thinking: one mode token, then either straight reasoning or think-while-seeing",
        fontsize=12, weight="bold", ha="center")
ax.text(0.3, 2.32, "scene complexity decides", fontsize=8.5, color="#73726c")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/071-Agent-driveagent-r1/fig-hybrid-thinking.png",
            dpi=150, bbox_inches="tight")
print("saved")
