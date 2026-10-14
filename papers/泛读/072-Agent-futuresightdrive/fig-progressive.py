# -*- coding: utf-8 -*-
"""fig-progressive.py: progressive easy-to-hard generation in pre-training,
and the single unified frame at inference.
Pre-train: Ql (lane dividers, static skeleton) -> Qd (3D boxes, dynamic objects)
-> Qf (full future frame). Inference: one unified frame, then trajectory.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

WIRE = "#6e737b"
IN, IN_F = "#1a6fa3", "#e2eff5"
ALT, ALT_F = "#6a55c2", "#ebe7f8"
OP, OP_F = "#0b7a5c", "#dcf2e8"
OUT, OUT_F = "#9e5c00", "#f8ecd2"

fig, ax = plt.subplots(figsize=(13.5, 6.8))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7); ax.axis("off")

def frame(x, y, w, h, title, color, face, draw_fn):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=face, edgecolor=color, linewidth=1.8)
    ax.add_patch(r)
    ax.text(x + w/2, y + h + 0.12, title, ha="center", fontsize=9.5, weight="bold", color=color)
    draw_fn(x, y, w, h)

def arrow(x1, y1, x2, y2, c=WIRE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.8))

ax.text(6.75, 6.6, "Progressive generation (pre-training): skeleton → objects → full scene",
        fontsize=12.5, weight="bold", ha="center")

def draw_ql(x, y, w, h):
    for x0 in [x + w*0.35, x + w*0.65]:
        ax.plot([x0, x0], [y + 0.2, y + h - 0.2], color="red", lw=2.4)

def draw_qd(x, y, w, h):
    draw_ql(x, y, w, h)
    for (fx, fy, fw, fh) in [(0.55, 0.45, 0.22, 0.28), (0.15, 0.55, 0.18, 0.22)]:
        br = mpatches.Rectangle((x + w*fx, y + h*fy), w*fw, h*fh,
                                facecolor="none", edgecolor="red", linewidth=2)
        ax.add_patch(br)

def draw_qf(x, y, w, h):
    road = plt.Polygon([[x + w*0.25, y], [x + w*0.75, y], [x + w*0.62, y + h], [x + w*0.38, y + h]],
                       closed=True, facecolor="#9aa7b5", edgecolor="none")
    ax.add_patch(road)
    draw_ql(x, y, w, h); draw_qd(x, y, w, h)

frame(0.4, 3.4, 2.6, 1.9, "Ql: lane dividers\n(static skeleton)", IN, IN_F, draw_ql)
arrow(3.0, 4.35, 3.6, 4.35)
frame(3.6, 3.4, 2.6, 1.9, "Qd: 3D boxes\n(dynamic objects)", ALT, ALT_F, draw_qd)
arrow(6.2, 4.35, 6.8, 4.35)
frame(6.8, 3.4, 2.6, 1.9, "Qf: full future frame\n(fine details)", OP, OP_F, draw_qf)

ax.text(5.0, 2.75, "P(Qf | Ql, Qd): details conditioned on physical skeleton — direct full-frame generation violates physics",
        fontsize=9, color=WIRE, ha="center", style="italic")

# ---- inference row ----
ax.text(6.75, 2.1, "Inference: one unified frame, then plan", fontsize=12, weight="bold", ha="center")
frame(0.4, 0.25, 2.6, 1.35, "", OP, OP_F, draw_qf)
ax.text(1.7, 1.95, "unified Q_CoT", ha="center", fontsize=9, weight="bold", color=OP)
arrow(3.0, 0.92, 3.7, 0.92)
box = mpatches.FancyBboxPatch((3.7, 0.35), 3.4, 1.15, boxstyle="round,pad=0.06",
                              facecolor=OUT_F, edgecolor=OUT, linewidth=1.8)
ax.add_patch(box)
ax.text(5.4, 0.92, "inverse dynamics:\nP(Wt | It, Q_CoT) → waypoints", ha="center", va="center",
        fontsize=9.5, color=OUT, weight="bold")
arrow(7.1, 0.92, 7.8, 0.92)
box2 = mpatches.FancyBboxPatch((7.8, 0.35), 5.0, 1.15, boxstyle="round,pad=0.06",
                               facecolor="white", edgecolor=WIRE, linewidth=1.3)
ax.add_patch(box2)
ax.text(10.3, 0.92, "world model:  It → Q_CoT  (imagine future)\nplanner:  (It, Q_CoT) → Wt  (infer action)",
        ha="center", va="center", fontsize=9, color=WIRE)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/072-Agent-futuresightdrive/fig-progressive.png",
            dpi=150, bbox_inches="tight")
print("saved")
