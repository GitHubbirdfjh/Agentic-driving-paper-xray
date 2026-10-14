# -*- coding: utf-8 -*-
"""fig-toolkit.py: the four Vision Toolkit tools of DriveAgent-R1.
Each tool: what it asks for, what it returns, what human habit it mimics.
Code names vs paper names are noted (they differ).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OP, OP_F = "#0b7a5c", "#dcf2e8"
WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(13.5, 6.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.4); ax.axis("off")

def card(x, title, code_name, paper_name, params, returns, mimic, y=0.7):
    r = mpatches.FancyBboxPatch((x, y), 3.05, 4.9, boxstyle="round,pad=0.08",
                                facecolor="white", edgecolor=OP, linewidth=1.8)
    ax.add_patch(r)
    ax.text(x + 1.525, y + 4.35, title, ha="center", fontsize=11, weight="bold", color=OP)
    ax.text(x + 1.525, y + 3.95, f"paper: “{paper_name}”\ncode: {code_name}",
            ha="center", fontsize=7.5, color=WIRE)
    ax.plot([x + 0.2, x + 2.85], [y + 3.6, y + 3.6], color=WIRE, lw=0.8)
    ax.text(x + 0.2, y + 3.3, "asks:", fontsize=8.5, weight="bold", color="#3d3d3a", va="top")
    ax.text(x + 0.2, y + 3.05, params, fontsize=8, color="#3d3d3a", va="top")
    ax.text(x + 0.2, y + 2.05, "returns:", fontsize=8.5, weight="bold", color="#3d3d3a", va="top")
    ax.text(x + 0.2, y + 1.8, returns, fontsize=8, color="#3d3d3a", va="top")
    ax.text(x + 0.2, y + 1.05, "human habit:", fontsize=8.5, weight="bold", color="#3d3d3a", va="top")
    ax.text(x + 0.2, y + 0.8, mimic, fontsize=8, color=WIRE, va="top", style="italic")

card(0.25, "1 · Retrieve View",
     "get_view_with_frame", "Retrieve View",
     'view_index ∈ {front,\nfront_left, front_right,\nback, back_left, back_right}\nframe_index ∈ {current,\n1s_ago … 5s_ago}',
     "the requested image\n(encoded into Ik)",
     "glancing at a mirror\nbefore a lane change")

card(3.55, "2 · RoI Inspection",
     "RoI_Inspection", "RoI Inspection",
     "view_index + bbox [x0,y0,x1,y1]\n(absolute pixels) +\ndescription\n\n→ crops from the HIGH-RES\nsource, not the encoder tokens",
     "magnified crop\n(re-encoded, new detail)",
     "squinting / zooming in\non a far traffic light")

card(6.85, "3 · Depth Estimation",
     "depth_estimation", "Depth Estimation",
     'view_index\n\n(Depth Anything V2\nViT-L, run on demand)',
     "monocular depth map\n→ relative distances,\nroad layout",
     "judging “how far is\nthat barrier gate”")

card(10.15, "4 · 3D Object Detection",
     "detect_3d_objects", "3D Object Detection",
     'view_index + object_text\n(open-vocabulary;\narbitrary new classes)\n\n(DetAny3D server)',
     "3D boxes: position,\nsize, orientation",
     "noticing the pedestrian\nthe dark hid")

ax.text(6.75, 6.0, "Vision Toolkit: four ways to go get evidence instead of guessing",
        fontsize=12.5, weight="bold", ha="center")
ax.text(0.25, 0.35, "max 3 calls per trajectory · each result re-encoded by the Vision Encoder and appended to history Hk",
        fontsize=8.5, color=WIRE)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/071-Agent-driveagent-r1/fig-toolkit.png",
            dpi=150, bbox_inches="tight")
print("saved")
