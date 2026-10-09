"""The three attention blocks of V2XPnP: temporal, self-spatial, multi-agent spatial."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
fig.patch.set_facecolor("white")

# ---------------- (a) temporal attention ----------------
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("(a) Temporal attention\nsame (h, w) across T frames", fontsize=11, weight="bold")
T = 5
for t in range(T):
    x = 0.6 + t * 1.8
    for i in range(3):
        for j in range(3):
            hot = (i == 1 and j == 1)
            c = "#b5d8ec" if hot else "#f2f0e8"
            e = "#1a6fa3" if hot else "#c9c6bb"
            r = patches.Rectangle((x + j * 0.55, 7.2 - i * 0.55), 0.5, 0.5,
                                  facecolor=c, edgecolor=e, lw=1.2 if hot else 0.8)
            ax.add_patch(r)
    ax.text(x + 0.82, 8.6, f"t-{T-1-t}" if t < T - 1 else "t (now)", ha="center", fontsize=8)
    if t < T - 1:
        ax.annotate("", xy=(x + 1.37, 6.35), xytext=(x + 1.68, 6.35),
                    arrowprops=dict(arrowstyle="<->", color="#0b7a5c", lw=1.6))
ax.text(5, 4.9, "query = current frame cell\nkeys/values = same cell in history\n(+ learned time embedding)", ha="center", fontsize=8.5)
ax.text(5, 3.3, "output: F_tem  (T, H, W, C) -> (H, W, C)", ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcf2e8", edgecolor="#0b7a5c"))
ax.text(5, 1.6, "A cell occluded now can 'remember'\nthe vehicle seen 2 frames ago.", ha="center",
        fontsize=8, color="#3d3d3a", style="italic")

# ---------------- (b) self-spatial attention ----------------
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("(b) Self-spatial attention\nmulti-scale windows (2, 4, 8)", fontsize=11, weight="bold")
grids = [(0.5, 2, "P_loc = 2"), (3.6, 4, "P_inter = 4"), (6.7, 8, "P_gl = 8")]
for gx, n, lab in grids:
    s = 2.6
    cell = s / n
    for i in range(n):
        for j in range(n):
            r = patches.Rectangle((gx + j * cell, 5.2 + (n - 1 - i) * cell), cell, cell,
                                  facecolor="#e2eff5", edgecolor="#1a6fa3", lw=0.6)
            ax.add_patch(r)
    ax.text(gx + s / 2, 4.6, lab, ha="center", fontsize=8)
ax.text(5, 3.3, "attention inside each window only\n+ relative position encoding\n+ split attention across scales", ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e2eff5", edgecolor="#1a6fa3"))
ax.text(5, 1.6, "Small window: fine local detail.\nLarge window: long-range context.", ha="center",
        fontsize=8, color="#3d3d3a", style="italic")

# ---------------- (c) multi-agent spatial attention ----------------
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("(c) Multi-agent spatial attention\nheterogeneous V-V / V-I / I-V / I-I", fontsize=11, weight="bold")
pos = {"V1 (ego)": (5, 7.6), "V2": (1.8, 4.4), "I1": (8.2, 4.4), "I2": (5, 2.4)}
edge_cols = {("V1 (ego)", "V2"): "#1a6fa3", ("V2", "V1 (ego)"): "#1a6fa3",
             ("V1 (ego)", "I1"): "#6a55c2", ("I1", "V1 (ego)"): "#6a55c2",
             ("I1", "I2"): "#9e5c00", ("I2", "I1"): "#9e5c00",
             ("V2", "I1"): "#0b7a5c", ("I1", "V2"): "#0b7a5c",
             ("V1 (ego)", "I2"): "#6a55c2", ("I2", "V1 (ego)"): "#6a55c2",
             ("V2", "I2"): "#0b7a5c", ("I2", "V2"): "#0b7a5c"}
pairs = [("V1 (ego)", "V2"), ("V1 (ego)", "I1"), ("V1 (ego)", "I2"),
         ("V2", "I1"), ("I1", "I2"), ("V2", "I2")]
for a, b in pairs:
    ax.annotate("", xy=pos[b], xytext=pos[a],
                arrowprops=dict(arrowstyle="->", color=edge_cols.get((a, b), "#6e737b"),
                                lw=1.6, shrinkA=14, shrinkB=14))
for name, (x, y) in pos.items():
    is_ego = "ego" in name
    circ = patches.Circle((x, y), 0.85, facecolor="#f8ecd2" if is_ego else "#f2f0e8",
                          edgecolor="#b34a06" if is_ego else "#6e737b", lw=2 if is_ego else 1.2)
    ax.add_patch(circ)
    ax.text(x, y, name.replace(" (ego)", "\n(ego)"), ha="center", va="center", fontsize=8, weight="bold" if is_ego else "normal")
ax.text(5, 0.9, "Each edge type owns its own weights W_att.\n"
                "Vehicle and infra sensors differ in height and noise -\n"
                "the model is allowed to trust them differently.",
        ha="center", fontsize=8, color="#3d3d3a", style="italic")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/v2xpnp-xray/fig-attention-blocks.png"
plt.savefig(out, dpi=150)
print("saved", out)
