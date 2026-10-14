# DreamerV3 RSSM (Recurrent State-Space Model) with the RAMP reuse point.
# Equation numbers follow the Nature 2025 / arXiv:2301.04104v2 text.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 6.4))
ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")

def box(x, y, w, h, text, sub="", fc="#e2eff5", ec="#1a6fa3", fs=10.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.7)
    ax.add_patch(p)
    cx, cy = x + w/2, y + h/2
    ax.text(cx, cy + (0.28 if sub else 0), text, ha="center", va="center",
            fontsize=fs, weight="bold", color="#111")
    if sub:
        ax.text(cx, cy - 0.52, sub, ha="center", va="center", fontsize=8.4, color="#333")

def arr(a, b, label="", c="#555", dy=0.24):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=1.5,
                                 color=c, shrinkA=2, shrinkB=4))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 + dy, label, ha="center", fontsize=8.8,
                color=c, style="italic")

# --- layout: three time steps t-1, t, t+1 left->right, posterior path vs prior path ---
ax.text(7, 7.75, "DreamerV3 RSSM: one forward step (Eq. 1), and where RAMP reuses it",
        ha="center", fontsize=13.5, weight="bold")

# observations
box(0.3, 2.6, 1.7, 1.3, "o_t", "camera / lidar", fc="#e2eff5", ec="#1a6fa3")
# posterior encoder q(z|h,x)
box(2.7, 2.6, 2.6, 1.3, "encoder q(z_t | h_t, o_t)", "CNN/MLP -> categorical\n32 x 64 codes", fc="#ebe7f8", ec="#6a55c2")
# GRU sequence model
box(6.1, 2.6, 2.6, 1.3, "GRU  f(h_t | h_{t-1}, z_{t-1}, a_{t-1})", "8192 units, 8 blocks", fc="#dcf2e8", ec="#0b7a5c")
# prior dynamics
box(9.5, 2.6, 2.4, 1.3, "prior p(z_t | h_t)", "MLP -> categorical\nused in imagination", fc="#f8ecd2", ec="#9e5c00")
# model state
box(6.1, 5.4, 2.6, 1.1, "model state s_t = {h_t, z_t}", "actor + critic see this", fc="#f5f4ed", ec="#6e737b")

arr((2.0, 3.25), (2.7, 3.25), "encode")
arr((5.3, 3.25), (6.1, 3.25), "z_t sampled", c="#6a55c2")
arr((8.7, 3.55), (9.5, 3.55), "h_t", c="#0b7a5c")
arr((7.4, 3.9), (7.4, 5.4), "s_t", c="#6e737b")
# decoder/reward/continue heads
box(6.1, 0.5, 2.6, 1.1, "heads: decoder / reward / continue", "p(x|h,z), p(r|h,z), p(c|h,z)", fc="#f9e3d6", ec="#b34a06")
arr((7.4, 2.6), (7.4, 1.6), "train")

# imagination arrow from prior
box(9.5, 5.4, 2.4, 1.1, "imagine 15 steps", "roll out p(z|h), no o", fc="#f8f4ed", ec="#9e5c00")
arr((10.7, 3.9), (10.7, 5.4), "", c="#9e5c00")

# RAMP reuse callout
p = FancyBboxPatch((2.55, 4.25), 3.05, 1.0, boxstyle="round,pad=0.05",
                   fc="#fff3d6", ec="#9e5c00", lw=2.0, linestyle=(0, (4, 3)))
ax.add_patch(p)
ax.text(4.08, 4.98, "RAMP reuses this", ha="center", fontsize=9.5, weight="bold", color="#9e5c00")
ax.text(4.08, 4.62, "posterior q(z_t|h_t,o_t): risk eval\nwithout a second encoder pass",
        ha="center", fontsize=8.2, color="#333")
ax.add_patch(FancyArrowPatch((4.08, 4.25), (4.35, 3.95), arrowstyle="-|>",
                             mutation_scale=13, lw=1.4, color="#9e5c00"))

# KL balance annotation
ax.text(7, 0.12, "L = L_pred + 1.0*L_dyn + 0.1*L_rep ;  L_dyn/L_rep = KL with stop-gradient on one side + free bits (clip below 1 nat); 1% uniform mix",
        ha="center", fontsize=8.6, color="#333", style="italic")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/112-wm-dreamerv3/fig-rssm.png", dpi=150, bbox_inches="tight")
print("ok")
