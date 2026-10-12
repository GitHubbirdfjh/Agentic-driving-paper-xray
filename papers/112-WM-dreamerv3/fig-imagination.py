# DreamerV3 imagination training for actor-critic (Fig. 3b of the paper, redrawn).
# Shows: replay -> world model -> imagined trajectories -> actor/critic updates.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.5, 5.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.5); ax.axis("off")

def box(x, y, w, h, text, sub="", fc="#e2eff5", ec="#1a6fa3", fs=10.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.7)
    ax.add_patch(p)
    cx, cy = x + w/2, y + h/2
    ax.text(cx, cy + (0.28 if sub else 0), text, ha="center", va="center",
            fontsize=fs, weight="bold", color="#111")
    if sub:
        ax.text(cx, cy - 0.52, sub, ha="center", va="center", fontsize=8.4, color="#333")

def arr(a, b, label="", c="#555", dy=0.26, fs=8.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=1.5,
                                 color=c, shrinkA=2, shrinkB=4))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 + dy, label, ha="center", fontsize=fs,
                color=c, style="italic")

ax.text(6.75, 6.1, "DreamerV3 actor-critic learning: policy trained on imagined rollouts, not real env steps",
        ha="center", fontsize=12.5, weight="bold")

box(0.3, 2.4, 2.2, 1.5, "replay buffer", "5M transitions\nreal env (x, a, r, c)", fc="#e2eff5", ec="#1a6fa3")
box(3.2, 2.4, 2.5, 1.5, "world model\n(RSSM)", "encode -> s_t={h_t,z_t}\nloss: L_pred+1.0L_dyn+0.1L_rep", fc="#dcf2e8", ec="#0b7a5c")
box(6.4, 2.4, 2.9, 1.5, "imagination", "15-step rollouts from s_t\nprior p(z|h), no real obs", fc="#f8ecd2", ec="#9e5c00")
box(10.0, 3.3, 3.0, 1.5, "actor  pi(a|s)", "REINFORCE + entropy\nreturn normalized", fc="#ebe7f8", ec="#6a55c2")
box(10.0, 1.2, 3.0, 1.5, "critic  v(R|s)", "symexp twohot,\nlambda-returns", fc="#f9e3d6", ec="#b34a06")

arr((2.5, 3.15), (3.2, 3.15), "batch 16x64", c="#1a6fa3")
arr((5.7, 3.15), (6.4, 3.15), "start states", c="#0b7a5c")
arr((9.3, 3.75), (10.0, 4.05), "actions", c="#6a55c2")
arr((9.3, 2.55), (10.0, 1.95), "returns", c="#b34a06")
arr((11.5, 3.3), (11.5, 2.7), "bootstrap", c="#555")
arr((8.0, 2.4), (8.0, 0.7), "critic replay loss (scale 0.3)", c="#b34a06", dy=-0.1)

box(3.2, 0.35, 9.8, 1.05, "robustness kit (fixed across all domains)",
    "symlog/symexp | 1% categorical unimix | free bits 1 nat | KL balance (dyn:rep) | percentile return norm | critic EMA | AGC + LaProp",
    fc="#f5f4ed", ec="#6e737b", fs=9.5)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/112-wm-dreamerv3/fig-imagination.png", dpi=150, bbox_inches="tight")
print("ok")
