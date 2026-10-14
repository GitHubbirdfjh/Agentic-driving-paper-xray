"""Self-drawn figure for 134-WM-think2drive: RSSM latent rollout training loop.
Shows: real CARLA data -> encoder q(z|h,x) -> latent state s=(h,z) ->
planner dreams T=15 steps in latent space -> actor/critic update.
Real simulator only used for world-model training, never for planner training."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(14, 5.2))
fig.patch.set_facecolor("#faf9f5")
ax.set_facecolor("#faf9f5")
ax.set_xlim(0, 14); ax.set_ylim(0, 6.4)
ax.axis("off")

def box(x, y, w, h, label, sub, fc, ec, fs=10):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                         fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2 + 0.22, label, ha="center", va="center",
            fontsize=fs, weight="bold", color="#141413")
    ax.text(x + w / 2, y + h / 2 - 0.32, sub, ha="center", va="center",
            fontsize=7.5, color="#3d3d3a")

def arrow(x1, y1, x2, y2, label=None, ls="-", lw=1.6, color="#6e737b"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls))

IN, OP, ALT = "#e2eff5", "#dcf2e8", "#ebe7f8"

# left: real data
box(0.3, 2.2, 2.2, 1.9, "Real CARLA", "128x128 BEV masks\n(privileged info)", IN, "#1a6fa3")
# encoder
box(3.3, 2.2, 2.2, 1.9, "Encoder q(z|h,x)", "posterior:\nsee obs -> judge z", OP, "#0b7a5c")
arrow(2.5, 3.15, 3.3, 3.15)
ax.text(2.9, 3.45, "train WM", ha="center", fontsize=8, color="#0b7a5c", style="italic")

# latent state
box(6.4, 2.2, 2.4, 1.9, "Latent state s=(h,z)", "h: GRU memory\ndeterministic\nz: stochastic", ALT, "#6a55c2")
arrow(5.5, 3.15, 6.4, 3.15)

# dream loop (right side, big rounded area)
ax.add_patch(mpatches.FancyBboxPatch((9.4, 0.7), 4.2, 4.9, boxstyle="round,pad=0.15",
                                     fc="#f5f4ed", ec="#6e737b", lw=1.4, ls=(0, (5, 3))))
ax.text(11.5, 5.15, "DREAM  (no CARLA here)", ha="center", fontsize=10,
        weight="bold", color="#6a55c2")
box(9.9, 3.5, 3.2, 1.1, "Planner  pi(a|s)", "actor: 30 discrete actions", "#f8ecd2", "#9e5c00", fs=9)
box(9.9, 1.15, 3.2, 1.35, "World model p(z|h)\n+ reward/termination", "prior: no obs -> predict z\nrollout T=15 steps", "#f9e3d6", "#b34a06", fs=9)
arrow(11.5, 3.5, 11.5, 2.55, color="#b34a06", lw=2)
ax.text(12.35, 3.05, "a_t", fontsize=9, color="#9e5c00", style="italic")
# loop back arrow
ax.annotate("", xy=(9.9, 4.05), xytext=(9.9, 2.0),
            arrowprops=dict(arrowstyle="->", color="#b34a06", lw=1.6,
                            connectionstyle="angle3,angleA=180,angleB=-90"))
ax.text(9.35, 3.0, "s_{t+1},\nr_t, c_t", fontsize=8, color="#b34a06", ha="center", style="italic")
arrow(8.8, 3.15, 9.4, 3.15, color="#6a55c2", lw=2)
ax.text(9.1, 3.5, "s_t", fontsize=9, color="#6a55c2", style="italic")

ax.text(7.0, 0.95, "planner NEVER sees real CARLA;\nit only drives inside the dream",
        fontsize=9, color="#3d3d3a", style="italic", ha="center")
ax.text(7.0, 0.35, "brick 1 (reset): re-init planner mid-training, keep world model",
        fontsize=9, color="#b34a06", ha="center")

ax.set_title("Think2Drive: train the planner by thinking in the latent world model (RSSM / DreamerV3)",
             fontsize=12, color="#141413", pad=12)
plt.tight_layout()
out = "/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/134-WM-think2drive/fig-rssm.png"
plt.savefig(out, dpi=150, facecolor="#faf9f5")
print("saved", out)
