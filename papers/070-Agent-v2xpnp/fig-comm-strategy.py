"""One-step vs multi-step communication in V2XPnP (numbers from Table 2 of the paper)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.patch.set_facecolor("white")

# ---------------- left: multi-step ----------------
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("Multi-step communication", fontsize=13, weight="bold")

frames = ["t-4", "t-3", "t-2", "t-1", "t"]
for k, f in enumerate(frames):
    y = 8.2 - k * 1.15
    r = patches.FancyBboxPatch((0.4, y), 2.6, 0.95, boxstyle="round,pad=0.08",
                               facecolor="#f8ecd2", edgecolor="#9e5c00", lw=1.2)
    ax.add_patch(r)
    ax.text(1.7, y + 0.48, f"frame {f}\nH x W x C", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(5.6, y + 0.48), xytext=(3.15, y + 0.48),
                arrowprops=dict(arrowstyle="->", color="#6e737b", lw=1.4))
ax.text(1.7, 9.45, "Agent 2 (helper)", ha="center", fontsize=10, weight="bold")

ego = patches.FancyBboxPatch((5.8, 2.6), 3.6, 5.2, boxstyle="round,pad=0.12",
                             facecolor="#e2eff5", edgecolor="#1a6fa3", lw=1.6)
ax.add_patch(ego)
ax.text(7.6, 7.35, "Ego agent", ha="center", fontsize=10, weight="bold")
ax.text(7.6, 6.15, "receive 5 packets\naccumulate sequence\ntemporal fusion\nAT THE EGO", ha="center", fontsize=8.5)
ax.text(7.6, 4.85, "5 transmissions", ha="center", fontsize=9, weight="bold", color="#b34a06")
ax.text(7.6, 4.0, "5 x 0.269 Mb", ha="center", fontsize=9, color="#b34a06")
ax.text(7.6, 3.3, "AP 68.2 / EPA 43.0", ha="center", fontsize=9, weight="bold")
ax.text(5, 0.9, "T separate transmissions; ego fuses across time.\n"
                "Packets can be delayed or lost; agent may leave the range.",
        ha="center", fontsize=8, color="#3d3d3a", style="italic")

# ---------------- right: one-step ----------------
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("One-step communication (V2XPnP)", fontsize=13, weight="bold")

for k, f in enumerate(frames):
    y = 8.2 - k * 1.15
    r = patches.FancyBboxPatch((0.4, y), 2.0, 0.85, boxstyle="round,pad=0.08",
                               facecolor="#f2f0e8", edgecolor="#6e737b", lw=1.0)
    ax.add_patch(r)
    ax.text(1.4, y + 0.43, f"frame {f}", ha="center", va="center", fontsize=8)
ax.text(1.4, 9.45, "Agent 2 (helper)", ha="center", fontsize=10, weight="bold")

tf = patches.FancyBboxPatch((2.75, 4.4), 2.3, 2.6, boxstyle="round,pad=0.12",
                            facecolor="#dcf2e8", edgecolor="#0b7a5c", lw=1.6)
ax.add_patch(tf)
ax.text(3.9, 6.5, "temporal fusion", ha="center", fontsize=9, weight="bold")
ax.text(3.9, 5.9, "AT THE AGENT", ha="center", fontsize=8.5)
ax.text(3.9, 5.25, "(temporal +\nself-spatial attn)", ha="center", fontsize=8)
ax.annotate("", xy=(2.75, 5.7), xytext=(2.42, 5.7),
            arrowprops=dict(arrowstyle="->", color="#6e737b", lw=1.4))

cond = patches.FancyBboxPatch((2.75, 2.5), 2.3, 1.5, boxstyle="round,pad=0.1",
                              facecolor="#ebe7f8", edgecolor="#6a55c2", lw=1.6)
ax.add_patch(cond)
ax.text(3.9, 3.25, "one condensed map\nF' : H x W x C", ha="center", fontsize=9, weight="bold")
ax.annotate("", xy=(6.6, 3.25), xytext=(5.15, 3.25),
            arrowprops=dict(arrowstyle="->", color="#0b7a5c", lw=2.2))

ego = patches.FancyBboxPatch((6.8, 2.6), 2.8, 5.2, boxstyle="round,pad=0.12",
                             facecolor="#e2eff5", edgecolor="#1a6fa3", lw=1.6)
ax.add_patch(ego)
ax.text(8.2, 7.35, "Ego", ha="center", fontsize=10, weight="bold")
ax.text(8.2, 6.5, "multi-agent\nspatial fusion", ha="center", fontsize=8.5)
ax.text(8.2, 4.9, "1 transmission", ha="center", fontsize=9, weight="bold", color="#0b7a5c")
ax.text(8.2, 4.3, "0.269 Mb", ha="center", fontsize=9, color="#0b7a5c")
ax.text(8.2, 3.5, "AP 71.6 / EPA 48.2", ha="center", fontsize=9, weight="bold")
ax.text(5, 0.9, "Each agent fuses its own history first;\n"
                "ego only fuses across agents. 5x less bandwidth, higher accuracy.",
        ha="center", fontsize=8, color="#3d3d3a", style="italic")

plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/multi-agent-driving/v2xpnp-xray/fig-comm-strategy.png"
plt.savefig(out, dpi=150)
print("saved", out)
