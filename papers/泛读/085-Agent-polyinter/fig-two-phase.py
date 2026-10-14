"""Fig: which parameters are trainable in Phase I vs Phase II.
Frozen = gray, trainable = green (op role color).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OP, OPF, WIRE = "#0b7a5c", "#dcf2e8", "#6e737b"
GRAY_F, GRAY_L = "#eceae4", "#9aa0a8"
BG = "#faf9f5"

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
fig.patch.set_facecolor(BG)

components = [
    ("encoders\n(all agents)", "frozen", "frozen"),
    ("detection head\n(ego)", "frozen", "frozen"),
    ("channel selection\nmodule", "train", "frozen"),
    ("spatial attention\nmodule", "train", "frozen"),
    ("general prompt G", "train", "frozen"),
    ("specific prompt Si\n(new neighbor)", "train", "train"),
    ("resizer\n(1x1 conv)", "train", "train"),
]

for ax, phase, title in zip(axes,
                            [["train", "train"], ["frozen", "train"]],
                            ["Phase I: base model training\n(ego-neb1-neb2 jointly)",
                             "Phase II: generalization\n(new neighbor type)"]):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 7); ax.set_ylim(0, 1.6); ax.axis("off")
    ax.set_title(title, fontsize=12)
    for i, (name, s1, s2) in enumerate(components):
        state = s1 if phase == ["train", "train"] and title.startswith("Phase I") else None
        # decide state by phase title
        if title.startswith("Phase I"):
            state = s1
        else:
            state = s2
        fc = OPF if state == "train" else GRAY_F
        ec = OP if state == "train" else GRAY_L
        tc = OP if state == "train" else WIRE
        r = mpatches.FancyBboxPatch((i * 0.97 + 0.08, 0.45), 0.88, 0.7,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax.add_patch(r)
        ax.text(i * 0.97 + 0.52, 0.98, name, ha="center", va="center",
                fontsize=9.5, color=tc, linespacing=1.3)
        ax.text(i * 0.97 + 0.52, 0.62,
                "trainable" if state == "train" else "frozen",
                ha="center", va="center", fontsize=9,
                style="italic", color=tc)
    # arrow showing inheritance
axes[0].annotate("", xy=(7.05, 0.8), xytext=(6.95, 0.8))
fig.text(0.5, 0.08, "Phase II inherits the whole interpreter; only the new neighbor's specific prompt + resizer are tuned",
         ha="center", fontsize=11, color=WIRE)

h1 = mpatches.Patch(facecolor=OPF, edgecolor=OP, label="trainable")
h2 = mpatches.Patch(facecolor=GRAY_F, edgecolor=GRAY_L, label="frozen")
axes[1].legend(handles=[h1, h2], loc="upper right", fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/085-Agent-polyinter/fig-two-phase.png",
            dpi=150, facecolor=BG)
print("saved")
