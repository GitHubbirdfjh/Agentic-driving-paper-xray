"""Barbell adaptation vs LLaMA-Adapter-V2 original: where adapters sit in 32 layers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8

OP, ALT, WIRE, IN = "#0b7a5c", "#6a55c2", "#6e737b", "#1a6fa3"

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)

configs = [
    ("LLaMA-Adapter-V2 (original)", {i: "adapter" for i in range(1, 32)},
     "adapters in every layer except layer 1;\nthey dominate, context gets drowned"),
    ("Barbell (Hint-AD)", {**{i: "front" for i in range(1, 13)},
                            **{i: "end" for i in range(24, 32)}},
     "context at layer 1; adapters only\nfront (2-13) + end (25-32)"),
]

for ax, (title, amap, note) in zip(axes, configs):
    ax.set_xlim(0, 3.4)
    ax.set_ylim(-0.5, 32.5)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_ylabel("LLaMA-2-7B layer")
    ax.set_xticks([])
    ax.invert_yaxis()
    for i in range(32):
        y = i
        if i == 0:
            c, ec, lab = "#e2eff5", IN, "layer 1: context tokens"
        elif amap.get(i) == "front":
            c, ec, lab = "#dcf2e8", OP, None
        elif amap.get(i) == "end":
            c, ec, lab = "#dcf2e8", OP, None
        elif amap.get(i) == "adapter":
            c, ec, lab = "#f9e3d6", "#b34a06", None
        else:
            c, ec, lab = "#f5f4ed", WIRE, None
        r = mp.Rectangle((0.9, y - 0.42), 1.2, 0.84, fc=c, ec=ec, lw=1.1)
        ax.add_patch(r)
        if i == 0:
            ax.text(2.35, y, lab, fontsize=7.5, color=IN, va="center")
    # legend-ish annotations
    ax.text(2.35, 6, "front adapters:\nhelp read context", fontsize=7.5, color=OP, va="center")
    ax.text(2.35, 28, "end adapters:\nlanguage fine-tune", fontsize=7.5, color=OP, va="center")
    if "original" in title:
        ax.text(2.35, 16, "adapters everywhere\n(context drowned)", fontsize=7.5, color="#b34a06", va="center")
    else:
        ax.text(2.35, 18, "middle layers:\nno adapters", fontsize=7.5, color=WIRE, va="center")
    ax.text(1.5, 33.6, note, fontsize=7.5, color=WIRE, ha="center", style="italic")
    ax.set_yticks([0, 7, 15, 23, 31])
    ax.set_yticklabels(["1", "8", "16", "24", "32"])

fig.suptitle("Barbell adaptation: pin the adapters to both ends, leave the middle alone",
             fontsize=11.5, weight="bold", y=0.98)
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/077-Agent-hint-ad/fig-barbell.png",
            dpi=150, bbox_inches="tight")
print("saved fig-barbell.png")
