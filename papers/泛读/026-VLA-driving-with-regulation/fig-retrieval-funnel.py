# -*- coding: utf-8 -*-
"""Two-level retrieval funnel of the TRR Agent (paper Sec. 3.2, Eqs. 3-5).
Paragraph-level coarse recall with text-embedding-ada-002, then
sentence-level refinement with paraphrase-MiniLM-L6-v2."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("TRR Agent: two-level retrieval  (paragraph -> sentence)", fontsize=13, pad=14)

stages = [
    (0.4,  "Regulatory\ncorpus D\n(state laws, manuals,\ncity rules, cases)", "#e2eff5", "#1a6fa3"),
    (3.0,  "Paragraph retrieval\nE = text-embedding-ada-002\nTop-K FAISS  (Eq. 3)", "#e2eff5", "#1a6fa3"),
    (5.4,  "Threshold filter\nsim >= tau_p = 0.28\n(Eq. 4)", "#f8ecd2", "#9e5c00"),
    (7.6,  "Sentence retrieval\nE' = paraphrase-MiniLM\n-L6-v2, Top-K  (Eq. 5)", "#ebe7f8", "#6a55c2"),
    (9.7,  "Rule snippets Rq\nfed to Reasoning\nAgent", "#dcf2e8", "#0b7a5c"),
]
for i, (x, label, face, edge) in enumerate(stages):
    box = FancyBboxPatch((x - 0.15, 1.2), 1.9, 2.4, boxstyle="round,pad=0.08",
                         facecolor=face, edgecolor=edge, linewidth=2)
    ax.add_patch(box)
    ax.text(x + 0.8, 2.4, label, ha="center", va="center", fontsize=9.5, color="#222")
    if i < len(stages) - 1:
        ax.annotate("", xy=(stages[i + 1][0] - 0.15, 2.4), xytext=(x + 1.75, 2.4),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.6))

ax.text(5.5, 4.25, "coarse: broad coverage", ha="center", fontsize=10, color="#1a6fa3", style="italic")
ax.text(5.5, 0.75, "fine: precision on long-form legal text", ha="center", fontsize=10, color="#6a55c2", style="italic")
ax.text(5.5, 0.35, "Worked values: K = 5,  tau_p = 0.28  (paper Sec. 4.1, 5 seeds)",
        ha="center", fontsize=9.5, color="#555", family="monospace")
plt.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-100/papers/026-VLA-driving-with-regulation/fig-retrieval-funnel.png"
fig.savefig(out, dpi=150)
print("saved", out)
