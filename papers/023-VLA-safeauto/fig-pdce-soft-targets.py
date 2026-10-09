"""Self-drawn figure for the SafeAuto xray doc.

Reproduces the PDCE soft-target construction from the paper's Figure 4
pseudo-code (Appendix B), for the concrete number "2.69" with sigma=0.35,
the value used in the paper's experiments.

Panel (a): digit-level soft target distributions D(mu_i, sigma) for each of
the three digit positions, as bar charts over digits 0-9. The ground-truth
digit of each position is highlighted.
Panel (b): place-level weights w_i = prod_{j<i} D(mu_j, sigma)[mu_j],
the cumulative-product decay that gives earlier (more significant) digits
higher influence.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

SIGMA = 0.35
NUMBER = "2.69"
DIGITS = [int(c) for c in NUMBER if c != "."]

# --- paper Figure-4 pseudo-code, verbatim logic ---
distribution_dict = {}
for num in range(10):
    prob = np.array([norm(num, SIGMA).cdf(i + 0.5) - norm(num, SIGMA).cdf(i - 0.5)
                     for i in range(10)])
    prob = prob / prob.sum()
    distribution_dict[str(num)] = prob

tgt_probs, weights, w = [], [], 1.0
for d in DIGITS:
    digit_probs = distribution_dict[str(d)] * w
    weights.append(w)
    w *= digit_probs[d]          # == distribution_dict[str(d)][d] * previous w
    tgt_probs.append(digit_probs)

for i, (d, tw) in enumerate(zip(DIGITS, tgt_probs)):
    print(f"position {i+1} (digit {d}): weight={weights[i]:.4f}, "
          f"P[{d}]={distribution_dict[str(d)][d]:.4f}, "
          f"top-3 probs={np.sort(distribution_dict[str(d)])[::-1][:3].round(3)}")

fig, axes = plt.subplots(1, 4, figsize=(13, 3.2),
                         gridspec_kw={"width_ratios": [1, 1, 1, 0.95]})
x = np.arange(10)
colors = ["#1a6fa3", "#0b7a5c", "#9e5c00"]
for ax, d, tw, c in zip(axes[:3], DIGITS, tgt_probs, colors):
    raw = distribution_dict[str(d)]
    bars = ax.bar(x, raw, color="#c9c6bb", edgecolor="white", linewidth=0.6, zorder=2)
    bars[d].set_color(c)
    ax.set_xticks(x)
    ax.set_title(f"position {DIGITS.index(d)+1}: target digit '{d}'\n"
                 f"D({d}, σ={SIGMA})  (× weight {weights[DIGITS.index(d)]:.3f})",
                 fontsize=10)
    ax.set_xlabel("digit")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3, zorder=0)
axes[0].set_ylabel("soft target probability")
axes[2].set_xlim(-0.6, 9.6)

ax = axes[3]
pos = np.arange(1, len(DIGITS) + 1)
ax.bar(pos, weights, color=colors, edgecolor="white", linewidth=0.8, width=0.55, zorder=2)
for p, wt in zip(pos, weights):
    ax.text(p, wt + 0.02, f"{wt:.3f}", ha="center", fontsize=10)
ax.set_xticks(pos)
ax.set_xticklabels([f"'{d}'\n(pos {i+1})" for i, d in enumerate(DIGITS)])
ax.set_title("place-level weights\n$w_i = \\prod_{j<i} D(\\mu_j,\\sigma)[\\mu_j]$",
             fontsize=10)
ax.set_ylabel("weight")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle('PDCE soft targets for "2.69", σ=0.35  (paper Fig.4 pseudo-code)',
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("fig-pdce-soft-targets.png", dpi=150, bbox_inches="tight")
print("saved fig-pdce-soft-targets.png")
