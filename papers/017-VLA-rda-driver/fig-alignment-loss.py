# -*- coding: utf-8 -*-
"""RDA-Driver Fig B: how the ranking alignment loss (Eq 3) behaves.

Left: L_rank = log(1 + sum exp(gap)) as a function of the score gap
      gap = D(s_neg) - s_pos, for 1 and 3 violated pairs.
Right: a concrete 3-sample worked example. Three CoTs are sampled from the
      vanilla-finetuned model; they are ranked by trajectory L2. Their
      token-average log-probs s are deliberately misaligned (the best
      trajectory has the worst score). The loss pulls up the good ones;
      detach means the bad ones are never pushed down.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

# ---- left: loss curve ----
ax = axes[0]
gap = np.linspace(-3, 3, 400)
for n_pairs, style, lab in [(1, "-", "1 violated pair"), (3, "--", "3 violated pairs")]:
    L = np.log(1 + n_pairs * np.exp(gap))
    ax.plot(gap, L, style, linewidth=2, label=lab)
ax.axvline(0, color="#6e737b", linewidth=0.8)
ax.set_xlabel("score gap = D(s_negative) - s_positive")
ax.set_ylabel("L_rank = log(1 + sum exp(gap))")
ax.set_title("Ranking loss: only the gap matters, not absolute scores", fontsize=11)
ax.legend(fontsize=9)
ax.annotate("gap >> 0 (negative scores\nhigher): loss explodes,\npositive gets pulled up hard",
            xy=(1.6, 2.2), fontsize=9, color="#b34a06")
ax.annotate("gap << 0 (positive already\nabove negatives): loss -> 0,\nnothing to fix",
            xy=(-2.6, 0.35), fontsize=9, color="#0b7a5c")
ax.set_ylim(-0.1, 4.6)

# ---- right: worked example ----
ax = axes[1]
# three sampled CoTs, ranked by trajectory L2 (c1 best, c3 worst)
l2 = np.array([0.50, 1.20, 2.10])
s = np.array([-0.80, -1.50, -0.60])  # token-avg log-probs: misaligned
names = ["c1 (best traj)", "c2 (mid)", "c3 (worst traj)"]

x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w / 2, -l2, w, color="#1a6fa3", label="-L2 (trajectory quality, up=better)")
b2 = ax.bar(x + w / 2, s + 2.0, w, color="#b34a06", label="s + 2 (token-avg log-prob, up=better)")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_title("Worked example: scores are misaligned with trajectory quality", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylabel("relative scale (higher = better)")

# annotate the three violated pairs and their exp(gap)
pairs = [((0, 1), "c1>c2: exp(-0.70)=0.497"),
         ((0, 2), "c1>c3: exp(+0.20)=1.221"),
         ((1, 2), "c2>c3: exp(+0.90)=2.460")]
for (i, j), txt in pairs:
    ax.annotate(txt, xy=((x[i] + x[j]) / 2 + 0.17, s[j] + 2.05),
                xytext=((x[i] + x[j]) / 2 + 0.17, s[j] + 2.35),
                fontsize=8, ha="center", color="#7a6b00",
                arrowprops=dict(arrowstyle="->", color="#7a6b00", lw=1))
ax.text(0.02, 0.96, "L = log(1 + 0.497 + 1.221 + 2.460) = 1.644\n"
        "gradient pulls s(c1), s(c2) UP;\n"
        "D(detach): negatives are anchors, never pushed down",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8ecd2", alpha=0.95))

fig.suptitle("Eq (3), L_rank: why it fixes misalignment without breaking the model", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/017-VLA-rda-driver/fig-alignment-loss.png",
            dpi=150)
print("saved fig-alignment-loss.png")
