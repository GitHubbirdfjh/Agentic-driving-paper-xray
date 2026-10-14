"""Figure: exponential tilting of the return distribution in CtRL-Sim.

Mechanism from the paper (App. C.2.1) and code (simulator.py get_tilt_logits /
process_predicted_rtg): returns are discretized into 350 bins over [-10, 0]
(code: min_rtg_veh=-10, max_rtg_veh=0, rtg_discretization=350). At inference the
agent samples returns from

    p_kappa(b) \propto p(b) * exp(kappa * b / 349),   b = 0..349

i.e. the learned categorical logits get a linear ramp kappa*linspace(0,1,350)
added before the softmax. The learned distribution p(b) below is SCHEMATIC
(the paper does not publish it); the tilting arithmetic is exact.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# role colors from the skill (light theme)
IN, OUT, LOSS = "#1a6fa3", "#9e5c00", "#b34a06"
INK, INK2 = "#141413", "#3d3d3a"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#6e737b",
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "mathtext.fontset": "cm",
})

N = 350
b = np.arange(N)
ret = -10.0 + 10.0 * b / (N - 1)          # actual return value per bin

# schematic learned return distribution: most mass at high returns (no collision),
# small bump at the collision end
rng = np.random.default_rng(0)
p = 0.92 * np.exp(-0.5 * ((b - 320) / 45.0) ** 2)
p += 0.08 * np.exp(-0.5 * ((b - 15) / 18.0) ** 2)
p = p / p.sum()

def tilt(p, kappa):
    q = p * np.exp(kappa * b / (N - 1))
    return q / q.sum()

p_pos = tilt(p, +10.0)
p_neg = tilt(p, -10.0)

fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.plot(ret, p, color=IN, lw=2.2, label=r"learned $p(G_t\mid s_t)$ (schematic)")
ax.plot(ret, p_pos, color=OUT, lw=2.2, label=r"tilted $\kappa=+10$ (good)")
ax.plot(ret, p_neg, color=LOSS, lw=2.2, label=r"tilted $\kappa=-10$ (adversarial)")
ax.axvline(-10, color="#6e737b", ls="--", lw=1)
ax.set_xlabel("discretized return-to-go bin (mapped to [-10, 0])")
ax.set_ylabel("probability")
ax.set_title("Exponential tilting reshapes the return distribution",
             color=INK, fontsize=13, pad=10)
ax.legend(frameon=False, fontsize=10.5, loc="center right")
ax.text(-6.2, 0.028, "low-return end =\ncollisions with ego",
        color="#6e737b", fontsize=10, va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#deddd8"))
ax.set_xlim(-10.2, 0.2)
fig.tight_layout()
fig.savefig("fig-tilt.png", dpi=150)
print("saved fig-tilt.png")
