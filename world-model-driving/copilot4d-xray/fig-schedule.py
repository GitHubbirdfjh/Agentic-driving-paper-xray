"""Self-drawn figure: mask schedule gamma(u)=cos(u*pi/2) and per-step parallel
decoding counts for K=10 diffusion steps on N=128x128=16384 tokens (Copilot4D)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#faf9f5",
    "axes.facecolor": "#faf9f5",
    "font.size": 11,
    "mathtext.fontset": "cm",
})

N = 128 * 128
K = 10
gamma = lambda u: np.cos(u * np.pi / 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- left: mask schedule curve ----
u = np.linspace(0, 1, 400)
ax1.plot(u, gamma(u), color="#1a6fa3", lw=2.5)
ax1.fill_between(u, gamma(u), alpha=0.12, color="#1a6fa3")
ax1.set_xlabel("u  (mask ratio knob)")
ax1.set_ylabel(r"$\gamma(u)=\cos(u\pi/2)$  (fraction masked)")
ax1.set_title("Training: mask $\\lceil\\gamma(u_0)N\\rceil$ tokens, $u_0\\sim U(0,1)$",
              fontsize=12)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
ax1.grid(alpha=0.25)
# annotate aggressive masking: at u=0.5 half the tokens are masked
ax1.annotate("u=0.5 -> 70.7% masked\n(aggressive schedule)",
             xy=(0.5, gamma(0.5)), xytext=(0.62, 0.35),
             arrowprops=dict(arrowstyle="->", color="#3d3d3a"),
             fontsize=10, color="#3d3d3a")
ax1.tick_params(labelsize=10)

# ---- right: tokens newly decoded per sampling step ----
# Algorithm 2: at step k (K-1 -> 0), keep top M_k = ceil(gamma(k/K) N) tokens
M = {k: int(np.ceil(gamma(k / K) * N)) for k in range(K + 1)}
steps = list(range(K - 1, -1, -1))          # sampling order: 9, 8, ..., 0
new_tokens = [M[k] - M[k + 1] for k in steps]
bars = ax2.bar([str(k) for k in steps], new_tokens, color="#9e5c00",
               edgecolor="#141413", lw=0.6)
ax2.set_xlabel("diffusion step k  (sampling order: 9 -> 0)")
ax2.set_ylabel("newly decoded tokens at this step")
ax2.set_title("Sampling: ~1600 new tokens/step in parallel (avg)",
              fontsize=12)
ax2.tick_params(labelsize=10)
ax2.grid(axis="y", alpha=0.25)
avg = N / K
ax2.axhline(avg, color="#b34a06", ls="--", lw=1.5,
            label=f"average = {avg:.0f}")
for b, v in zip(bars, new_tokens):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 60, str(v),
             ha="center", va="bottom", fontsize=8.5, color="#3d3d3a")
ax2.legend(fontsize=10, framealpha=0.9)
ax2.set_ylim(0, max(new_tokens) * 1.18)

fig.suptitle("Copilot4D: one frame = 16384 tokens, decoded in K=10 parallel steps",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("fig-schedule.png", dpi=150, bbox_inches="tight")
print("saved fig-schedule.png, total tokens:", sum(new_tokens))
