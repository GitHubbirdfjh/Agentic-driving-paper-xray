"""Fig: safety score sigmoid curve from the released code.
score = 1 / (1 + exp(-2 * (d_min - 3)))
Mark the 3m inflection point and two worked examples.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, WIRE = "#faf9f5", "#6e737b"
LOSS, LOSS_F = "#b34a06", "#f9e3d6"
OUT = "#9e5c00"

d = np.linspace(0, 8, 400)
s = 1 / (1 + np.exp(-2 * (d - 3)))

fig, ax = plt.subplots(figsize=(8.5, 4.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(d, s, color=LOSS, linewidth=2.4)
ax.fill_between(d, s, alpha=0.25, color=LOSS_F if False else LOSS)
ax.axvline(3, color=WIRE, linestyle="--", linewidth=1.2)
ax.axhline(0.5, color=WIRE, linestyle=":", linewidth=1.0)
ax.text(3.08, 0.55, "inflection at $d_{min}=3$m\nscore = 0.5", fontsize=10, color=WIRE)

# worked examples
for dm, lab in [(2.0, "$d_{min}=2$m\nscore $=1/(1+e^{2})\\approx 0.12$"),
                (5.0, "$d_{min}=5$m\nscore $\\approx 0.98$")]:
    sc = 1 / (1 + np.exp(-2 * (dm - 3)))
    ax.plot(dm, sc, "o", color=OUT, markersize=9, zorder=5)
    ax.annotate(lab, xy=(dm, sc), xytext=(dm + (1.2 if dm < 3 else -2.6), sc + 0.18),
                fontsize=10, color=OUT,
                arrowprops=dict(arrowstyle="->", color=OUT, lw=1.2))

ax.set_xlim(0, 8); ax.set_ylim(-0.05, 1.15)
ax.set_xlabel("min distance between two vehicles' planned waypoints $d_{min}$ (m)", fontsize=11)
ax.set_ylabel("safety score", fontsize=11)
ax.set_title("Critic's safety score (from released code): $\\sigma(2\\,(d_{min}-3))$",
             fontsize=12.5)
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/086-Agent-colmdriver/fig-safety-curve.png",
            dpi=150, facecolor=BG)
print("saved")
