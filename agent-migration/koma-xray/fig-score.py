"""Safety score curve of KoMA's ranking-based reflection (paper Sec. III-E).

Safety_score(t) = 20*(TTC(t)-1.5)/3   if 1.5 < TTC < 3
                = 10                  if TTC > 3
                = 0                   otherwise
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

ttc = np.linspace(0, 5, 501)
score = np.where(ttc <= 1.5, 0.0, np.where(ttc >= 3.0, 10.0, 20.0 * (ttc - 1.5) / 3.0))

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(ttc, score, linewidth=2.5, color="#1a6fa3")
ax.fill_between(ttc, score, alpha=0.12, color="#1a6fa3")

for x in (1.5, 3.0):
    y = 0.0 if x == 1.5 else 10.0
    ax.plot([x], [y], "o", color="#b34a06", markersize=7, zorder=5)
    ax.annotate(f"TTC = {x}s", xy=(x, y), xytext=(8, 14 if x == 1.5 else -22),
                textcoords="offset points", fontsize=10, color="#141413",
                arrowprops=dict(arrowstyle="-", color="#6e737b", lw=1))

ax.axvspan(0, 1.5, alpha=0.08, color="#b34a06")
ax.axvspan(1.5, 3.0, alpha=0.06, color="#9e5c00")

ax.text(0.7, 5.0, "critical risk\nscore = 0", ha="center", fontsize=10, color="#b34a06")
ax.text(2.25, 5.0, "linear ramp\n20*(TTC-1.5)/3", ha="center", fontsize=10, color="#9e5c00")
ax.text(4.1, 5.0, "safe\nscore = 10", ha="center", fontsize=10, color="#0b7a5c")

ax.set_xlim(0, 5)
ax.set_ylim(-0.5, 11)
ax.set_xlabel("Time To Collision (TTC) after action execution [s]")
ax.set_ylabel("Safety score (0-10)")
ax.set_title("KoMA reflection: safety score as a function of TTC", fontsize=12, pad=10)
ax.grid(alpha=0.25, linestyle="--")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False)

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-40/agent-migration/koma-xray/fig-score.png", dpi=150)
print("saved")
