# Temperature sweep in the VizDoom "Take Cover" dream experiment.
# Data from Table 2 of "World Models" (Ha & Schmidhuber, 2018):
# tau, virtual score, actual score (mean +- std over 100 rollouts)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

tau      = [0.10, 0.50, 1.00, 1.15, 1.30]
virtual  = [2086, 2060, 1145, 918, 732]
v_std    = [140, 277, 690, 546, 269]
actual   = [193, 196, 868, 1092, 753]
a_std    = [58, 50, 511, 556, 139]
labels   = ["0.10", "0.50", "1.00", "1.15", "1.30"]

x = np.arange(len(tau))
w = 0.36

fig, ax = plt.subplots(figsize=(9.5, 5.2))
b1 = ax.bar(x - w/2, virtual, w, yerr=v_std, capsize=4, label="score inside the dream",
            color="#e69f00", edgecolor="#9e5c00")
b2 = ax.bar(x + w/2, actual, w, yerr=a_std, capsize=4, label="score transferred to real env",
            color="#56b4e9", edgecolor="#1a6fa3")
ax.axhline(750, color="#b34a06", ls="--", lw=1.4, label="task solved threshold (750)")
ax.axhline(210, color="#6e737b", ls=":", lw=1.2, label="random policy (~210)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("sampling temperature tau of the MDN-RNN dream", fontsize=11)
ax.set_ylabel("survival time steps (mean +- std, 100 rollouts)", fontsize=11)
ax.set_title("Dream temperature: cheating the world model vs. robust transfer\n"
             "VizDoom Take Cover, 'World Models' (Ha & Schmidhuber, 2018), Table 2",
             fontsize=12.5, weight="bold", pad=12)
ax.legend(fontsize=9.5, loc="upper right")
ax.set_ylim(0, 2450)

ax.annotate("low tau: dream too easy -\nagent exploits model flaws",
            xy=(0, 2086), xytext=(0.9, 2100),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9, color="#333")
ax.annotate("sweet spot tau=1.15:\nbest real-env transfer",
            xy=(3, 1092), xytext=(3.6, 1500),
            arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9, color="#333")

fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/108-wm-worldmodels2018/fig-temperature.png", dpi=150, bbox_inches="tight")
print("ok")
