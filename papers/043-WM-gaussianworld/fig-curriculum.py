"""Self-drawn: GaussianWorld streaming-training curriculum (Table 4, config
nusc_surroundocc_stream.py). p = probability of dropping previous-frame
Gaussians; seq length grows 5 -> 38 over 30 epochs. Schedule E (probabilistic)
wins over fixed schedules A-D.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

p_sched = [(0.2, 5), (0.1, 5), (0.05, 5), (0.033, 5), (0.0, 10)]
l_sched = [(5, 5), (10, 5), (20, 5), (30, 5), (38, 10)]

ep = np.arange(31)
p = np.zeros(31); sl = np.zeros(31)
start = 0
for (pv, n), (lv, _) in zip(p_sched, l_sched):
    p[start:start + n + 1] = pv
    sl[start:start + n + 1] = lv
    start += n

fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
a1.step(ep, p, where="post", lw=2.2, color="#b34a06")
a1.set_ylabel("drop prob $p$\n(prev Gaussians discarded)", fontsize=10.5)
a1.set_ylim(-0.02, 0.25)
a1.set_title("GaussianWorld streaming-training curriculum (config, Table 4 schedule E)",
             fontsize=12.5, weight="bold", pad=10)
a1.grid(alpha=0.3)

a2.step(ep, sl, where="post", lw=2.2, color="#1a6fa3")
a2.set_ylabel("sequence length\n(frames)", fontsize=10.5)
a2.set_xlabel("epoch", fontsize=11)
a2.set_ylim(0, 42)
a2.set_xticks([0, 5, 10, 15, 20, 25, 30])
a2.grid(alpha=0.3)
for x in [5, 10, 15, 20]:
    a1.axvline(x, ls="--", c="#9aa0a8", lw=1)
    a2.axvline(x, ls="--", c="#9aa0a8", lw=1)

a2.text(15, 33, "Early: model can't evolve yet -> train on short seqs\n"
                "and frequently restart from scratch ($p$ high).\n"
                "Late: long seqs, $p=0$, full streaming.",
         fontsize=9.5, style="italic", color="#333333",
         bbox=dict(fc="#f5f4ed", ec="#deddd8", boxstyle="round,pad=0.5"))

# inset: table 4 comparison
tbl = (("A: fixed long seq + sum acc", "19.77", "30h"),
       ("B: fixed long seq + mean acc", "19.81", "30h"),
       ("C: no curriculum, no acc", "18.63", "20h"),
       ("D: only long seq (30 ep)", "19.82", "15h"),
       ("E: probabilistic curriculum", "20.24", "15h"))
t = a2.text(0.97, 0.03, "\n".join(f"{r[0]:<32s} mIoU {r[1]}  {r[2]}" for r in tbl),
            transform=a2.transAxes, ha="right", va="bottom", fontsize=8.2,
            family="monospace", bbox=dict(fc="#e2eff5", ec="#1a6fa3", boxstyle="round,pad=0.4"))
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/043-WM-gaussianworld/fig-curriculum.png",
            dpi=150, bbox_inches="tight")
print("saved fig-curriculum.png")
