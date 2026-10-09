# AutoVLA xray fig: r_CoT(L) = 1 / (1 + exp(-(L - L_tol) * gamma))
# Paper (Appendix D.3): L_tol = 400, gamma = 2e-3, lambda_r = 0.3
# Bright theme per paper-xray 5.3 (light palette); Chinese annotations are in
# the document text, figure uses short English labels.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#141413"
GRAY = "#6e737b"
ACCENT = "#d97757"   # loss/penalty role
BLUE = "#1a6fa3"     # input role
GREEN = "#0b7a5c"    # operator role

L_tol, gamma, lambda_r = 400, 2e-3, 0.3
L = np.linspace(0, 1200, 1201)
r_cot = 1.0 / (1.0 + np.exp(-(L - L_tol) * gamma))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "mathtext.fontset": "cm",
})

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
fig.patch.set_facecolor("#faf9f5")
ax.set_facecolor("#faf9f5")

ax.plot(L, r_cot, color=ACCENT, lw=2.4, label=r"$r_{\mathrm{CoT}}(L)$")
ax.axvline(L_tol, color=GRAY, lw=1.2, ls="--")
ax.text(L_tol + 12, 0.06, r"$L_{\mathrm{tol}}=400$", color=GRAY, fontsize=11)

# operating regions actually used by the model
ax.axvspan(0, 60, color=BLUE, alpha=0.10)
ax.text(30, 0.90, "fast template\n(short L)", color=BLUE, ha="center", fontsize=10)
ax.axvspan(60, 700, color=GREEN, alpha=0.08)
ax.text(380, 0.90, "CoT region (up to 700 tokens,\nannotation max length)", color=GREEN, ha="center", fontsize=10)

# annotate the actual penalty swing between a fast output and a long CoT
def rcot(l):
    return 1.0 / (1.0 + np.exp(-(l - L_tol) * gamma))

for l0, tag in [(10, "fast"), (700, "long CoT")]:
    y = rcot(l0)
    ax.plot(l0, y, "o", color=ACCENT, ms=6)
    ax.annotate(f"{tag}\nL={l0}: r={y:.2f}",
                xy=(l0, y), xytext=(l0 - 190 if l0 > 400 else l0 + 60, y + 0.12),
                fontsize=10, color=ACCENT, ha="left",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.2))

swing = rcot(700) - rcot(10)
ax.text(760, 0.35,
        f"swing = {swing:.2f}\n$\\times\\lambda_r=0.3$ -> {lambda_r * swing:.3f} reward units\n(PDMS lives in [0, 1])",
        color=ACCENT, fontsize=10, va="center")

ax.set_xlim(0, 1200)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"CoT reasoning length $L$ (tokens)")
ax.set_ylabel(r"$r_{\mathrm{CoT}}$ (sigmoid length penalty)")
ax.set_title("AutoVLA RFT: how hard does the reward really punish long reasoning?",
             fontsize=13, pad=12)
ax.legend(frameon=False, loc="center right")
for spine in ax.spines.values():
    spine.set_linewidth(1.1)
fig.tight_layout()
out = "/home/hatch/workspace/your_files/paper-xray-40/vla-driving/autovla-xray/fig-cot-penalty.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print("saved", out)
print("r(10)=%.3f r(400)=%.3f r(700)=%.3f swing=%.3f" % (rcot(10), rcot(400), rcot(700), swing))
