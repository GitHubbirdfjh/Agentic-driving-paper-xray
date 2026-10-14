"""Self-drawn figure: absorbing-uniform discrete diffusion transition matrix Q_k
for a tiny example (M=3 codebook entries + 1 mask token), with concrete
numbers alpha=0.3 (mask prob), beta=0.2 (uniform-noise prob).
Rows = from-state x_{k-1}, columns = to-state x_k. Follows A.5 of Copilot4D."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#faf9f5",
    "font.size": 11,
    "mathtext.fontset": "cm",
})

M = 3
alpha, beta = 0.3, 0.2
omega = 1 - alpha - beta          # stay
nu = beta / M                    # jump to a *specific* other token

Q = np.full((M + 1, M + 1), nu)
np.fill_diagonal(Q, omega + nu)
Q[:, M] = alpha                  # last column: absorbed into mask
Q[M, :] = 0.0
Q[M, M] = 1.0                    # mask is absorbing
assert np.allclose(Q.sum(axis=1), 1.0)

fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(Q, cmap="Blues", vmin=0, vmax=1)
labels = ["tok A", "tok B", "tok C", "MASK"]
ax.set_xticks(range(M + 1)); ax.set_xticklabels(labels)
ax.set_yticks(range(M + 1)); ax.set_yticklabels(labels)
ax.set_xlabel("to-state $x_k$", fontsize=12)
ax.set_ylabel("from-state $x_{k-1}$", fontsize=12)
ax.set_title("Absorbing-uniform transition matrix $Q_k$  (M=3, "
             r"$\alpha=0.3$, $\beta=0.2$)", fontsize=12, pad=12)

for i in range(M + 1):
    for j in range(M + 1):
        v = Q[i, j]
        if v == 0:
            txt = "0"
        else:
            txt = f"{v:.3f}".rstrip("0")
        ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                color="white" if v > 0.45 else "black",
                fontweight="bold" if i == j or j == M else "normal")

# side annotations
fig.text(0.02, 0.02,
         r"$\omega=1-\alpha-\beta=0.5$: token stays put" "\n"
         r"$\nu=\beta/M=0.067$: token scrambled to a random code" "\n"
         r"$\alpha=0.3$: token absorbed into MASK (last column)" "\n"
         "MASK row: once masked, always masked (absorbing)",
         fontsize=10, color="#3d3d3a", va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ed", ec="#deddd8"))
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("transition probability", fontsize=10)
fig.tight_layout(rect=[0, 0.16, 1, 1])
fig.savefig("fig-transition.png", dpi=150, bbox_inches="tight")
print("saved fig-transition.png")
print("row sums:", Q.sum(axis=1))
