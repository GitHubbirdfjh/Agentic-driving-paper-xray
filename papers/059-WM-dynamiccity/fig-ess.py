# -*- coding: utf-8 -*-
"""Self-drawn figure: Expansion & Squeeze Strategy (ESS) tensor geometry.

Shows how the six HexPlane feature maps are broadcast-expanded to the full
4D volume T x X x Y x Z x C and fused with a Hadamard product,
instead of querying each 3D point individually (code: decoder.py, DecoderBase).
Concrete numbers: CarlaSC config, after encoder: T=8, X=64, Y=64, Z=4, C=64 (query channels).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_IN, C_Q = 16, 64
T, X, Y, Z = 8, 64, 64, 4  # after encoder downsampling (CarlaSC, latent)

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.6)
ax.axis("off")

COLS = {"plane": "#dcebf5", "edge": "#1a6fa3", "prod": "#f8ecd2",
        "prode": "#9e5c00", "out": "#dcf2e8", "oute": "#0b7a5c",
        "ann": "#6e737b"}

ax.text(6.5, 7.25, "Expansion & Squeeze: 六张平面如何拼回一个 4D 体",
        ha="center", fontsize=15, weight="bold")
ax.text(6.5, 6.92, "每张平面只在自己没有的维度上做广播, 六路逐元素相乘得到 T×X×Y×Z×C",
        ha="center", fontsize=10.5, color=COLS["ann"])

planes = [
    ("Pxy", "X×Y", X, Y, "广播 T, Z", 0.6),
    ("Pxz", "X×Z", X, Z, "广播 T, Y", 2.7),
    ("Pyz", "Y×Z", Y, Z, "广播 T, X", 4.8),
    ("Ptx", "T×X", T, X, "广播 Y, Z", 6.9),
    ("Pty", "T×Y", T, Y, "广播 X, Z", 9.0),
    ("Ptz", "T×Z", T, Z, "广播 X, Y", 11.1),
]
for name, dims, d1, d2, note, xc in planes:
    r = Rectangle((xc - 0.75, 4.55), 1.5, 1.25, facecolor=COLS["plane"],
                  edgecolor=COLS["edge"], lw=1.6, zorder=2)
    ax.add_patch(r)
    ax.text(xc, 5.45, name, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(xc, 5.02, f"{dims}×{C_Q}", ha="center", va="center", fontsize=9.5)
    ax.text(xc, 4.30, note, ha="center", va="center", fontsize=8.5, color=COLS["ann"])
    ax.annotate("", xy=(xc, 3.95), xytext=(xc, 4.15),
                arrowprops=dict(arrowstyle="-|>", color=COLS["edge"], lw=1.4))

# product node
prod = Rectangle((4.4, 2.55), 4.2, 1.1, facecolor=COLS["prod"],
                 edgecolor=COLS["prode"], lw=2.0, zorder=2)
ax.add_patch(prod)
ax.text(6.5, 3.28, "逐元素相乘  ⊙", ha="center", va="center", fontsize=13, weight="bold")
ax.text(6.5, 2.85, "Hadamard product, 六路广播对齐后相乘", ha="center", va="center",
        fontsize=9, color=COLS["ann"])

# output volume
out = Rectangle((4.9, 0.55), 3.2, 1.35, facecolor=COLS["out"],
                edgecolor=COLS["oute"], lw=2.0, zorder=2)
ax.add_patch(out)
ax.text(6.5, 1.48, "X′txyz", ha="center", va="center", fontsize=13, weight="bold")
ax.text(6.5, 1.02, f"T×X×Y×Z×C = {T}×{X}×{Y}×{Z}×{C_Q}", ha="center",
        va="center", fontsize=9.5)
ax.text(6.5, 0.68, "→ 拼接位置编码 → 卷积上采样 → 类别", ha="center", va="center",
        fontsize=8.5, color=COLS["ann"])
ax.annotate("", xy=(6.5, 1.9), xytext=(6.5, 2.5),
            arrowprops=dict(arrowstyle="-|>", color=COLS["prode"], lw=1.6))

# side annotation: why not per-point query
ax.text(0.25, 2.2, "对比: 逐点查询要对 T·X·Y·Z = 131,072 个点\n"
        "各做一次六平面采样 + MLP, 显存与时间\n"
        "都随分辨率线性涨; ESS 只做 6 次广播\n"
        "乘法 + 一次并行卷积解码。",
        fontsize=9, color=COLS["ann"], va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4ed", ec="#deddd8"))

ax.text(12.75, 0.3, "CarlaSC 配置下形状 (T=8, X=Y=64, Z=4, C=64)",
        ha="right", fontsize=8.5, color=COLS["ann"])

plt.tight_layout()
out_path = "/home/hatch/workspace/your_files/paper-xray-100/papers/059-WM-dynamiccity/fig-ess.png"
plt.savefig(out_path, dpi=150)
print("saved", out_path)
