# -*- coding: utf-8 -*-
"""Shared block-diagram helpers for the WM x-ray figures (bright theme)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {
    "in":   ("#1a6fa3", "#e2eff5"),
    "out":  ("#9e5c00", "#f8ecd2"),
    "op":   ("#0b7a5c", "#dcf2e8"),
    "loss": ("#b34a06", "#f9e3d6"),
    "alt":  ("#6a55c2", "#ebe7f8"),
    "norm": ("#7a6b00", "#f8f4cf"),
    "wire": "#6e737b",
}

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def new_fig(w=12, h=7):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax

def box(ax, x, y, w, h, label, kind="op", fontsize=9, sub=None):
    line, fill = C[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.2",
                       facecolor=fill, edgecolor=line, linewidth=1.6)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2 + (1.5 if sub else 0), label, ha="center", va="center",
            fontsize=fontsize, color="#141413", weight="bold", linespacing=1.35)
    if sub:
        ax.text(x + w/2, y + h/2 - 4.2, sub, ha="center", va="center",
                fontsize=7.2, color="#3d3d3a")
    return p

def arrow(ax, x1, y1, x2, y2, label=None, fontsize=7.5, color=None, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                        linewidth=1.4, color=color or C["wire"], linestyle=style,
                        shrinkA=1, shrinkB=2)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 2.2, label, ha="center", va="bottom",
                fontsize=fontsize, color="#3d3d3a",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

def title(ax, s):
    ax.text(50, 96, s, ha="center", va="center", fontsize=13, weight="bold", color="#141413")

def caption(fig, s):
    fig.text(0.5, 0.015, s, ha="center", va="bottom", fontsize=8, color="#73726c",
             wrap=True)
