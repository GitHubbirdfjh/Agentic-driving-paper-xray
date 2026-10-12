# -*- coding: utf-8 -*-
"""Drive the Thoughts: three monitors progression with F1 scores."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

labels = ["Rule-based\n(N/A)", "LLM raw\nQwen3.5-4B", "LLM raw\nKimi K2.5", "LLM raw\nGPT-5.5",
          "F-LLM\nQwen3.5-4B", "F-LLM\nKimi K2.5", "F-LLM\nGPT-5.5"]
f1 = [0.38, 0.56, 0.62, 0.59, 0.54, 0.75, 0.75]
prec = [0.41, 0.41, 0.49, 0.47, 0.37, 0.70, 0.74]
rec = [0.35, 0.88, 0.85, 0.81, 1.00, 0.81, 0.77]
colors = ["#b34a06", "#1a6fa3", "#1a6fa3", "#1a6fa3", "#6a55c2", "#6a55c2", "#6a55c2"]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(labels))
b = ax.bar(x, f1, color=colors, edgecolor="black", linewidth=0.8, width=0.62)
for i, v in enumerate(f1):
    ax.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=10, weight="bold")
    ax.text(i, 0.03, f"P {prec[i]:.2f}\nR {rec[i]:.2f}", ha="center", fontsize=7.5, color="white")
ax.set_xticks(x, labels, fontsize=9)
ax.set_ylabel("F1 (INCONSISTENT = positive)", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("CoT-trajectory inconsistency detection on 100 reliable pairs (26 inconsistent / 74 consistent)",
             fontsize=12, weight="bold")
ax.annotate("+0.16 from lane-relative\nevidence (GPT-5.5)", xy=(3, 0.59), xytext=(1.2, 0.85),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ebe7f8"))
ax.annotate("judge scale barely matters:\nKimi F-LLM = GPT-5.5 F-LLM = 0.75",
            xy=(5.5, 0.75), xytext=(5.5, 0.32), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8ecd2"))
ax.text(0.5, -0.30, "Takeaway: representation (lane-relative evidence) beats judge scale. "
        "Rule-based closed-vocab matching collapses on paraphrases (F1 0.38).\n"
        "Qwen3.5-4B reaches recall 1.00 but precision 0.37 — a fire alarm that rings at burnt toast.",
        transform=ax.transAxes, fontsize=8.5, color="#333333", ha="center")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/agentic-driving-idea-report/xray/132-Agent-drivethoughts/fig-monitors.png", dpi=150)
print("saved1")
