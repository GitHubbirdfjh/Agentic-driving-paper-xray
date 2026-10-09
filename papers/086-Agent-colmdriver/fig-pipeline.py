"""Fig: CoLMDriver dual-pipeline architecture.
High-frequency low-level pipeline (perception-planning-control) +
low-frequency high-level guidance (VLM intention planner, LLM negotiation w/ critic).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG = "#faf9f5"
IN, IN_F = "#1a6fa3", "#e2eff5"      # perception input
OUT, OUT_F = "#9e5c00", "#f8ecd2"    # waypoints / control output
OP, OP_F = "#0b7a5c", "#dcf2e8"      # learned modules
ALT, ALT_F = "#6a55c2", "#ebe7f8"    # LLM / new parts
WIRE = "#6e737b"

fig, ax = plt.subplots(figsize=(12.5, 6.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.2); ax.axis("off")

def box(x, y, w, h, label, fc, ec, tc, fs=10):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=ec, linewidth=1.6)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs,
            color=tc, linespacing=1.25)

def arrow(x0, y0, x1, y1, c=WIRE):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4,
                                mutation_scale=12))

# ---- low-level pipeline (bottom, high frequency) ----
ax.text(0.4, 2.55, "low-level pipeline @ high frequency", fontsize=11, color=WIRE, style="italic")
box(0.4, 1.15, 2.0, 1.0, "perception\n(PointPillars)", IN_F, IN, IN)
box(3.0, 1.15, 2.6, 1.0, "intention-guided\nwaypoint planner", OP_F, OP, OP)
box(6.2, 1.15, 2.0, 1.0, "control\n(2x PID)", OUT_F, OUT, OUT)
arrow(2.4, 1.65, 3.0, 1.65); arrow(5.6, 1.65, 6.2, 1.65)
ax.text(8.6, 1.65, "steer / throttle / brake", fontsize=10, color=OUT)
arrow(8.2, 1.65, 8.55, 1.65, OUT)

# ---- high-level pipeline (top, low frequency) ----
ax.text(0.4, 5.55, "high-level guidance @ low frequency", fontsize=11, color=WIRE, style="italic")
box(0.4, 4.15, 2.4, 1.0, "VLM intention planner\n(InternVL2-4B, LoRA)", ALT_F, ALT, ALT, fs=9.5)
box(3.4, 4.15, 2.6, 1.0, "dynamic grouping\n(safety score < 0.75)", OP_F, OP, OP, fs=9.5)
box(6.6, 4.15, 2.4, 1.0, "LLM negotiator\n(Qwen2.5-3B)", ALT_F, ALT, ALT)
box(9.6, 4.15, 2.4, 1.0, "evaluator (critic)\nconsensus/safety/eff.", ALT_F, ALT, ALT, fs=9.5)
arrow(2.8, 4.65, 3.4, 4.65); arrow(6.0, 4.65, 6.6, 4.65); arrow(9.0, 4.65, 9.6, 4.65)
# critic feedback loop
ax.annotate("", xy=(6.9, 4.15), xytext=(11.7, 4.15),
            arrowprops=dict(arrowstyle="-|>", color=ALT, lw=1.4, mutation_scale=12,
                            connectionstyle="arc3,rad=0.35"))
ax.text(9.4, 3.35, "critic feedback\n(next round)", fontsize=9, color=ALT, ha="center")

# intention flows down to waypoint planner
arrow(1.6, 4.15, 1.6, 2.6)
ax.text(1.75, 3.4, "driving intention\n(no conflict)", fontsize=9, color=ALT)
arrow(7.8, 4.15, 4.6, 2.6)
ax.text(6.9, 3.35, "negotiated intention\n(conflict)", fontsize=9, color=ALT)

# waypoint feedback up to evaluator
ax.annotate("", xy=(10.8, 4.15), xytext=(4.3, 2.15),
            arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.2, mutation_scale=12,
                            linestyle="dashed", connectionstyle="arc3,rad=-0.25"))
ax.text(8.3, 2.75, "planned waypoints\n(score safety/eff.)", fontsize=9, color=WIRE, ha="center")

ax.set_title("CoLMDriver: two parallel pipelines, async connection via driving intentions",
             fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/086-Agent-colmdriver/fig-pipeline.png",
            dpi=150, facecolor=BG)
print("saved")
