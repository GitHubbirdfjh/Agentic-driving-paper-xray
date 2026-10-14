# -*- coding: utf-8 -*-
"""NuInstruct SQL 数据生成五步管线（严格网格版）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_DB = "#dbe9f6"; C_STEP = "#e8f5e9"; C_TASK = "#fff3e0"; C_VER = "#fce4ec"

fig, ax = plt.subplots(figsize=(13, 8.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis("off")

def box(x, y, w, h, text, fc, fs=10):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                 fc=fc, ec="#555", lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, linespacing=1.55)

def arrow(x1, y1, x2, y2, lw=1.6, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#455a64", lw=lw, ls=ls))

ax.text(6.5, 8.25, "NuInstruct 数据生成：SQL 管线（对应论文 Fig.2）", fontsize=13, weight="bold", ha="center")

# ---- Row 1: ① 建库 ----
box(0.2, 6.1, 2.3, 1.55, "① 过滤与建库\nnuScenes 标注\n→ ego / instance 信息", C_STEP, 10)
tables = [("Scene\nInformation\nscene→帧列表", 3.0), ("Frame\nInformation\n帧→ego+实例", 4.62),
          ("Ego\nInformation\n位姿/速度/道路", 6.24), ("Instance\nInformation\n3D框/速度/道路", 7.86)]
for t, x in tables:
    box(x, 6.25, 1.55, 1.25, t, C_DB, 8.6)
for i in range(3):
    arrow(4.55 + 1.62*i, 6.875, 4.62 + 1.62*i, 6.875, lw=1.2)
arrow(2.5, 6.875, 3.0, 6.875, lw=1.6)
ax.text(5.93, 5.9, "场景信息数据库：4 张表，一对多映射", fontsize=9.5, ha="center", style="italic")

box(10.3, 6.1, 2.5, 1.55, "② 采样\n随机取连续 3 个关键帧\n（2Hz 标注 → i, i+1, i+2）", C_STEP, 10)

# ---- Row 2: SQL sets ----
box(2.4, 4.15, 6.2, 1.15, "任务 SQL 集：Perception（6）/ Prediction（4）/ Risk（6）/ Planning w/ R（1）\n"
      "每个子任务 SQL = 子任务函数（如 Find_Distance）+ 指令模板", C_TASK, 8.6)
arrow(10.9, 6.1, 10.9, 5.55, lw=1.5)
arrow(8.6, 5.3, 8.6, 4.72, lw=1.2, ls=(0, (4, 3)))
ax.text(9.4, 5.0, "从库里取\n帧与标注", fontsize=8.5, ha="left", color="#333")

# ---- Row 3: ③ ④ ⑤ ----
box(0.2, 2.05, 3.9, 1.6, "③ 检索\n子任务函数在场景库里查答案\n例：Find_Distance(<行人c1>)\n→ Response: (4, -1)", C_STEP, 10)
box(4.9, 2.05, 3.9, 1.6, "④ 保存\ninstruction-response 对\n11,850 关键帧 → 91,355 对\n≈ 每帧 7.7 对", C_STEP, 10)
box(9.6, 2.05, 3.2, 1.6, "⑤ 校验\n人工 或 GPT-4\n剔除错误对", C_VER, 10)
arrow(4.1, 2.85, 4.9, 2.85, lw=1.6)
arrow(8.8, 2.85, 9.6, 2.85, lw=1.6)
arrow(5.2, 4.15, 2.6, 3.65, lw=1.5)

# ---- Row 4: 任务依赖 ----
ax.text(0.2, 1.55, "高阶 SQL 继承低阶 SQL（人类驾驶的决策链条）：", fontsize=10.5, weight="bold", ha="left")
deps = [("Perception\n6 子任务", 0.2), ("Prediction\n4 子任务", 3.6),
        ("Risk\n6 子任务", 7.0), ("Planning w/ R\n1 子任务", 10.4)]
for t, x in deps:
    box(x, 0.25, 2.4, 1.05, t, C_TASK, 9.5)
for i in range(3):
    arrow(2.6 + 3.4*i, 0.775, 3.6 + 3.4*i, 0.775, lw=1.4)
ax.text(6.5, 1.62, "依赖：Perception → Prediction；(Perception, Prediction) → Risk；(Risk, Prediction) → Planning",
        fontsize=8.6, ha="center", color="#333")

plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/089-Agent-nuinstruct/fig-sql-pipeline.png", dpi=150)
print("saved fig-sql-pipeline.png")
