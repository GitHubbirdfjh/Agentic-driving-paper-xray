# -*- coding: utf-8 -*-
"""BEV-InMLLM 架构：MV-MLLM（式1-2）+ BEV 注入模块（式3-4），标注张量形状（干净两列版）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

C_VID = "#e3f2fd"; C_Q = "#ede7f6"; C_BEV = "#fff8e1"; C_LLM = "#f3e5f5"; C_FRZ = "#eceff1"

fig, ax = plt.subplots(figsize=(13.5, 8.0))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.2); ax.axis("off")

def box(x, y, w, h, text, fc, fs=9.5):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                 fc=fc, ec="#555", lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, linespacing=1.55)

def arrow(x1, y1, x2, y2, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#455a64", lw=lw))

def label(x, y, s, fs=9, bg=None):
    t = ax.text(x, y, s, fontsize=fs, color="#b34a06", ha="center", style="italic", weight="bold")
    if bg:
        t.set_bbox(dict(fc="white", ec="none", pad=1.5))
    return t

ax.text(6.75, 7.85, "BEV-InMLLM：多视图视频 → 指令感知的 BEV 注入（论文 Fig.5 + 式1–4）", fontsize=13, weight="bold", ha="center")

# ===== 左列：MV-MLLM =====
ax.text(3.25, 7.35, "MV-MLLM（式1–2）", fontsize=11, weight="bold", ha="center")
box(0.2, 5.9, 2.95, 1.15, "多视图视频\n{V_i}, i=1..6\n每视图 3 帧，224x224", C_VID)
box(3.45, 5.9, 2.95, 1.15, "Vision Encoder\n+ Connection Module\n(冻结，只训 ST-Adapter)", C_FRZ)
arrow(3.15, 6.475, 3.45, 6.475); label(3.3, 6.75, "式(1)")
box(0.2, 4.45, 2.95, 1.15, "视觉 tokens\nF^i_vis ∈ R^(N_vis×D_vis)", C_VID)
box(3.45, 4.45, 2.95, 1.15, "Multi-view Q-Former\nQ_mv ∈ R^(32×1408)\nCrossAttn(Q_mv, concat)\n(可训练)", C_Q, 9)
arrow(3.15, 5.025, 3.45, 5.025); label(3.3, 5.31, "式(2)")
arrow(1.675, 5.9, 1.675, 5.6)
arrow(4.925, 5.9, 4.925, 5.6)
box(3.45, 3.0, 2.95, 1.15, "F_mv ∈ R^(32×1408)\n多视图语义 tokens", C_Q)
arrow(4.925, 4.45, 4.925, 4.15)
box(3.45, 1.55, 2.95, 1.15, "LLM（冻结）\nF_mv + 指令 tokens → 回复", C_LLM)
arrow(4.925, 3.0, 4.925, 2.7)

# ===== 右列：BEV 注入 =====
ax.text(10.15, 7.35, "BEV 注入模块（式3–4，即插即用）", fontsize=11, weight="bold", ha="center")
box(7.2, 5.9, 2.9, 1.15, "BEV 特征提取器\nLSS / BEVDet / BEVFusion\n(冻结)", C_FRZ)
box(10.4, 5.9, 2.9, 1.15, "指令 tokens\nL_inst ∈ R^(N_inst×D_inst)", C_LLM)
box(7.2, 4.45, 6.1, 1.15, "Instruction-aware BEV Q-Former（可训练）\n查询 = Q_bev(可学习, R^(32×1408)) ⊕ L_inst；  键值 = F_bev ∈ R^(200×200×D_bev)", C_BEV, 9)
arrow(8.65, 5.9, 8.65, 5.6)
arrow(11.85, 5.9, 11.0, 5.6)
label(6.85, 5.025, "式(3)", bg=True)
box(7.2, 3.0, 6.1, 1.15, "F^inst_bev ∈ R^((32+N_inst)×1408)\n指令相关的 BEV tokens", C_BEV)
arrow(10.25, 4.45, 10.25, 4.15)

# ===== 注入箭头 =====
arrow(7.2, 3.575, 6.4, 3.575, lw=2.0)
label(6.8, 3.575, "式(4)", bg=True)

# ===== 训练说明 =====
ax.text(0.2, 0.35, "训练只更新：ST-Adapter、MV Q-Former、BEV Q-Former、注入模块；\n"
        "视觉编码器、BEV 提取器、LLM 全冻结。\n"
        "默认超参：N_view=6, K_mv=K_bev=32, D_vis=1408, W=H=200；AdamW, lr=1e-4, cosine, 20 epochs",
        fontsize=9, ha="left", color="#333", linespacing=1.7,
        bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.35"))

plt.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/089-Agent-nuinstruct/fig-bevin.png", dpi=150)
print("saved fig-bevin.png")
