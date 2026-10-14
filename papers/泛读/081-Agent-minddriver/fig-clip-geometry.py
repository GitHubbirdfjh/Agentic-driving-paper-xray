"""MindDriver Stage-1 奖励 r_img 的几何意义：CLIP 嵌入空间中的余弦相似度。"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), subplot_kw={"aspect": "equal"})
thetas = [15, 45, 80]
labels = ["语义一致（奖励≈0.97）", "部分偏离（奖励≈0.71）", "语义错位（奖励≈0.17）"]

for ax, th, lab in zip(axes, thetas, labels):
    th = np.deg2rad(th)
    v_gt = np.array([1.0, 0.0])
    v_dr = np.array([np.cos(th), np.sin(th)])
    ax.annotate("", xy=v_gt, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9e5c00", lw=3))
    ax.annotate("", xy=v_dr, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#6a55c2", lw=3))
    arc = np.linspace(0, th, 60)
    ax.plot(0.35*np.cos(arc), 0.35*np.sin(arc), color="#3d3d3a", lw=1.5)
    ax.text(0.42, 0.10, r"$\theta$", fontsize=14, color="#3d3d3a")
    ax.text(1.02, 0.02, r"$E_{CLIP}(I_{GT})$", fontsize=11, color="#9e5c00", weight="bold")
    ax.text(v_dr[0]*1.02, v_dr[1]*1.02 + 0.05, r"$E_{CLIP}(I_{dream})$",
            fontsize=11, color="#6a55c2", weight="bold")
    ax.set_xlim(-0.25, 1.35); ax.set_ylim(-0.25, 1.35)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lab, fontsize=11, weight="bold", pad=10)

fig.suptitle(r"Stage-1 奖励 $r_{img}=\cos\theta$：只看语义方向对不对，不看像素像不像",
             fontsize=12.5, weight="bold", y=1.03)
fig.text(0.5, -0.01,
         "两个向量都先归一化（除以各自的模），点积就只剩夹角信息："
         "红绿灯画在哪不重要，重要的是梦里有没有红绿灯、位置对不对。",
         ha="center", fontsize=9.5, color="#3d3d3a")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/your_files/paper-xray-100/papers/081-Agent-minddriver/fig-clip-geometry.png",
            dpi=150, bbox_inches="tight")
print("saved fig-clip-geometry.png")
