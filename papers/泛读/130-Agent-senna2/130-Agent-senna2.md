# 一致性不是免费的：Senna-2 用三阶段训练把 VLM 和端到端焊在一起

> **论文**：Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning
> **作者**：Yuehao Song, Shaoyu Chen（项目 lead）, Hao Gao, Yifan Zhu, Weixiang Yue, Jialv Zou, Bo Jiang, Zihao Lu, Yu Wang, Qian Zhang, Xinggang Wang（通讯）——华科 + 地平线，Senna 原班人马
> **Venue**：arXiv:2603.11219v1（2026-03-11）
> **代码**：未找到官方代码（hustvl/Senna 仓库仅提及论文，未收录 Senna-2 实现）——以下判断全部基于论文正文
> **核心洞见（一句话）**：Senna 把 VLM 和端到端拼在一起后，两层经常各说各话（VLM 喊减速，规划器照样加速）；Senna-2 用三阶段训练显式地"焊"它们——开环时只罚不一致的样本，闭环时让规划器先在 3DGS 里用 RL 学会安全高效、再自底向上把 VLM 拉到跟规划器一致。

---

## 先看 Fig.1：consistency gap 长什么样

论文的第一张图是作者的底气所在。它画的是"相对速度比"的分布：规划轨迹第 3 秒的速度除以初速度。VLM 下了三种纵向决策（Accelerate / Keep / Decelerate），理想情况下这三个决策对应的速度比分布应该是三个分开的峰：加速的峰在 1 右边，保持的峰在 1 附近，减速的峰在 1 左边。

Senna 的分布（Fig.1b）是三个混在一起的驼峰——VLM 说 "Decelerate"，规划器输出的速度比可能是 1.2（加速）；VLM 说 "Accelerate"，输出的可能是 0.9。这就是 consistency gap 的可视化：**顶层指导失效了**。Senna-2 的分布（Fig.1c）是三个明显分开的峰。

这个图为什么有说服力？因为它绕开了"决策对不对"这个难问题，只问"规划器听不听话"。听话是一个纯粹的内部一致性问题，不需要真值、不需要人类评判——只要 VLM 说减速，轨迹的速度比就该小于 1。这是一个**可证伪的、便宜的**度量。作者选它当第一张图，说明他们知道审稿人最可能质疑的是"一致性提升是不是刷出来的"，而分布分离是刷不出来的（除非改 fK 阈值，见怀疑式阅读第 4 点）。

## 架构：Decision Adapter 的双 token 设计

Fig.2 的架构分三块：VLM、Decision Adapter、E2E planner。

VLM 换成了 Qwen2.5-VL-3B（Senna 用的是 Vicuna-7B），输入是**单张前视图**（Senna 是 6 张环视）。决策空间也变了：纵向 {accelerate, decelerate, keep, stop} × 横向 {straight, turn left, turn right, change lane left, change lane right}。横向从 3 类扩到 5 类，多了变道——这是对 Senna"12 分类太粗"批评的直接回应。

Decision Adapter 是新增的部件，输出两种 token：

$$F_{vlm} = \text{MLP}(\text{Concat}(T_{vlm}, T_{vel}, T_{dir}))$$

$T_{vlm}$ 是 VLM 最后一层 hidden state 过 MLP——保留 VLM 的推理上下文，是"软"的、连续的；$T_{vel}$、$T_{dir}$ 是可学习的类别 embedding，根据解码出的 meta-action 查表得到——是"硬"的、离散的。两者 concat 再过 MLP 融合成 $F_{vlm}$，然后通过 AdaLN（DiT 那套）注入 diffusion planner，全局调制轨迹生成。

为什么需要两种？Table 4 的消融给了答案（都是只训 Stage 1 的公平对比）：去掉 $T_{vlm}$（只留硬 token），FDE 从 0.567 涨到 0.634（+11.8%），CR 从 0.281 涨到 0.299（+6.4%）；去掉硬 token（只留 $T_{vlm}$），FDE 涨到 0.624（+10.1%），CR 涨到 0.312（+11.0%）。两个都重要，但分工不同：**软 token 带语义（为什么这么决策），硬 token 带类别（决策是什么）**。只给类别，planner 不知道上下文；只给语义，planner 抓不住决策的离散结构。这个"软硬兼施"的设计是整篇论文里最扎实的架构增量。

E2E planner 本体是 DiT（Diffusion Transformer），条件是感知 token（地图、agent、导航），轨迹目标是**残差轨迹**（residual trajectory）：专家轨迹减去"按初速度外推"的参考轨迹。预测残差而不是绝对轨迹，是 DiffusionDrive 那套做法的好处——残差的分布更集中、方差更小，diffusion 学得更容易。

## Stage 1：三路预训练，各练各的

VLM 预训练（公式 2）：

$$L_{VLM} = -\sum_t \log P(d_t \mid d_{<t}, Q)$$

标准的自回归 QA 损失。问题 $Q$ 是前视图 + 导航文本 + 自车速度，答案 $d$ 是用 fK 从 GT 轨迹反推的 meta-action 文本。注意这里 VLM 只看**单张前视图**——Senna 的 6 视角被砍了。论文的理由是"性能和效率的 trade-off"，但代价是真实的：变道决策需要的侧后方信息，VLM 现在看不到了。

E2E 预训练（公式 3）：

$$L_{E2E} = \mathbb{E}_{r_0,\epsilon,t}[\|\epsilon - \epsilon_\theta(r_t, t, c)\|^2]$$

标准 diffusion 去噪损失，$r_0$ 是 GT 残差轨迹，$c$ 是感知条件。此时还没有 VLM 的事。

Adapter 预训练：**冻结 VLM**，只训 adapter + E2E，损失就是 $L_{E2E}$。冻结 VLM 是关键——如果不冻结，$L_{E2E}$ 的梯度会倒灌进 VLM，把它的语言能力洗掉（Senna 三阶段训练里"防止常识遗忘"的教训在这里被制度化了）。Adapter 的任务纯粹是"翻译"：把 VLM 的决策翻译成 planner 能用的条件。

## Stage 2：开环对齐——"自举"的真实含义

这是论文最巧妙、也是最需要祛魅的一节。定义二值一致性指示器（公式 4）：

$$C(\tau, d) = \begin{cases} 1 & f_K(\tau) = d \\ 0 & \text{otherwise} \end{cases}$$

$f_K$ 是运动学映射：把规划器输出的轨迹 $\tau$ 反推成 meta-action（比如末速度比初速度大 20% 就判 accelerate），和 VLM 的决策 $d$ 比对。两者类别一致就是 1，否则 0。

损失（公式 5）：

$$L_{stage2} = (1 - C(\tau, d))(L_{E2E} + \gamma L_{VLM})$$

读法：**只在不一致的样本上加监督**。如果模型自己已经一致（$C=1$），这一项的系数是 0，**不产生任何梯度**。

论文的包装是"inspired by negative sample reinforcement"，说一致样本的预测被当作"隐式专家信号"提供自强化。但把公式摊开看：系数为 0 就是 0，没有梯度就没有"强化"。机制的真实面目是**硬数据过滤**：训练时跳过所有"已经一致"的样本，只在"不一致"的样本上用专家轨迹+GT 决策做监督。

这个设计为什么有效？Table 5：Stage 1→Stage 1+2，Avg F1 从 0.701 涨到 0.764（+9.0%），而开环 FDE 从 0.567 微涨到 0.575。直觉：Stage 1 之后，模型在大部分简单样本上已经一致，继续在这些样本上做回归训练只会让 planner 更拟合专家轨迹的细节，对一致性没帮助；把梯度预算集中到不一致的"硬样本"上，相当于做了一次针对 consistency gap 的 hard example mining。包装词（"自举""隐式专家信号"）可以去掉，**hard mining** 四个字就够了。

但也有代价：被跳过的"一致样本"里，可能包含 planner 轨迹质量差但恰好类别一致的样本（比如 VLM 说 decelerate，轨迹也减速了，但减速曲线很抖）。这些样本在 Stage 2 得不到回归监督，轨迹质量的提升停滞了。这解释了为什么 Stage 2 的 FDE 比 Stage 1 差（0.575 vs 0.567）——**一致性训练和轨迹精度训练在这个公式下是互斥的**，因为公式 5 里根本没有"一致但轨迹差"的处理分支。

$\gamma$ 是平衡 $L_{E2E}$ 和 $L_{VLM}$ 的权重，论文没给具体值（记为缺口）。

## Stage 3：闭环 HRL——自底向上的方向值得玩味

Stage 3 在 3DGS 重建的 1,300 个高风险片段里做在线 RL（1,044 训 / 256 测）。"自底向上"（bottom-up）指优化顺序：**先训低层 planner，再训高层 VLM**。

低层损失（公式 6-8）。安全项：

$$L_{safe} = \mathbb{E}_{\tau\sim\pi_\theta}\sum_{t=1}^{T-1} \mathbf{1}[f_{ttc}(\tau) < \delta_t] \cdot \frac{1}{2}\|\tau_{t+1} - sg(\tau_t)\|^2$$

读法：如果某条 rollout 轨迹的 TTC 小于阈值 $\delta_t$（论文取 3 秒），就把每个轨迹点 $\tau_{t+1}$ 往前一个点 $\tau_t$ 拉（$sg$ 是 stop-gradient，$\tau_t$ 不动）。几何上这是在**收缩轨迹**——让车走得更短，也就是减速。效率项反过来：

$$L_{eff} = \mathbb{E}_{\tau\sim\pi_\theta}\sum_{t=1}^{T-1} \mathbf{1}[f_v(\tau) < \delta_v] \cdot \frac{1}{2}\|\tau_t - sg(\tau_{t+1})\|^2$$

如果速度低于阈值（导航限速和 demo 速度），就把 $\tau_t$ 往 $\tau_{t+1}$ 拉——**拉伸轨迹**，加速。$L_{low} = L_{safe} + L_{eff}$。

这个设计的精妙在于**只用几何操作表达了"该慢则慢、该快则快"**，不需要学一个复杂的 reward 模型。stop-gradient 的使用也很讲究：收缩时锚定前点、拉伸时锚定后点，保证梯度只往一个方向推，不会把整条轨迹平移。但注意：**两个 reward 都是纵向的**（收缩/拉伸只改变速度剖面，不改变路径形状）。横向的安全（别撞护栏、别压线）没有显式 reward，全靠 $L_{high}$ 间接管。

高层损失（公式 9）：

$$L_{high} = -\log P(f_K(\tau) \mid Q)$$

把 RL 优化后的轨迹 $\tau$ 用 fK 反推成决策，让 VLM 去拟合这个决策。**信息流向是自底向上的**：E2E 先在 3DGS 里学会安全高效地开，VLM 再学会"同意" E2E 的开法。论文的叙事是"对齐"，但实质是**低层教高层**。

这个方向选择是有风险的：如果 E2E 在 RL 里学到了一个"正确但反直觉"的行为（比如为了效率贴着前车很近），VLM 会被训练成"觉得这种行为是对的"，它的常识判断被 E2E 的行为模式覆盖了。Senna Stage 1 费那么大劲保留的常识，在 Stage 3 可能被悄悄洗掉。论文没有评估 Stage 3 之后 VLM 在通用 QA 上的表现（记为缺口）。

总损失（公式 10）：$L_{stage3} = L_{high} + \beta L_{low}$，$\beta$ 未公开具体值。

## 数字验算

- **19.3% F1 提升**（Table 1，vs Senna）：Avg F1 0.637→0.760，(0.760-0.637)/0.637=19.3% ✓。分母是重训的 Senna（论文声明所有模型在同一数据集上训练，算公平）。
- **5.7% FDE 下降**（Table 2，开环）：0.633→0.597，(0.633-0.597)/0.633=5.7% ✓。但注意这是 Senna-2（新架构+10,000 小时数据）vs Senna 的**系统级**对比，不是"对齐训练"的增益。Stage ablation（Table 5）显示，对齐训练本身让 FDE 从 0.567 涨到 0.597——**对齐的代价是开环精度**。
- **30.6% AF-CR 下降**（Table 3，闭环）：0.111→0.077，(0.111-0.077)/0.111=30.6% ✓。At-fault collision rate 只计自车责任事故，比总 CR 更能说明问题，选这个指标是诚实的。
- **34.7% AF-CR 下降**（Stage 3 的贡献）：Stage 1+2 的 0.118→0.077，(0.118-0.077)/0.118=34.7% ✓。闭环增益几乎全来自 Stage 3 的 RL。
- **NAVSIM v2 EPDMS 86.6**：vs 前最佳 ResAD 的 85.5，+1.1。幅度不大，且 NAVSIM 是非反应式开环评估，和 3DGS 闭环不是一回事。这个数字放在论文里更像是"我们也不差"的防御性证据，不是主菜。
- **Fig.4 的空路控速实验**：固定 VLM 决策（accelerate/keep/decelerate）做 rollout，速度曲线明显分开——这是"决策跟随能力"的定性证据，和 Fig.1 的分布分离呼应。

## cut-in 场景 walkthrough：三阶段各修了什么

场景（论文 Fig.5b）：右侧车辆 cut-in 进自车道。

**Senna（无对齐）的行为**：VLM 输出错误的 "accelerate"（误判 cut-in 车辆会让行），planner 听话地加速，错过刹车窗口，撞上 merge 车辆；撞完之后 7.5 秒，VLM 又输出一个不合理的 "turn right"。两个错误：决策错一次，规划器无条件执行；撞完之后决策继续错——没有纠错机制。

**Senna-2 Stage 1 之后**：VLM 和 planner 各自在 10,000 小时数据上训过，cut-in 场景的 GT 决策是 decelerate，VLM 大概率能输出 decelerate。但 Stage 1 的 planner 是纯模仿学习的，cut-in 这种长尾场景在数据里少，planner 可能输出"减速但减得不够"的轨迹——fK 反推是 decelerate，和 VLM 一致（$C=1$），但 TTC 可能已经小于 3 秒。**一致，但不安全**。这是 Stage 2 解决不了的（Stage 2 只管一致性）。

**Stage 2 之后**：如果 VLM 输出 accelerate 而 planner 输出减速轨迹（$C=0$），这个样本会被抓出来，用专家轨迹+GT 决策监督。但上面的"一致但不安全"样本在 Stage 2 是**零梯度**——它被跳过了。

**Stage 3 之后**：1,300 个高风险 3DGS 片段里就有 cut-in。Rollout 时 planner 若减速不够，TTC<3s 触发 $L_{safe}$，轨迹被收缩（减速）；然后 $L_{high}$ 让 VLM 拟合 fK(收缩后的轨迹)=decelerate。最终行为：VLM 全程 decelerate，planner 减速避让，然后 VLM 输出 change-lane-left 完成绕行。论文 Fig.5b 展示的就是这个结果。

这个 walkthrough 说明三阶段的分工：Stage 1 给能力，Stage 2 修"不听话"，Stage 3 修"听话但不安全"。**Stage 2 和 Stage 3 修的是两个正交的问题**，这也是 Table 5 里它们增益互补（F1 vs AF-CR）的原因。

## 怀疑式阅读

1. **Stage 2 的"自强化"是零梯度**（上文已展开）。$(1-C)$ 系数为 0 的样本不产生梯度，"隐式专家信号"是个包装。机制=hard example mining，效果真实（F1 +9.0%），解释注水。

2. **Stage 3 的信息流向是低层教高层**。$L_{high}$ 让 VLM 拟合 E2E 轨迹反推的决策。如果 E2E 的 RL 策略有系统性偏差（比如过于激进），VLM 会被同化。论文没有评估 Stage 3 后 VLM 的常识保留度——Senna 花大力气保留的常识，可能在这里被洗掉一部分。

3. **开环精度随阶段单调变差**：FDE 0.567→0.575→0.597。论文用"balance"一词带过，但这是一个真实的 trade-off：一致性和闭环安全是以开环拟合精度为代价的。摘要的"5.7% FDE reduction"是系统级对比（新架构+大数据 vs 旧 Senna），不要误读为"对齐训练提升了精度"——恰恰相反。

4. **fK 阈值未公开**。Consistency F1 的定义依赖"速度比多少算 accelerate"这类阈值。阈值宽，F1 自然高。论文没给阈值，别人复现不了这个指标，Fig.1 的分布分离也有调节阈值的自由度。

5. **数据红利 vs 方法红利没拆开**。Senna-2 用了 ~10,000 小时 / 360M 帧，Senna 用了 100 万 3 秒片段（~833 小时），12 倍的数据量。Table 2 的基线是重训过的，算公平；但 Table 4/5 的 ablation 只拆了架构和 stage，没拆数据量。19.3% 的 F1 提升里，数据贡献了多少？不知道。

6. **两个 reward 都是纵向的**。$L_{safe}$ 收缩、$L_{eff}$ 拉伸，都只改变速度剖面。横向（方向）的安全没有显式 reward，全靠 $L_{high}$ 让 VLM 去"同意"轨迹。如果 RL 学出一个"纵向安全但横向别扭"的策略，VLM 会被训练成觉得它是对的。

7. **3DGS 重建质量未讨论**。1,300 个高风险片段用 3DGS 重建，动态物体（cut-in 的车辆、行人）在新视角下的伪影会直接影响 TTC 计算和 reward 可信度。重建误差是 RL 的噪声源，论文没提。

8. **VLM 降配的信息损失**。Qwen2.5-VL-3B + 单前视 vs Senna 的 Vicuna-7B + 6 视角。变道决策需要的侧后方信息 VLM 看不到了，横向决策扩到 5 类（加了变道）但输入信息反而少了。这个 tension 论文用"trade-off"一笔带过。

9. **Limitation 的 memory bank 一句话带过**。VLM 跑不到 10Hz，所以和 E2E 异步，用 memory bank 缓存 VLM 特征。缓存的特征过期（stale）了怎么办？VLM 3 秒前的决策还适用吗？没有讨论——而这恰恰是 RAMP 要做的异步时间戳补偿要解决的问题。

10. **NAVSIM +1.1 EPDMS 幅度小**。86.6 vs 85.5，且 NAVSIM v2 是伪仿真开环评估。这个数字更像是防御性的，不是主菜。

## 作者的底牌

两张。第一张是**3DGS 闭环 RL 的基础设施**：1,300 个高风险片段的 3DGS 重建 + 在线 rollout，这套 infra 是 RAD（NeurIPS 2025，一作 Hao Gao 也是 Senna-2 作者）留下来的。没有这套 infra，Stage 3 做不了。第二张是**问题的选点**：Senna 发完一年半，所有人都知道 dual-system 有 consistency gap（SimLingo、VLMAD、RDADriver 都在做对齐），但之前的工作都停在"数据级/表示级"对齐，Senna-2 是第一个把"决策-规划一致性"做成**可优化的训练目标**（$C(\tau,d)$ + 三阶段）的。这张"首个显式优化"的牌，在 2026 年初的时间点是硬的。

## 与 RAMP 的血缘/划界

**血缘（RAMP 从 Senna-2 这里继承了什么）**：

- **"校验"思想的训练时版本**。$C(\tau,d) = \mathbf{1}[f_K(\tau) = d]$ 是把"高层决策 vs 低层轨迹"的匹配程度变成可计算信号的最简形式。RAMP 的闭环风险校验是它的运行时版本：把"决策 vs 轨迹"换成"轨迹 vs 世界模型 rollout 的风险"，把训练时优化换成运行时拦截。在 related work 里，Senna-2 是"一致性校验"一支的代表。
- **闭环 RL 的先例**。Stage 3 在 3DGS 里 rollout + TTC reward，是"用闭环反馈优化驾驶策略"的完整示范。RAMP 的 Bench2Drive 闭环验证走的是同一条路（虽然 RAMP 是评估、Senna-2 是训练）。
- **异步的教训**。Limitation 承认 VLM 跑不到 10Hz、靠 memory bank 缓存——这是"慢层异步"的工程现实，RAMP 的异步流水线+时间戳补偿是更系统的回答。

**划界（related work 里要写清楚的三条）**：

1. **校验对象不同：一致性 ≠ 安全性**。Senna-2 校验的是"轨迹听不听决策的话"。但听话的轨迹也可能不安全——论文 132（Drive the Thoughts）明确测出：consistent 的轨迹里也有 unsafe 的（2/74）。RAMP 校验的是"轨迹安不安全"（世界模型 rollout + 风险评分），这是比一致性更底层的安全属性。Senna-2 的 Stage 2 甚至**只修不一致、不修不安全**（一致样本零梯度）。
2. **时机不同：训练时 vs 运行时**。Senna-2 的对齐发生在训练阶段，部署时没有任何运行时检查——VLM 判错了，系统照样执行。RAMP 的校验发生在运行时的每一帧，是部署时的安全网。一个是"让模型少犯错"，一个是"犯错时拦下来"，互补不重叠。
3. **Senna-2 没有记忆、没有多规划器、没有世界模型**。它的"校验器"fK 是手工阈值的运动学离散化，信息量和世界模型 rollout 不在一个量级；它的 VLM 是单决策者，不是调度器。

一句话：Senna-2 回答了"怎么让两层说同一种话"，RAMP 回答的是"两层说同一种话之后，谁来保证这话是对的、谁来记住上次说错话的教训"。
