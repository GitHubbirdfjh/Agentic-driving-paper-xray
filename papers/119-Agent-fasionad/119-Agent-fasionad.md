# 快慢双系统最干净的范例：FASIONAD 的不确定性开关与语义反馈

> **信息块**
> - 论文：FASIONAD: FAst and Slow FusION Thinking Systems for Human-Like Autonomous Driving with Adaptive Feedback
> - 作者：Kangan Qian, Zhikun Ma, Yangfan He, Ziang Luo, Tianyu Shi, Tianze Zhu, Jiayin Li, Jianhui Wang, Ziyu Chen, Xiao He, Yining Shi, Zheng Fu, Xinyu Jiao, Kun Jiang, Diange Yang, Takafumi Matsumaru（清华、早稻田、明尼苏达、多伦多、厦门马来西亚分校、电子科大、AI2Robotics 等）
> - Venue：arXiv:2411.18013，v1 2024-11-27；**未发表**（2026-10-11 检索：无期刊/会议录用记录）
> - 链接：https://arxiv.org/abs/2411.18013
> - 代码：**未公开**（正文无代码链接；2026-10-11 全网检索无官方仓库）
> - 续作：FASIONAD++（arXiv:2503.08162，2025-03，8 页）：加入 high-level instruction 与 information bottleneck，在 nuScenes / Town05 Short / Bench2Drive 上验证，开环 L2 降 6.7%、碰撞率降 28.1%
> - 一句话核心洞见：把 Kahneman 的"快思考/慢思考"搬进驾驶栈，但**慢系统永远不直接碰方向盘**——它只在快系统的 reward 分布出现高不确定性时被唤醒，用 VLM 做场景问答，输出压缩成二进制 planning state 向量和元动作 embedding，再回灌给快系统做"语义反馈"。切换信号是学出来的（Laplace 不确定性），不是手写的 if-else。
> - 研读所用版本：v1（18 页，含附录），PDF 已存本目录 `2411.18013.pdf`

## 先回答一个问题：为什么"快慢"这件事值得单独写一篇论文

2024 年下半年，LLM-for-driving 已经挤满了人：GPT-Driver 让 LLM 直接输出轨迹，Agent-Driver 让 LLM 当 agent 调工具，DriveVLM 让 VLM 做规划。所有这些工作的阿喀琉斯之踵是同一个：**VLM 太慢**。DriveLM 的图问答需要多轮 LLM 推理，单步几秒——开车等不起。

摆在面前的设计空间有三个选项。选项一：只用快系统（端到端），快但长尾场景翻车。选项二：只用慢系统（VLM 全程推理），准但慢到不能用。选项三：大部分时间用快系统，只在"拿不准"的时候请慢系统出山。FASIONAD 选的是三，真正的技术含量不在"快慢"这个比喻（Kahneman 的书 2011 年就出了），而在两个具体问题的答案：**"拿不准"怎么量化？慢系统的输出怎么喂回快系统才不添乱？**

第一个问题的答案是全文最漂亮的一处：不对轨迹做不确定性估计（那很难），而对 **reward 做不确定性估计**。快系统本来就要给每个候选轨迹打分（safety/comfort/efficiency/economic 四项加权），作者把 reward 看成随机变量，用 Laplace 分布建模：

$$p(\text{Reward}\mid\Theta) = \prod_{t=1}^{T}\frac{1}{2b}\exp\left(-\frac{\|r_t - \hat{r}_t\|_1}{b}\right)$$

选 Laplace 不选 Gaussian 的理由写得很坦率：L1 的重尾对 outlier 鲁棒，驾驶场景里 reward 信号本来就噪。切换逻辑是：reward 高且不确定性低 → 快系统直接走；否则 → 唤醒慢系统。用大白话讲：**快系统给自己的每个选项打了分，如果分数普遍很低、或者分数忽高忽低拿不准，就摇人**。

但这里有个论文没交代的缺口：$\hat{r}_t$（预测的 reward）和尺度参数 $b$ 是从哪来的？Reward 模型 $F_{Reward}$ 输出的是确定性的标量分数，Laplace 建模需要一个预测分布——这通常意味着要有一个额外的 head 预测 $b$，或者用 ensemble/dropout 估计。但正文和附录都没有给出这个不确定性估计模块的训练目标。式（8）是直接"宣布"出来的。这是最需要怀疑的一处：**切换机制是全文的承重墙，但它的地基（$b$ 怎么学、阈值怎么定）在论文里是空的**。阈值的具体数值同样没有报告。

## 快系统：GenAD 的生成式骨架 + 一个四项 reward

快系统的输入输出记号很干净：$(I_t, C_t) \to W_t$，多视角图像 + 高层导航指令 → BEV 轨迹点序列。内部是三段式：

1. **实例 token 建模**：BEVFormer 提 BEV 特征 $B \in \mathbb{R}^{bs\times H\times W\times C}$，然后 agent token $T_A$、map token $T_M$ 分别与 BEV 做 cross-attention（式 11），再把 agent+ego token 自注意力融合后与 map token 做 cross-attention（式 12），得到实例级的场景表示。这套是 VAD 那一路的配方。
2. **轨迹生成器**：抄 GenAD 的生成式框架——先学一个未来运动场的先验分布 $p(Z\mid I(B')) \sim \mathcal{N}(\mu_f, \sigma_f^2)$（式 16），再自回归地生成轨迹 $p(W_{T+1}\mid W_T,\dots,W_{T+f-1}, Z)$。生成 $N_C \times N_K$ 个候选（导航指令数 × 每个指令 top-K 采样）。
3. **Reward 模型选优**：$F_{Reward} = \alpha_{safety}C_{safety} + \alpha_{comfort}C_{comfort} + \alpha_{efficiency}C_{efficiency} + \alpha_{economic}C_{economic}$。附录把每一项展开了：$C_{safety}$ 含碰撞指数衰减 $C_{coll} = e^{-d_{coll}/\sigma_{coll}}$、方向对齐 $C_{deviation} = \sum_i(1-\cos(\theta_i-\theta_{target}))$ 等；$C_{comfort}$ 是横纵加速度+离心力；$C_{efficiency}$ 是目标速度偏离+时间；$C_{economic}$ 甚至有个油耗项 $k_v\cdot V + c_v$。四个 $\alpha$ 权重和里面十几个子权重——**正文一个数值都没给**。这是典型的"reward 工程"，调参空间巨大，复现者只能自己重新调。

训练 loss 也是拼装的：$\mathcal{L}_{fast} = \lambda_{plan}\mathcal{L}_{plan} + \lambda_{det}\mathcal{L}_{det} + \lambda_{seg}\mathcal{L}_{seg}$，三个 $\lambda$ 同样没给数值。

## 慢系统：VLM 不看数字，看"画出来的轨迹"

慢系统只吃多视角图像：$I_t \to (P_t, A_t)$，输出 planning state 和元动作。设计上有两个真正用心的地方。

**第一，visual prompt：把数字翻译成 VLM 能懂的语言。** VLM 不擅长处理坐标数字（"waypoint 在 (3.2, -1.5)"这种输入对它是噪音）。作者把快系统生成的轨迹点**投影回前视相机图像上**，画出来，让 VLM 看图说话——"这条画出来的线会不会压线、会不会撞到那个行人"。这是个人类直觉很强的设计：人看导航也是看那条蓝线画在哪，而不是读坐标。再配一个 BEV prompt（3D 框画在俯视图上），补空间关系。附录 A.2 的例子很具体：雨天路口，输出 $P_t = [1,1,1]$（行人/邻车/红灯三个 bit 全亮）、元动作 [Stop, Wait, Prepare Turn]、文字反馈"Stop at crosswalk. Wait for pedestrians to clear."

**第二，planning state 是 K 维二进制向量。** 通过 Yes/No 问答管线得到（"Is there a car ahead?"→1/0）。二进制的好处是**可验证、可压缩**——VLM 的长篇大论被蒸馏成一串 bit，下游只用消费 bit，不用解析自然语言。这是对 VLM 幻觉的工程性防御：幻觉通常藏在修饰语里，bit 向量里没有修饰语的位置。

慢系统的训练也值得一说：不用人类反馈做 RLHF（InstructGPT 路线），而是用**自动生成的 planning state / 元动作真值**做 reward-guided tuning：$\mathcal{L}_{rvlm} = \text{Reward}(s_{1:T_i})\cdot \Phi(s_{T_i}\mid s_{1:T_i-1})$，PPO + masking，token 级监督改成序列级——因为"整句话才有意义，单个 token 没有"。最终 $\mathcal{L}_{slow} = \lambda_{MLE}\mathcal{L}_{MLE} + \lambda_{rvlm}\mathcal{L}_{rvlm}$。思路是对的：驾驶领域不缺真值（仿真器里什么都有），没必要花钱请人标偏好。

## 反馈：信息瓶颈是"翻译官"，不是"传声筒"

慢系统的输出怎么回灌快系统？直接拼接 VLM 的 embedding 会引入大量噪音（VLM 描述里一半是废话）。作者用 information bottleneck 做蒸馏：

$$\mathcal{L}_{KD} = \sum \log q_d(y_t\mid y_i) - \beta\, KL(q_e(y_i\mid z_{current})\parallel p(z))$$

目标函数就是经典的 $ \max I(y_t; y_i) - \beta I(y_i; H)$：保留对决策有用的互信息，掐掉和输入的冗余相关。编码器/解码器是 3 层 MLP（hidden 512，LeakyReLU）。蒸馏后的语义向量 $A_{distill}$ 通过 multi-head attention 更新 ego token：$T'_E = MHA(T_E, A_{distill}) + T_E$，然后快系统用更新后的 token 重新生成轨迹。

元动作则走另一条路：可学习的 embedding 表 $e_A \in \mathbb{R}^{N_A \times d_A}$，元动作→embedding 一一映射，ego token 做 cross-attention 的 query 去查这张表。两条路（planning state 走 IB、元动作走 embedding 表）在图 3 里画得很清楚。

消融表 3 证明这两条路是互补的：IB-only（L2 0.74/碰撞 0.21%）、HA-only（0.77/0.19%）、合起来（0.69/0.18%）。表 4 证明 prompt 设计重要：Full.P（0.69/0.18%）vs Simple.P（0.80/0.32%）——prompt 写得越细，VLM 越准。

## 数字：先看表，再看正文——正文写错了

**开环 nuScenes（表 1）**。两行要分开看。带星号的行用了 ego 状态输入：FASIONAD* 平均 L2 0.28 米、碰撞 0.09%，DriveVLM* 0.31/0.10%。不带星号的行（纯视觉）：FASIONAD 0.69/0.18%，GenAD 0.91/0.43%，VAD-Base 1.22/0.53%。

两个提醒。第一，星号行的提升有水分：Li 等人 CVPR 2024（"Is ego status all you need?"，本文引用 [27]）已经证明，ego 状态输入会显著抬高开环分数——模型 essentially 在学"按当前速度外推"。所以 0.28/0.09 这个 headline 数字要打折，公平的比较是无星号行：0.69 vs GenAD 0.91，提升 24%，依然是赢，但没那么夸张。第二，**正文第 7 页写"FASIONAD achieved…lowest average L2 distance of 0.80 meters and…collision rate of 0.32%"——这个数字和表 1 的任何一行都对不上**，它恰好等于表 4 里 Simple.P 那一行的数字（0.80/0.32%）。这是把消融表里最差一档的数字抄进了正文，属于写作事故。按表 1，无星号主结果是 0.69/0.18%。

**闭环 CARLA Town05 Short（表 2）**。DS 64.83%、RC 89.04%，对比 VAD 64.29/87.26、Agent-Driver 64.31/87.31、ST-P3 55.14/86.74。赢是赢了，但幅度很小（DS 只比 VAD 高 0.54 个点）。而且正文 4.1 节有句话很关键："we implement a rule-based wrapper around the learning-based policy, following standard practice in benchmark evaluations. This wrapper helps minimize infractions during testing."——**闭环分数里含了一个手写的规则安全 wrapper**。所有对比方法 presumably 也都加了 wrapper（说是 standard practice），但 wrapper 的具体规则没公开，DS 数字里有多少是 wrapper 的功劳、多少是 FASIONAD 的，分不出来。

**FPS 6.9**（RTX 3090，表 1 脚注 †）。号称"competitive inference speed"，但这是快系统的速度。慢系统（VLM）被触发时的延迟、触发频率——论文里一个字没提。一个双系统论文不报慢系统的开销，就像混动车不报发动机介入时的油耗。FASIONAD++ 的摘要里依然只报开环数字。

**结论里的"10-15% reduction in collision metrics"**：对不上任何一对基线。无星号行 0.18% vs GenAD 0.43% 是 58% 降幅；星号行 0.09% vs DriveVLM* 0.10% 是 10%。大概是拿星号行比的 DriveVLM*，但正文没写清楚。

还有一个 abstract 里承诺、实验里没兑现的东西："we introduce a new benchmark derived from the nuScenes dataset, designed to distinguish between fast and slow scenarios"——结果部分没有任何一张表是这个 benchmark 的。图 6 定性对比了 fast/slow pathway 在三类场景下的表现，但那不是 benchmark。**承诺的 benchmark 不存在**，这是摘要里最不诚实的一句。

## 怀疑式阅读：这篇论文真正缺的三块拼图

**第一，切换阈值和 $b$ 的学习机制缺失**（前面已详述）。一个双系统架构的论文，切换逻辑是灵魂，灵魂的超参数一个没给。这让人怀疑：实验里的"自适应切换"到底是 Laplace 不确定性驱动的，还是调参调出来的固定阈值碰巧 work？没有阈值敏感性分析，无从判断。

**第二，慢系统触发频率未知。** 如果慢系统在 5% 的帧被触发，6.9 FPS 的快系统 + 偶发 VLM 调用，整体实时性勉强可接受；如果 30% 的帧都触发，系统就退化成"慢系统为主"，快系统的存在就没有意义。论文没有报这个数字，而这是评价双系统设计的**第一性指标**。FASIONAD++（8 页短文）同样没报。

**第三，reward 模型的十几个权重是调参黑箱。** $F_{Reward}$ 里四项 $\alpha$ 和每项内部的子权重（$W_{coll}, W_{dist}, W_{deviation}, W_{speed}, W_{lat}, \dots$）全部未公开。附录 A.1 把公式展开得很漂亮，但没有数值就没有复现。考虑到切换机制和最终选优都依赖这个 reward，整篇论文的实证部分是建立在一组私有超参数上的。

公平地说，这篇论文也有做得漂亮的地方。Visual prompt（把轨迹画出来给 VLM 看）是个真洞见，它承认了 VLM 的模态短板而不是假装不存在；planning state 二进制化是对 VLM 幻觉的诚实防御；reward-guided VLM tuning 用自动真值代替人类反馈，在驾驶领域是合理的（仿真器就是无限真值机）。这三处都是"把问题想清楚了"的痕迹，不是堆料。

## FASIONAD++ 一眼：续作补了什么

arXiv:2503.08162，8 页，同一拨人。三个增量：(1) 动态切换机制保留；(2) information bottleneck + high-level plan feedback 做慢系统指导；(3) 双向知识交换（visual prompt 增强慢系统推理、慢系统反馈精炼快系统）。新增 Bench2Drive 验证。开环数字：L2 降 6.7%、碰撞率降 28.1%（相对谁没说清楚，摘要语焉不详）。本质上是把 v1 附录里的 IB 和 QA-reward 机制扶正为主贡献，框架没换。引用时可以把两篇并引，v1 讲框架、++ 讲 Bench2Drive 上的验证。

## 与 RAMP 的血缘 / 划界

**这是本批五篇里和 RAMP 的"调度-校验"思想血缘最近的第二篇（仅次于 DualAD），但血缘点不一样。**

FASIONAD 给 RAMP 留了三笔遗产：

第一，**"慢系统不碰方向盘"的架构戒律**。FASIONAD 的慢系统（VLM）输出的是 planning state 和元动作——是**建议**，不是**控制**。最终轨迹永远由快系统生成。这和 RAMP "LLM 只做调度与校验、不直接输出轨迹"的戒律是同一条。RAMP 的 related work 可以直接引用 FASIONAD 作为"慢系统只给反馈"的先例，然后指出自己的推进：FASIONAD 的慢系统是单次 VLM 问答，RAMP 的慢路径是"LLM 调度器 + 世界模型前向 + 情景记忆检索"的组合慢系统，且校验是闭环的（对每条候选轨迹做风险打分），而 FASIONAD 的反馈是开环一次性的。

第二，**不确定性驱动的切换**。FASIONAD 用 reward 分布的 Laplace 不确定性决定何时唤醒慢系统；RAMP 用世界模型 rollout 的碰撞风险决定何时触发校验/切换规划器。两者都是"用一个学出来的标量门控昂贵的慢路径"，但门控信号的语义不同：FASIONAD 的门控是**认知不确定性**（我拿不准），RAMP 的门控是**风险**（这会撞）。RAMP 可以借 FASIONAD 的 Laplace 建模思路来处理风险估计本身的不确定性（risk of risk），这是个具体的、可写进 future work 的技术点。

第三，**信息瓶颈压缩**。FASIONAD 用 IB 把 VLM 的冗长输出压成决策相关的紧凑表示；RAMP 面临同样的问题——世界模型中间分布、情景记忆检索结果都是高维输出，调度器消费不了。FASIONAD 的 $\max I(y_t;y_i) - \beta I(y_i;H)$ 目标可以直接借用到 RAMP 的"记忆检索结果压缩"上。这是个方法论级别的迁移，不是故事级别的。

**划界**：FASIONAD 的快慢是"同一个规划器的两种模式"（快慢都输出轨迹/轨迹修正），RAMP 的分层是"调度器选规划器、校验器否决轨迹"——FASIONAD 没有**多规划器**的概念，也没有**记忆**。另外 FASIONAD 的慢系统是被动触发的（不确定性高才唤醒），RAMP 的校验是主动常开的（每条轨迹都要过风险校验）。"被动建议 vs 主动校验"是两篇论文在 related work 里可以正面交锋的一句话。
