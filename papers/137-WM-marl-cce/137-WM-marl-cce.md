# 当世界模型学会"勾心斗角"：MARL-CCE 的竞争均衡交通仿真

> 信息块
>
> - **论文**：MARL-CCE: Modelling Competitive Behaviors in Autonomous Driving Under Generative World Model（全称 Modelling Competitive Behaviors in Autonomous Driving Under Generative World Model）
> - **作者**：Guanren Qiao, Guorui Quan, Rongxiao Qu, Guiliang Liu*（香港中文大学（深圳）, University of Manchester；*通讯）
> - **发表**：ECCV 2024（venue 已用多份世界模型综述的 ECCV 2024 列表核实；论文 PDF 来自 ECCV 2024 官方 ecva.net，编号 05085）
> - **链接**：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05085.pdf（ECCV 官方；无 arXiv 版本，**引用时注意**）
> - **代码**：https://github.com/qiaoguanren/MARL-CCE（仓库已更名为 Multi-agent-Competitive-Autonomous-Driving，17 stars，最后更新 2025-07-15；公开，可复现）
> - **一句话核心洞见**：交通仿真器不应该只会"模仿人类开车"——人类开车是互相博弈的（抢道、加塞、互不相让）。这篇论文把交通仿真写成一个**一般和博弈**（general-sum Markov game），用生成式世界模型当"沙盘"，在里面用多智能体 RL 求**粗相关均衡（CCE）**，让仿真车学会"自私但理性"的竞争行为，从而生成长尾里的安全关键事件。

---

## 先讲一个所有人都见过的场景

早晚高峰的四车道，一个路口。四辆车同时到达，来自四个方向，各自都想"第一个走"。人类司机的真实行为是什么？**互相试探**：往前蹭一点，看别人让不让；不让就再蹭一点；最后总有一个均衡——可能是谁先动了大家跟着动，可能是僵持几秒。这不是"合作"，也不是"随机"，这是一种**均衡**：给定别人都这么开，我单方面换个开法也占不到便宜。

传统的交通仿真器（TrafficSim、ITRA 等）是怎么做的？**行为克隆**：在人类驾驶数据上训一个轨迹预测器，让它模仿。问题有两个：第一，人类数据里"互相僵持、极限博弈"的长尾事件本来就少，模仿学不会；第二，模仿出来的车是"各开各的"，不会对自车的行为做反应——你别它一下，它照样按原轨迹走。这就是论文引言里那句最狠的话：仿真器"擅长建模那些自动驾驶已经解决的场景，却建模不出那些还没解决的关键场景"（paraphrase）。

作者的解法：**别模仿，直接博弈**。把每个车当成自私的智能体，给它一个目标（尽快到达目的地），让它们在一个学好的世界模型里互相博弈，博弈的均衡点就是"竞争行为"的刻画。

---

## 背景补讲：三个博弈论概念，大白话

**一般和博弈（general-sum game）**：每个玩家有自己的奖励函数，不存在"你的 gain 就是我的 loss"。交通就是典型的一般和博弈：我想快点到，你也想快点到，但我快不一定就让你慢（比如我变道走了，你反而更顺了）。对比零和博弈（如下棋，你赢我就输），一般和博弈的均衡更难求——因为"合作"和"竞争"混在一起。

**纳什均衡（NE）**：每个玩家都独立选策略，谁单方面改策略都占不到便宜。NE 的问题在交通里有两个：第一，它要求策略独立，但人类开车是看别人脸色行事的（相关性）；第二，一般和博弈里 NE 可能有很多个，算法不保证找到"好的"那个。

**粗相关均衡（CCE, Coarse Correlated Equilibrium）**：NE 的放宽版。允许玩家们的策略**相关**（比如都参考了同一个"交通信号"——前车动了我才动），只要求"事前"没有玩家想单方面偏离。打个比方：路口的四个司机互相看，达成一种默契"红车先走"。只要这种默契下没人想单方面打破（打破了反而更慢），它就是 CCE。CCE 比 NE 好求（存在性、收敛性都更友好），而且更符合人类驾驶的"看脸色"本质。论文公式 (3) 的定义：$\max_i [V^{\dagger,\pi_{-i}}_{0,i}(s) - V^{\pi}_{0,i}(s)] \le \epsilon$——"最想偏离的那个玩家，偏离后能多赚的也不超过 ε"。

**Dec-POMDP（部分可观测多智能体 MDP）**：每个车只能看到局部（自己的观测 $o_i$：地图、历史轨迹、邻居信息），动作是位置变化 $(\Delta x, \Delta y)$。这是把"交通"翻译成 RL 语言的标准框架。

---

## 作者的底牌

**第一张：QCNet（2023）的世界模型骨架是现成的。** 4.1 节的世界模型（transition + observation + 候选动作预测器）本质上是在 QCNet 的多模态轨迹预测架构上加了一个"世界模型"的帽子：用 cross-attention 建模 agent-map、agent-agent、时序关系，用 Laplace 混合分布预测候选动作。作者不需要从零发明多智能体建模，站在 QCNet 肩膀上。

**第二张：CCE-V-Learning（Jin et al. 2022）和 Magnetic Mirror Descent（Sokota et al. 2023）提供了"能收敛到 CCE"的理论工具。** 这是全文理论部分最硬的两块引用：optimistic Bellman update（给没去过的状态加探索奖励 $r^{opti} = r + \alpha_t(s)$，$\alpha_t(s) = c/m(s)$，去得少的地方奖励多）保证探索；magnetic mirror descent（目标函数里加两项 KL：离模仿策略 $\pi^o_i$ 别太远、离上一轮策略 $\pi^{j-1}_i$ 别太远）保证" fidelity"（别学飞了，不像人）和稳定。两者拼起来，得到 CCE-MASAC 算法。

**第三张：Argoverse 2 的真实场景是现成的"考场"。** 实验用的 4 个场景（逆行、四路口交汇、T 型路口汇入、密集四路口）都是从 Argoverse 2 里挑的 hold-out 场景，真实、有代表性。这比在 CARLA 里手搭场景更有说服力。

---

## 方法：世界模型当沙盘，CCE-MASAC 当玩家

整个方法分两步，和 Think2Drive 有异曲同工之妙，但目的完全不同。

**第一步：训一个生成式世界模型（4.1 节，论文图 1）。** 三个部件：

1. **Transition model**：输入上一步隐状态 $s_{t-1}$ 和所有车的动作，输出 $s_t$。实现上是：Map Encoder 把地图编码成 $z^c$；Agent-to-Map cross-attention 把地图信息融进每辆车的历史轨迹。隐状态 $s_t$ 本质上是"过去 W 步所有车的时空特征拼起来"。
2. **Observation model**：$o_{i,t} = f^{MLP}_i(z^A_{i,t})$，把隐状态解码成每辆车的局部观测。
3. **候选动作预测器（Candidate Action Predictor）**：这是最有特色的一块。Agent-to-Temporal cross-attention（历史→现在）+ Agent-to-Agent cross-attention（邻居→自己），然后 **K 个策略网络的混合**：每个候选策略输出一个 Laplace 分布的参数 $(\mu^k_{i,t}, b^k_{i,t})$，动作概率是 $\pi_{i,t}(a|s) = \sum_k \omega^k_i p_L(a|\mu^k_{i,t}, b^k_{i,t})$。**为什么用 Laplace 而不用高斯？** Laplace 分布的尾巴更重（指数衰减 vs 平方衰减），对"突然急刹、猛打方向"这种长尾动作的建模更友好——而这正是安全关键事件里的动作。训练损失（公式 4）：选和真值最像的那个 mode（best-mode loss）+ 选对 mode 的分类损失（cls loss）。

注意这个世界模型的训练是**纯模仿**（在离线数据 $D_o$ 上做监督学习），它学会的是"人类一般怎么开"。博弈的部分在第二步。

**第二步：在世界模型里求 CCE（4.2 节）。** 固定世界模型，用 CTDE（集中训练、分布执行）框架独立更新每个车的 actor 和 critic。目标函数（公式 6/8/10/11，一路化简）：

$$\max_{\pi_i \in \Pi^c_i} \mathbb{E}\left[\sum_t \gamma^t r^{opti}_i(o_t,a_t) - \frac{1}{\eta_1}D_{KL}(\pi_{i,t}\|\pi^o_{i,t}) - \frac{1}{\eta_2}D_{KL}(\pi_{i,t}\|\pi^{j-1}_{i,t})\right]$$

三项各管一件事：第一项是乐观奖励（探索没去过的地方，找均衡）；第二项是"别离人类开法太远"（fidelity，$\pi^o$ 是世界模型里学到的模仿策略）；第三项是"别离上一轮太远"（稳定）。化简到公式 (11) 后，它长得像 maximum-entropy RL，于是可以用 soft actor-critic 的套路实现：soft policy evaluation（公式 12，学 Q）+ soft policy improvement（公式 14，更新策略）。Laplace 策略的 reparameterization trick：$a = \mu - b \cdot \mathrm{sgn}(\tilde{\epsilon})\ln(1-|\tilde{\epsilon}|)$，$\tilde{\epsilon} \sim U[-1,1]$——这是 Laplace 分布的逆 CDF 采样，类比高斯的 $a = \mu + \sigma\epsilon$。

**直觉**：想象 4 辆车在一个学好的"沙盘"里反复博弈。每辆车都想"快点到"，但都受两条缰绳牵着：别开得太不像人（KL to 模仿策略），别一轮比一轮变得太激进（KL to 上一轮）。最后收敛到的，就是"自私但 rational"的 CCE——抢道、加塞、互不相让，但都在人类行为的合理范围内。

---

## 实验：数字的条件

**CCE-gap**（5.1 节，图 3）：衡量"离均衡有多远"——固定其他车，算第 i 辆车最优偏离能多赚多少。CCE-MASAC 在 4 个场景下 CCE-gap 最小，收敛最快。对比基线：MASAC（无 optimistic/mirror descent）、MAPPO（不稳定，难收敛到 CCE）、QCNet/GameFormer（纯模仿，不考虑竞争，CCE-gap 大）。**这个对比设计是公平的**：QCNet/GameFormer 本来就不是为均衡设计的，用 CCE-gap 去量它们属于"用别人的短板比自己的长板"，但作者的本意是说明"模仿学不到竞争"，这个结论是成立的。

**世界模型的有效性**（5.2 节，图 4）：在学好的世界模型里，单智能体 SAC 的折扣累积奖励超过行为克隆（QCNet），说明世界模型是一个"有效的 RL 环境"。有意思的是，CCE-MASAC 的奖励**低于**单智能体 SAC——作者的解释是"竞争降低了每个个体的奖励"，这恰恰是博弈的特征：均衡点不是个体最优，是"谁也占不到便宜"的点。这个"奖励更低但更真实"的观察是全文最有味道的一个实验细节。

**可视化**（5.3 节，图 5）：4 个场景的 CCE 轨迹：抢道、激进汇入、路口拥堵。定性展示，没有量化指标——是 showcase，不是证据。

**实验设置的诚实度**：3 个随机种子，报告均值±标准差；场景是 hold-out（不在预训练数据里）。附录 B/E 有超参和训练细节。但**没有报告训练墙钟时间和样本量**——世界模型+MARL 两阶段的训练代价未知。

---

## 怀疑式阅读

**1. "世界模型"的叫法有水分。** 4.1 节的"生成式世界模型"本质上是一个**监督学习的多智能体轨迹预测器**（QCNet 骨架 + Laplace 混合），它的"transition"是在隐特征上做的，没有显式的物理状态转移，也没有奖励/终止预测（不像 Think2Drive 的 RSSM 有 reward predictor 和 termination predictor）。它更像一个"神经仿真器"，而不是 Dreamer 意义上的世界模型。RAMP 引用时要注意这个区别。

**2. CCE 的"理论保证"只在离散环境有证明。** 4.2 节说 optimistic V-learning "proven to converge to CCE under discrete environments"，然后"we extend this algorithm to solve continuous decision-making problems"——**这个 extend 是没有证明的**，是启发式的。论文没有藏（原文写了 extend），但读者要意识到理论保证在连续动作空间是断掉的。

**3. 评估指标 CCE-gap 的计算依赖"最优偏离策略"。** 5.1 节说 $\pi^\dagger_i$ "can be computed by leveraging the prior knowledge in reward designs"（附录 A）。也就是说，CCE-gap 的" ground truth 最优"是用奖励设计的先验知识算出来的——如果这个先验不准，CCE-gap 的绝对值就不可信。好在论文主要用它做**相对比较**（谁更小），相对排序对先验误差不那么敏感。

**4. 只建模竞争，不建模合作（作者自己承认，5.4 节 Limitations）。** 真实交通是竞合混合的（汇入时的让行就是合作）。CCE-MASAC 的均衡点是"人人自私"，可能会**高估** агрессивность（攻击性），生成的长尾事件比真实更极端。用它做仿真器训规划器，有"把规划器训得过度保守"的风险。

**5. 17 stars 的小众仓库，社区验证少。** 代码公开了（2025-07 还在更新），但引用和复现很少。实验结论的可信度主要靠论文本身的写作质量（3 种子、hold-out 场景、诚实的 limitations）。

**做得漂亮的地方**：5.4 节的 Limitations 是本批 5 篇里写得最诚实的——"没考虑约束""没考虑合作"两条都点了，没有藏。图 4 的"CCE-MASAC 奖励低于单智能体 SAC"是一个反直觉但真实的观察，作者没有为了"我们的奖励最高"而回避。这些细节说明作者对自己在做什么有清醒的认识。

---

## 与 RAMP 的血缘 / 划界

**血缘**：MARL-CCE 是"世界模型当交互式交通仿真器"的代表。它和 RAMP 的关系不在"规划"，而在**"世界模型里的时间"**：RAMP 的风险校验需要 world-model rollout 出"自车候选轨迹下，周围车会怎么反应"——这正是一个**交互式**的 rollout，不是单车运动学外推。MARL-CCE 的世界模型（agent-to-agent cross-attention 建模他车对自车的反应）是这种交互式 rollout 的思想参照。

**划界**，三条：

1. **世界模型的用途**：MARL-CCE 的世界模型是**离线博弈的沙盘**——在里面求均衡、生成长尾数据，是"数据工厂"；RAMP 的世界模型是**在线决策的校验器**——每次规划都做一次前向，是"安全气囊"。前者离线、重博弈、慢；后者在线、重风险、快。RAMP 不需要求均衡，只需要"给定候选轨迹，他车最可能/最危险的反应是什么"——这是一个比 CCE 轻得多的问题。

2. **博弈假设**：MARL-CCE 假设所有车都是自私智能体（一般和博弈）；RAMP 的风险校验可以假设他车是"部分合作、部分自私"的混合（比如 IDM 式的反应模型），不需要完整的博弈求解。这是 RAMP 相对它的简化，也是实时性的来源。

3. **表示**：MARL-CCE 的世界模型输出的是轨迹分布（Laplace 混合），RAMP 的是隐分布 $q(z|o)$。如果 RAMP 想在隐空间里建模他车反应，MARL-CCE 的 agent-to-agent attention 结构（4.1 节）是可以借鉴的模块设计——"把交互建模从轨迹空间搬到隐空间"。

一句话：MARL-CCE 回答"怎么生成足够难的交互场景"，RAMP 回答"在线怎么判断当前候选够不够安全"——前者是出题人，后者是阅卷人。RAMP 的测试场景库如果用 MARL-CCE 这类方法生成长尾交互场景做压力测试，会是一个漂亮的实验设计。
