# 候选轨迹的"计划生育"：ManeuverFlow 如何用固定预算养出最好的假设

> **全局编号**：169
> **论文**：ManeuverFlow: Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving
> **作者**：Yuhao Wei, Yong Chen, Chen Yang, Yingchen Wang, Delin Ouyang, Lingfeng Qi, Qi Lan, Zhanhua Zhang, Jie Li, Guofa Li（重庆大学 + 吉利汽车研究院人工智能中心）
> **发表**：arXiv 预印本 arXiv:2610.13409（v1，2026-10-09；本文按 v1 精读，共 17 页含附录）
> **代码**：已公开，https://github.com/wyh-autonomy/maneuverflow（2026-10-13 核实：1 star，仓库创建于 2026-08-18，早于论文提交；描述为 "Official implementation of ManeuverFlow-TF for end-to-end autonomous driving on NAVSIM v1"）
> **一句话核心洞见**：候选轨迹不该是"撒一大把再筛"，而该是"按场景需求定额分配"——先用场景自适应的机动先验把有限的候选名额派给最相关的驾驶意图，一步生成；再用**一份**与候选无关的未来 BEV 预测做共享证据库，每条候选只去"查"证据而不重复"算"未来；最后用同一份证据做 PDM 因子打分和 Frenet 残差精化。三步（假设-验证-改进）共享同一套证据，是全文承重的设计。

## 图 2 先讲了一个残酷的事实

论文的 Figure 2 值得先看，因为它是整篇论文的"底牌"。作者冻结了生成器，在 12,146 个 NAVSIM 场景上做了个事后分析：候选池从小到大嵌套扩大，看"池子里最好的那条轨迹"的 PDMS 怎么变。曲线单调上升——**池子越大，里面藏着的好轨迹越好**。HumanAgent（人类司机）94.8 是条虚线参考。

这个观察的潜台词很扎心：你现在的规划器不是"想不出好轨迹"，而是"好轨迹在池子里，但你的候选预算太小没把它生出来，或者生出来了没把它选出来"。于是问题被重新定义为**固定预算下的假设生命周期管理**：怎么在 N=64 这样的小预算里，让每一条候选都值得出生、都被公平地验证、都有机会被改进。这就是标题里 "Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement" 的由来——生成、评估、精化，一个假设的完整一生。

## 前人在哪两条路上撞墙了

Figure 1 把现有范式画成三张小图，作者的分类比引言的文字更锋利：

**(a) 无显式模态结构的学习查询。** 一堆 learned queries 直接回归轨迹，没有"这条 query 负责左转、那条负责直行"的分工。结果是 mode collapse：queries 在训练中挤到轨迹空间的同一个区域，生的候选互相重复。这就是 DiffusionDrive 这类工作的隐痛——名义上多模态，实际上几条轨迹长得差不多。

**(b) 固定 anchor + 逐候选未来 rollout。** WoTE 是代表：每个 anchor 候选都跑一遍循环 BEV 世界模型做未来预测，再打分。未来预测是"按候选重复算"的，K 个候选就是 K 次未来预测，计算量随候选数线性涨。这就是作者说的 scaling bottleneck 的一半：**想多生（大候选集）就得多算**。

另一半 bottleneck 在生成侧：扩散/流模型迭代采样，小候选集覆盖不了机动空间，大候选集生成和打分都贵。MeanFuser 倒是把生成压缩到一步 MeanFlow，但它的 ARM（自适应重构模块）把多个 proposal 融合成**一条**输出轨迹——最初的多个假设在下游评估、精化、选择环节不再是显式的候选，等于把"多假设"这个接口自己掐死了。

所以作者要同时回答两个问题：怎么**生**（场景自适应、模态分明、一步生成），和怎么**验**（未来证据只算一次、按候选查）。

## M-GMP：给候选名额做"计划生育"

Maneuver Gaussian Mixture Prior 是全文第一个关键设计。它的输入是场景，输出是"这 M 个机动类别各分几个候选名额"。

先说这 M 个类别怎么来的：作者把专家轨迹按**位移、平均航向、速度趋势**三个运动学量聚成 M 个"机动家族"（motion families），每个家族在归一化差分空间里拟合一个固定因子高斯。注意这里的"机动"不是语义标注（没有"左转""变道"的人工标签），而是**轨迹空间里的运动模式**——位移大+航向变化大+减速，大概对应"路口转弯"；位移大+航向不变+匀速，大概对应"直行"。这是用数据自己长出来的分类，不是人定义的。

推理时，场景相关的机动权重 $\pi_n^{man} = \text{Softmax}(D^{man}(q_n^{man}, c_n^{SN}))$ 决定每个类别分几个名额：直道场景，直行类分走大头；路口场景，转弯类名额上升。Table 5 显示 top-3 路由的 recall@3 达到 98.30%——目标机动类别有 98% 的概率落在选中的前 3 个类别里。

和 KMeans-8 的对照（M4 vs M_KM4）说明问题：M-GMP-8 比 KMeans-8 高 0.54 PDMS。KMeans 也是聚类，但它是在**静态**轨迹空间聚的；M-GMP 的权重是**场景条件**的——同一个"右转"类别，在直道场景和路口场景分到的名额不一样。这是"场景自适应"四个字的全部含义。

## M3F：一步把噪声"运"成轨迹

Maneuver-Mixture MeanFlow Field 负责把 M-GMP 采样出的源状态，一步 transport 成轨迹。技术底座是 Improved Mean Flows（Geng et al. 2026）的 boundary-field + JVP 构造，作者的驾驶特异性贡献有三处：运动结构化的 M-GMP 源、场景条件的类别分配、**机动路由的专家网络**（maneuver-routed experts）。

最后一点值得展开：DiT Block 里的前馈网络不是共享的，而是按机动类别路由到不同的专家。直行专家的权重和转弯专家的权重是两套参数。这意味着"左转"这个意图的生成逻辑和"直行"是解耦的——一个类别的训练信号不会污染另一个。这是对 (a) 类方法 mode collapse 的结构性回答：不是靠损失函数去"鼓励多样性"，而是靠**参数隔离**让多样性成为架构的默认属性。

Figure 4 的架构细节：插值后的轨迹状态 token 先后查询 BEV+ego 特征、agent 记忆，时间条件 AdaLN 调制每个 block，机动编码选择对应的前馈专家。推理时每个先验走一次完整区间 M3F 求值：$\tau_{n,i} = \Gamma^{-1}(e_{n,i} - u_\theta(e_{n,i}, 0, 1 | C_n, w_{n,i}^{man}))$，$\Gamma^{-1}$ 把位移累积成轨迹并用 atan2 解码航向。

消融表里 M2 vs M1（+0.44 PDMS，参数量已对齐 Dense-XL）说明：**光是参数多不解释增益，机动路由的结构才解释**。

## 一份未来，多次查询：Trajectory Evidence Memory

这是全文第二个承重设计，也是和 RAMP 最有对话关系的地方。

WoTE 的做法是每个候选跑一遍世界模型做未来 rollout。ManeuverFlow 反过来：**世界模型每个规划步只预测一次**与候选无关的未来 BEV 特征场 $F_n^b = \{B_{n,j}^b\}_{j=1}^{H_B}$（$H_B=4$，1 秒间隔）。然后每个候选 $\tau_{n,i}$ 用一个学到的 Trajectory-Guided BEV Feature Probe，把自己的路径点投影到当前 BEV 和时间对齐的预测状态上，沿路径采样、聚合成**轨迹专属的证据** $E_{n,i}$。

用一句话说：**未来只算一次，证据按需查询**。特异性来自"查"（轨迹条件的检索），不来自"算"（候选条件的预测）。这直接回答了 2.2 节末尾提出的关键挑战："从共享预测中提取可区分的、轨迹专属的证据，而不重复做候选级 rollout"。

训练时的一个细节：共享感知 backbone 在在线路径编码当前观测，复用去编码未来观测做 stop-gradient 目标分支。未来 BEV 损失是 Smooth-L1 + 余弦对齐（绝对特征）加 Smooth-L1（时间残差），只在有效空间重叠 $\Omega_{n,j}$ 上算。

## Metric Prism：把打分拆成五个 PDM 因子

打分器不直接输出一个总分，而是预测五个 PDM 因子：NC（无碰撞）、DAC（可行驶区域合规）、EP（进度）、TTC（碰撞时间）、Comf（舒适性），再用固定的 log-PDM 组合 $C^{PDM}$ 合成总分。候选先被编码成 score token，做集合级自注意力（候选之间互相比较），再 cross-attend 到各自的证据记忆。

为什么拆因子而不是端到端学一个总分？两个原因：一是**可解释性**——选错了能定位到是哪个因子在捣乱；二是**与评测对齐**——NAVSIM 的 PDMS 本来就是这五个因子的组合，学因子等于直接学评测的语言。消融显示未来证据在固定候选预算下带来 +1.58 PDMS / +2.29 EP——注意 EP 的增益比 PDMS 大，说明未来证据主要帮的是"敢往前开"（进度），而不是"别撞"（安全因子本来就接近饱和，NC 98.42）。

## ECFR：用同一份证据做局部精化

Evidence-Coupled Frenet Refiner 拿**同一份**证据记忆 $E_{n,i}$，预测路径对齐 Frenet 坐标系下的局部残差 $\Delta_{n,i}^F$，更新轨迹。选中的 R=N/4 个候选被精化结果替换（保留原机动索引作为 provenance），其余保留；然后**全部**重新查询共享未来场、Metric Prism 联合重打分，最高分者胜出。

这里有个微妙的设计选择：精化后的轨迹要**重新查询证据、重新打分**，而不是直接沿用精化前的分数。因为 Frenet 残差改变了路径，旧证据已经不对齐了。这个"改完重验"的闭环，是 "Hypothesize–Verify–Improve" 里 Improve 和 Verify 的第二次握手。

消融：ECFR 带来 +0.65 PDMS / +0.95 EP，NC 微降 0.10——精化让车更敢开，代价是极小的安全边际让渡。作者诚实地报告了这个 trade-off。

## 怀疑式审读

**1. 附录的 "Claim Boundaries" 是全文最诚实的部分，值得先读。** 作者自己写了三条边界：top-3 路由可能漏掉稀有或组合行为；定性例子和学到的 PDM 因子既不证明跨方法优越性、也不证明安全性；**与候选无关的未来场不建模对自车不同计划的交互式响应**——也就是背景车不会因为你的轨迹改变而改变行为。这是共享未来预测的内在代价：WoTE 的逐候选 rollout 至少在建模"如果我这么开，别人会怎样"，ManeuverFlow 的共享场是"不管我怎么开，未来都一样"。作者在 2.2 节批评过 RiskWorld 的同构复用"稀释动作特异性"，但自己的共享未来场在交互维度上是同样的妥协——只不过特异性从"查证据"这一步找补回来一部分。

**2. 基线选择有个时间差问题。** Table 1 里最强的几个基线（MeanFuser CVPR 2026、VADv2 ICLR 2026、DriveSuprim AAAI 2026）都是 2026 年的工作，而 ManeuverFlow 是 2026-10 的 arXiv 预印本。作者用的是"official results"而非重跑——在"同一个官方 benchmark 版本下"这个限定语下是可接受的，但 NAVSIM 版本迭代快，跨版本数字不可比，引用时要注意这个限定。

**3. 两阶段训练的 λ 系数。** 联合损失 $L_{joint} = \lambda_{aux}L_{aux} + \lambda_{M3F}L_{M3F} + \lambda_{verify}L_{verify} + \lambda_{ref}L_{ref}$，正文没有给出四个 λ 的取值和敏感性分析。四个目标之间有没有拉扯（比如生成目标要多样性、打分目标要集中性）？论文没讨论。

**4. 轻量 backbone 的叙事。** 作者反复强调 ResNet-34 TransFuser 是"轻量"的，Proposal FPS 在 N=64 时 53.86。但这是"感知+M3F"的 FPS，不含世界模型未来场预测和 Metric Prism/ECFR 的开销——Table 5 的 FPS 定义是 "perception–M3F"，端到端延迟没有完整数字。车端部署的时延账单是缺的。

**5. 代码刚公开，1 star。** 仓库创建于 2026-08-18（早于论文提交），10-13 刚更新过。官方实现只有 TF 变体在 NAVSIM v1 上的代码，LTF 和 WorldEngine 的部分未见。复现前建议先看仓库的 README 更新到什么程度。

**6. 做得漂亮的地方也要说。** 附录主动写 Claim Boundaries、在顶会投稿里是少见的诚实；消融表的控制做得干净（Dense-XL 参数量对齐、KMeans-8 先验对齐、MCur4 未来分支对齐），每个增益都能定位到具体设计；Figure 2 的冻结池分析是个好的"底牌"——先用数据证明问题存在，再给方案。

## 与 RAMP 的血缘/划界

**血缘**：ManeuverFlow 和 RAMP 在"世界模型只算一次、多次复用"这个效率直觉上是同路人。它的"一次预测未来 BEV 场 + 逐候选查询证据"和 RAMP 的"单次前向中间分布复用"是同一个大问题的两种解法：**未来预测的边际成本应该摊薄**。它是 RAMP 在 related work §2(b) "world models for decision-making" 簇里最接近的公开邻居（比 SimVTA 更近，因为 SimVTA 是训练时后训练，ManeuverFlow 是推理时验证）。

**划界**（三条，写进论文够用）：

1. **单系统 vs 调度架构**。ManeuverFlow 是一个端到端的单一网络：生成、评分、精化全在里面，没有"派单"这个概念；RAMP 是推理时 agentic 架构，LLM 调度异构规划器，规划器本身可以是规则型、学习型、扩散型。ManeuverFlow 回答"怎么把一个网络做强"，RAMP 回答"怎么把多个网络管好"。

2. **同构消费者 vs 异构消费者**。ManeuverFlow 的共享未来场服务的是同构的一组候选（都是打分）；RAMP 的单次前向中间分布服务的是决策链上下游两个异构消费者（LLM 调度先验 + 独立风险验证）。这是 RAMP 创新点 B 的充要划界，ManeuverFlow 恰好是"差一格"的例子：它有共享，但消费者是同构的。

3. **无记忆**。ManeuverFlow 没有情景记忆模块——它的"记忆"是 Trajectory Evidence Memory，指单步内沿轨迹聚合的证据，不是跨时间的经历库。RAMP 的风险加权衰减记忆在它射程之外。

**一句话定位（可直接进 related work）**：ManeuverFlow (Wei et al., 2026) 一次生成多意图候选轨迹，将世界模型预测的未来特征沿轨迹聚合为证据做 PDM 因子评分与 Frenet 残差精化；与 RAMP 的区别在于它是单一端到端系统，无异构规划器调度与情景记忆，且其共享未来场不建模对自车计划的交互式响应。
