# 录像带里藏着更好的开法：FPD-Drive 让"看过未来"的老师给学生出多道标准答案

> **全局编号**：170
> **论文**：FPD-Drive: Future Policy Distillation for End-to-End Autonomous Driving
> **作者**：Li-Heng Chen¹²（共同一作）, Qifeng Chen¹（共同一作）, Chengwei Wu¹, Yingjie Zhang¹, Sheng Yang¹, Hongbo Fu², Shaoqing Ren¹³（¹蔚来 NIO，²香港科技大学，³中国科学技术大学）
> **发表**：arXiv 预印本 arXiv:2610.13232（v1，2026-10-09；本文按 v1 精读，共 18 页含附录）
> **代码**：已公开，https://github.com/countnemochan/fpd-drive（2026-10-13 核实存在；项目页 https://countnemochan.github.io/FPD-Drive/）
> **一句话核心洞见**：一段驾驶录像只记录了人类司机实际开的那一条轨迹，但录像带的未来帧里藏着"这个场景还可以怎么开"的信息——让一个能看未来帧的 teacher 先把录像看完，为同一场景生成多条经 PDM 评估的高质量轨迹，再把这些"标准答案"分给学生的不同 proposal；学生这辈子没见过未来，却学会了多种开法。打分器另有一套"纠错"机制，专门修安全边界附近排错序的毛病。

## 一盘录像带只教你一种开法，这就是问题

端到端驾驶的训练数据有一个结构性缺陷，论文引言把它摆在最前面：每段驾驶录像只记录了**实际执行的那一条轨迹**。同一个路口，人类司机选择了"早减速、慢转弯"，但"晚减速、快转弯"可能同样合法——录像带里没有第二条轨迹告诉你。

这个缺陷为什么致命？因为现在的主流规划器都是多模态的：一次生成 M=64 条候选轨迹，再选一条执行。如果你只用单条人类演示做监督，64 条候选里只有离演示最近的那一两条能拿到有效梯度，其余的要么被当成"错"压下去，要么拿不到任何信号。**多模态的生成器，配的是单模态的监督**——这就是作者要推翻的那个默认假设。

前人有两条修补路线。Hydra-MDP 把评测指标蒸进多个打分头；CLOVER、DriveSuprim 用预定义轨迹词表做多目标监督。但预定义词表是"人工手写"的动作模式，场景自适应性差——词表里没有的开法，永远学不会。FPD-Drive 的回答是：别手写，让**看过录像未来帧的 teacher**来出题。

## 作者手里的三张牌

从论文的组织方式看，作者开工前有三张底牌，每张都降低了这件事的风险。

第一张是**现成的骨架**。Teacher 和学生共享一套 coarse-to-fine 结构，直接扩展自 DrivoR（Kirby et al., CVPR 2026）：proposal decoder 粗生成 + DiT 残差精化 + scorer 打分。作者不用从零设计网络，全部精力押在"监督信号"这个单一变量上。Table 4 的第一行就是证据：光骨架（无未来、无蒸馏）已经有 94.0 PDMS——起点本身就高。

第二张是**免费的特权信息**。训练录像的未来帧本来就在硬盘上躺着，不用额外采集、不用标注。Privileged learning（Vapnik 的 LUPI）在驾驶里不是新概念（Learning by Cheating, CoRL 2020），但之前多用于"用真值路网/BEV  privileged teacher"。FPD-Drive 的特权是**录像的未来观测本身**——更廉价，也更贴近"如果当时多看两秒会怎么开"的直觉。

第三张是**现成的裁判**。NAVSIM 的 PDM-Closed evaluator 可以给任意轨迹打分（NC/DAC/EP/TTC/Comf 五个分量）。Teacher 生成的候选好不好，不用人看，跑一遍 evaluator 就知道。这解决了多目标监督里最难的一环：**怎么判断 teacher 生成的"第二答案"是不是真的好**，而不是瞎编的。

三张牌里最硬的是第三张：有一个自动化的、可复现的质量裁判，整个"teacher 出题"流程才站得住。没有它，多目标蒸馏就是无源之水。

## Teacher 怎么"看未来"：global scene encoder

Teacher 和学生的唯一区别，在于 teacher 多了一个 global scene encoder $F$。它把当前帧的 scene tokens $h$ 和未来帧的视觉 tokens 拼在一起：

$$f = F(h, \bar{E}(F_t))$$

其中 $F_t$ 是录像中记录的未来多帧图像，$\bar{E}$ 是复用视觉编码器抽的 future tokens。$F$ 用 spatial、temporal、cross-attention 三种注意力把"现在"和"未来"融合成一份全局场景表示 $f$，这份表示通过 cross-attention 同时喂给 teacher 的 refiner 和 scorer。

直觉上可以这样想：普通 planner 看到的是"路口现在没车"，teacher 看到的是"路口现在没车，但 2 秒后右侧会杀出一辆车"。于是 teacher 生成的候选里，会天然包含"提前减速让行"这种**当前帧看不出来的开法**——而这正是学生要学的。

附录 Table 8 做了未来 token 注入位置的消融，很说明问题：只注入 trajectory decoder，94.13→94.26（几乎没用）；只注入 DiT refiner，→95.29（大头）；只注入 scorer，→94.79。**未来信息最值钱的地方是"精化"和"打分"，而不是"粗生成"**。这符合直觉：粗 proposal 本来就是发散的、多样的，不缺想法；缺的是把想法打磨好、把好坏分清楚——而这正是需要"预知未来"的地方。

## 多目标怎么"分"：选目标、配对子、三个损失

这是全文承重的机制，分三步走。

**第一步：选目标。** Teacher 为每个场景生成 M=64 条候选，PDM-Closed evaluator 逐条打分。先过安全过滤（NC/DAC 不合格的扔掉），然后贪心选：第一条取分数最高的，之后每条取"与已选集合的最小 ADE 最大"的那条，且 pairwise ADE 至少 1 米。K=4 是每场景目标上限，实际平均选出 $\bar{J}=3.1$ 条（Table 5a）。

这里有两个设计判断值得停下来看。第一，**为什么用 ADE 做多样性度量而不是别的**：ADE 是轨迹空间的欧氏距离，简单、可算，且 1 米的阈值在驾驶场景里大致对应"换了一条车道"或"明显不同的速度剖面"的量级——作者没明说，但这个阈值的选择决定了"多样性"的粒度。第二，**贪心而不是最优**：max-min 贪心是子模优化里的经典近似，$1-1/e$ 近似比，对 K=4 这种小集合足够好，没必要上组合优化。

**第二步：配对子。** J 个 teacher 目标 $Y=\{y_j\}$ 要分给学生的 M 个 proposal。这里用最优二分匹配（匈牙利算法的连续松弛，实际用 scipy 实现）：

$$\pi^* = \arg\min_{\pi \text{ injective}} \sum_{j=1}^J d_{xy}(b_{\pi(j)}, y_j)$$

$d_{xy}$ 是对应位置的平均欧氏距离。关键点是 **injective（一对一）**：每个 teacher 目标只许配给一个 student proposal，不许两个目标挤到同一个 proposal 上。这保证了学生的不同 proposal 被推向**不同的**行为模式——多模态的覆盖是靠"分灶吃饭"实现的，而不是靠"大家都往最高分挤"。

没配上目标的 proposal 怎么办？配给它 ADE 最近的那个 teacher 目标（保留已匹配的不动），用来构造 DiT 的 noisy residual 输入——**所有 M 条候选都参与去噪，但只有 J 个配对子进监督损失**。这是一个精巧的安排：去噪过程见过全分布，监督信号只来自高质量子集。

**第三步：三个损失。**

$$L_{prop} = \frac{1}{J}\sum_j d_1(b_{\pi^*(j)}, y_j)$$

$L_{prop}$ 是 proposal 阶段的 L1 轨迹损失，$d_1$ 是每步位置+航向角的绝对误差之和的均值。注意这里用的是 $d_1$（L1）而不是匹配时用的 $d_{xy}$（L2）：匹配要的是"找最近的"，L2 对大偏差更敏感、找得准；监督要的是"学得像"，L1 对离群点更鲁棒——两个阶段用不同的距离，是有意的。

$$L_{diff} = \mathbb{E}_{k,\epsilon}\frac{1}{HJ}\sum_j \|\hat{\epsilon}_{\pi^*(j)} - \epsilon_{\pi^*(j)}\|_F^2$$

标准的扩散去噪损失，DiT 预测噪声。$H=8$ 是轨迹步数。

$$L_{traj} = \mathbb{E}_{k,\epsilon}\frac{1}{J}\sum_j d_1(\hat{\tau}_{\pi^*(j)}, y_j)$$

精化后轨迹的 L1 损失。三个损失分工：$L_{prop}$ 管"粗生成别跑偏"，$L_{diff}$ 管"去噪过程学对"，$L_{traj}$ 管"最终输出像目标"。Table 4 显示光加 FPMT（无 SASC）就从 94.1 干到 94.7——多目标蒸馏是第一大增益来源。

## 打分器怎么"纠"：Safety-Aware Score Correction

多目标解决了"生成"，但"选择"是另一个战场。论文指出了 scorer 的两种典型错误，都发生在**安全边界附近**：

**错误一：不安全的候选排得太高。** 几何上和安全轨迹只差几十厘米，但 NC=0（撞了）。SASC 的做法是：对每个高分但不安全的候选 $\tau^-$，在同一场景里找 ADE 最近的、通过安全检查的 $\tau^+$，组成 pair，加一个 margin loss：

$$L_{neg} = \sum_{r \in S}\frac{1}{\max(1,|P_r|)}\sum_{(\tau^-,\tau^+)\in P_r} \text{softplus}(1 - \ell_r(\tau^+) + \ell_r(\tau^-))$$

$S=\{\text{NC}, \text{DAC}\}$ 只盯安全分量。$\ell_r$ 是该分量的 logit。这个公式读起来是：让安全候选的 logit 比不安全的高出至少 1（margin），否则就罚。用 softplus 而不是 hinge，是为了梯度处处光滑——margin 满足了之后梯度自然衰减，不会像 hinge 那样硬截断。

**错误二：安全的候选被低估。** 通过了全部安全检查，但 scorer 给的安全概率低于 0.9。加一个正标签 BCE：

$$L_{pos} = \sum_{r \in S}\frac{1}{\max(1,|H_r|)}\sum_{\tau \in H_r} \text{softplus}(-\ell_r(\tau))$$

0.9 这个阈值是作者定的——低于 0.9 就算"低估"。Table 11 显示两个 loss 都开，unsafe case 率从 1.91% 降到 1.35%，PDMS 94.7→94.9。**SASC 的增益不大（0.2 PDMS），但它修的是安全，不是进度**——unsafe case 率降了 30%，这个数字比 PDMS 更值得看。

## 数字怎么读：三个 benchmark，一个比一个难

**NAVSIM v1 navtest（PDMS）**：FPD-Drive 94.9，Pro 版 95.5。Table 1 里 Human Driver 是 94.8——base 版刚好摸到人类线，Pro 版超过。注意 DriveZero-Scale 也有 95.3，差距在 0.2 以内；这个量级的差距，换一组随机种子可能就抹平了（论文没报告种子方差，这是缺的信息）。

**NAVSIM v2 navhard（EPDMS）**：FPD-Drive 55.9，Pro 58.1。这里要慢下来看：v2 的 EPDMS 和 v1 的 PDMS **不是同一个口径**。v2 加了 DDC（逆行）、TLC（闯红灯）两个乘性因子，舒适度拆成 HC/EC，还引入了两阶段评估（第二阶段周围车辆会对自车行为做反应）。55.9 这个数字看起来比 94.9 低很多，不是因为模型变差了，是尺子变严了。跨版本对比数字没有意义，论文把两个表分开是对的。

**HUGSIM 零样本闭环**：RC 55.3 / HD-Score 45.1（345 个场景，平均值）。这是全文最有信息量的数字，因为 NAVSIM 是开环（non-reactive）评估，而 HUGSIM 是真闭环。FPD-Drive 在 HD-Score 上 45.1，比第二名 DriveZero 的 39.4 高出一截——说明蒸馏来的多目标监督，泛化到了训练没见过的闭环交互里。但"preliminary results on 345 scenarios"这个标注要留意：345 个场景、没有难度分层的显著性分析，HD-Score 的方差未知。

**时延**：FPD-Drive 88.4/98.0 ms（mean/p99），DrivoR 58.2/58.5 ms。多出来的 30ms 主要来自 DiT 的 8 步 DDIM 去噪。Table 12 显示去噪步数从 8 加到 12，PDMS 94.9→94.9（零增益）——**8 步是拐点，再多就是纯烧钱**。附录 F 作者自己承认了 iterative DiT sampling 的开销，未来方向是单步去噪（点名 MeanFuser）。

## 怀疑式审读

**1. README 和论文的数字对不上。** 官方仓库 README 写 base 版 v2 EPDMS 是 54.75，论文 Table 2 和摘要写的是 55.9。94.88→94.9 是四舍五入，54.75→55.9 不是。两种可能：README 是旧 checkpoint 的数字，或者论文用的是另一次运行。1.15 个点的差距，在 v2 这个尺度上不算小（base 和 Pro 的差距也就 2.2）。引用 v2 数字时建议以论文为准，但要知道仓库里躺着一个更低的数。

**2. PDM-Closed 既是裁判又是考官。** Teacher 目标用 PDM-Closed evaluator 筛选（Table 4：PDM-Closed 选出的目标质量 99.6，learned scorer 选的只有 95.6），学生用这些目标训练，最后在 NAVSIM 上用同一套 PDM 指标评估。这是一个**评估指标内生化**的闭环：模型在"考 PDM 试"这件事上，复习资料就是 PDM 出题人给的。公平地说，Hydra-MDP、DriveSuprim 都是这个路数，这是领域通行做法；但读者应该知道，94.9 里有一部分是"对评估器的拟合"，换到真实世界（没有 PDM evaluator 当裁判的地方），增益可能会打折。HUGSIM 的零样本结果（45.1 vs 39.4）是作者给出的外部验证，方向是对的，但 345 个场景的 preliminary 结果还不够硬。

**3. Pro 版的增益有多少来自数据，论文没分离。** FPD-Drive-Pro 加了 158.8K 生成场景（视频生成管线 + VLM 评估器过滤），v1 从 94.9→95.5。Figure 3 的 scaling 曲线显示分数随数据量单调涨——那 Pro 的 0.6 个点里，多少是蒸馏方法的功劳，多少是"多喂了 158.8K 数据"的功劳？论文没有"Pro 数据 + 无蒸馏"的对照组。这个消融的缺失，让方法增益和数据增益搅在一起。

**4. 三阶段训练的工程代价。** Stage 1 训 30 epoch，Stage 2 训 teacher 10 epoch，Stage 3 冻 teacher 再训学生 10 epoch，全程 8×H800。Teacher 的目标还要离线 cache。论文没给总墙钟时间，但三阶段 + 离线目标生成，复现门槛不低。好在推理时只要学生（123.3M 参数），部署代价是干净的。

**5. λ 权重没交代。** 总损失 $L_{student} = \lambda_{prop}L_{prop} + \lambda_{diff}L_{diff} + \lambda_{traj}L_{traj} + \lambda_{metric}L_{metric} + \lambda_{neg}L_{neg} + \lambda_{pos}L_{pos}$，六个 λ 的取值正文和附录都没给。六个损失项量级不同（L2 的去噪损失和 BCE 的打分损失天然不在一个尺度），权重调不好某一项会直接哑火。这是复现时最可能踩的坑，论文没说。

**6. 做得漂亮的地方也要说。** Table 5b 的对照做得干净：self-distillation（无未来）vs FPD（有未来），在两种目标筛选标准下未来特权都赢——这就把"增益来自未来信息"和"增益来自多目标"两个变量分开了。附录 F 的 Limitations 只有三句话但都在点上（DiT 开销、长时交互、极端天气/实车未验证），没有藏着掖着。Figure 2 的定性对比把 latency（mean/p99）直接标在图上，时延诚实。

## 与 RAMP 的血缘/划界

**血缘**：FPD-Drive 和 RAMP 在"候选轨迹的生成与选择"这个第四轴上是邻居。它的 Safety-Aware Score Correction 修的正是 RAMP 风险仲裁要解决的那类问题——安全边界附近的排序错误。RAMP 的 related work §2（候选评估簇）引用它，位置是合适的：一个"训练时把选择能力蒸进打分器"的代表。

**划界**（三条，不贬低）：

1. **训练时蒸馏 vs 推理时调度**。FPD-Drive 的全部智能都发生在训练时：teacher 看未来、学生学多目标、打分器被校正，部署时就是一个 123.3M 的前馈网络，一次前向出结果。RAMP 的智能发生在推理时：LLM 每周期调度规划器、世界模型每周期做前向、记忆每周期检索。FPD-Drive 回答"怎么把一个网络训强"，RAMP 回答"怎么在运行时把多个模块管好"。

2. **未来只用一次 vs 未来每步都用**。FPD-Drive 的未来观测是训练集里的录像，推理时不存在；它的"未来知识"被压缩进了学生权重。RAMP 的世界模型在推理时每周期都跑，未来预测是**在线**的。这意味着 FPD-Drive 处理不了训练分布外的突发（权重是 frozen 的），而 RAMP 的在线预测+记忆检索是为长尾准备的——代价是推理时延（88ms vs RAMP 的 LLM 调度开销）。

3. **无记忆、无调度**。FPD-Drive 没有情景记忆模块，没有多规划器调度的概念。它的多目标是"多条轨迹"，RAMP 的多规划器是"多个决策者"。这是两个正交的维度：FPD-Drive 训出的强单规划器，理论上可以作为 RAMP 规划器池的一员——和 ReCogDrive 的定位是一样的（"优化个体"可被"治理团队"收编）。

**一句话定位（可直接进 related work）**：FPD-Drive (Chen et al., 2026) 用能访问录像未来帧的 teacher 为同一场景生成多条经 PDM 评估的高质量轨迹目标蒸馏给学生，并用安全感知校正修打分器在安全边界附近的排序错误；与 RAMP 的区别在于它的未来知识只在训练时使用、部署时为单一前馈网络，无在线世界模型、无情景记忆、无多规划器调度。
