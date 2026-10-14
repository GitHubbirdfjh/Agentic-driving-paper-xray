# 把"复盘"焊进参数：Reflect-VLA 如何用自蒸馏替代 RL 的试错

> **全局编号**：171
> **论文**：Reflect-VLA: Internalizing Reflective Guidance via Dual-Path On-Policy Self-Distillation for Autonomous Driving
> **作者**：Yuechen Luo*, He Chong*, Fang Li*, Shaoqing Xu*†（共同一作，Xu 为项目负责人）, Hexin Zhang, Ruiqi Zhang, Ziying Song, Lei Yang, Zhi-xin Yang✉, Fuxi Wen✉（清华大学、澳门大学、中国科学技术大学、北京交通大学、南洋理工大学）
> **发表**：arXiv 预印本 arXiv:2610.13453（v1，2026-10-09；本文按 v1 精读，共 14 页含附录）
> **代码**：官方仓库已建 https://github.com/luo-yc17/Reflect-VLA（2026-10-08 创建，0 star；2026-10-13 核实：仅含 README.md + 两张图片，**无训练/推理代码**；另有项目页 reflect-vla-web）
> **一句话核心洞见**：RL 用标量奖励告诉模型"这条轨迹不行"，却没说"哪里不行、该怎么改"；作者发现只要把评估结果翻译成显式的诊断＋纠正指令（Reflective Guidance），同一模型第二遍就能开得好得多——但推理时没条件跑两遍。于是把"第二遍"当作训练时的特权信息：学生自己 rollout，PDMS 只负责把轨迹分成"好的自留、坏的送修"，坏的用 teacher 的 token 级纠正分布做反向 KL 蒸馏。 dense 的纠正信号替代了 RL 的稀疏试错，post-training 快了一个数量级，推理时单遍生成。

## Figure 1(a) 是全篇的底牌

论文开篇没有先讲方法，而是先摆了一个观察：同一个 VLA 模型，直接推理时开出一条有问题的轨迹；但如果把"这条轨迹的评估结果＋诊断＋纠正建议"拼进输入再让它生成一遍，第二遍的轨迹明显更好。Figure 1(a) 画的就是这个对比。

这个观察的潜台词是：**模型不是不会开，是第一遍"没想起来"**。评估反馈里藏着它本来就具备、但单遍生成时没调用出来的规划知识。于是问题从"怎么把模型训得更强"变成了"怎么把第二遍的能力内化到第一遍里"。这是一个典型的特权信息（privileged information）设定：训练时可以作弊（多跑一遍＋看评估），推理时必须单遍。

## 标量奖励到底缺了什么

要理解作者为什么不满 RL，得先看清 RL 在驾驶 VLA 上是怎么用的。GRPO 这类方法：模型 rollout 若干条轨迹，NAVSIM 打分（PDMS），分数高的加权、分数低的压低。PDMS 本身是五个因子的组合（NC×DAC×加权和），但进到 RL 损失里时，它就是一个标量。

标量的问题在长尾场景被放大：一辆车在路口犹豫，是"没看懂红绿灯"（场景理解错）、"选了左转但该直行"（行为选择错）、还是"方向对了但转弯半径太大"（轨迹执行错）？这三种失败可能拿到同一个 0.62 分。**分数能排序，不能归因；能惩罚，不能指路。** 模型要在参数空间里自己反推出"0.62 分到底是因为什么"，这就是 RL  sample-inefficient 的根源——不是模型笨，是监督信号太粗。

作者的回答是把评估反向翻译成语言：不是"0.62 分"，而是"TTC 裕度不足＋建议纵向减速 0.3m/步"。这就是 Reflective Guidance。

## Reflective Guidance：四个组件，但它是模板写的

Guidance 由四个部分组成，按顺序实例化：

1. **Trajectory Verdict（裁决）**：把评估器输出翻译成"这条轨迹有没有问题"的明确判断，附 PDMS 各因子数值做定量上下文；
2. **Planning Diagnosis（诊断）**：定位主导失败原因——碰撞风险、可行驶区域违例、TTC 裕度不足、进度不够、轨迹不对齐；安全类失败优先于效率类；能定位到具体 waypoint 的就定位到；
3. **Expert Alignment（对齐）**：把初始轨迹和参考轨迹的几何差异翻译成定性调整——横向保不保持、纵向加不加速、终点航向修不修，并带上导航指令保证纠正方向不跑偏；
4. **Refinement Directive（指令）**：把诊断和对齐结果变成有序的纠正计划——先解决碰撞/TTC（拉开安全距离），再解决可行驶区域（拉回车道），最后微调横向/纵向/航向，并指定一个主要关注点（场景理解、横向控制或纵向控制）。

这里必须诚实地指出一个关键事实（附录 Q4 亲口承认）：**这套 guidance 不是 VLM 生成的，是确定性模板按规则拼出来的**。输入是评估器分数、几何差异、导航指令，输出是填好槽位的文本。没有引入第二个 VLM，也不需要人工标注。

这句话两面都要读。好的一面：方法不依赖更强的外部模型，复现门槛低，"诊断"能力来自评估器和几何计算，是白盒的。弱的一面：guidance 的质量天花板就是模板设计者的领域知识——它能说出的纠正指令，不会超出模板预设的失败类型。如果真实失败模式超出模板覆盖（比如需要理解交警手势这种模板没写的），guidance 就给不出有用的纠正。这也是为什么作者不敢把 guidance 留在推理时——它本质上是一个"评估器＋规则"的包装，而真正的学习发生在蒸馏那一步。

记号上，基础输入是 $q^{base} = (I^{cam}, q^{cmd}, q^{ego}, q^{his})$（前视图像、导航指令、自车状态、历史轨迹），带 guidance 的输入是 $q^{gui} = (q^{base}, o^{init}, g)$（式 3）。轨迹本身被转成相对位移的离散 token：每步 $(x,y,h)$ 各一个 token，分辨率 0.01（0.01m / 0.01rad），$x$ 取值范围 [-1.5, 15.0] 对应 1651 个 token——词汇表不小，但注意这是三个独立的 token 流，不是 1651×1001×101 的联合词汇。

## Dual-OPSD：好的自留，坏的送修

蒸馏这一步是全文承重的设计。流程：

1. 学生从 $q^{base}$ 做 on-policy rollout：$o^S \sim \pi_\theta(\cdot|q^{base})$（式 6），注意是**贪心解码**（temperature=0），不是采样——作者要的是学生"当前最相信"的那条轨迹，不是分布探索；
2. NAVSIM 评估器打分 $r$，按阈值 $s=0.85$ 路由（式 7）：$r \ge s$ 进 $O^+$（可靠），$r < s$ 进 $O^-$（风险）；
3. 可靠轨迹走 **self-preservation**：$\mathcal{L}_{preserve} = -\sum_t \log \pi_\theta(o^S_t|q^{base}, o^S_{<t})$（式 8）——就是拿自己生成的轨迹当标签再学一遍，目的是"别把已经会的东西改坏"；
4. 风险轨迹走 **distillation**：冻结的 teacher（同一模型，但输入是带 guidance 的 $q^T = (q^{base}, o^S, g)$）给出 token 级分布，学生最小化反向 KL：$\mathcal{L}_{distill} = \sum_t D_{KL}(\pi^{stu}(\cdot|q^{base}, o^S_{<t}) \parallel \pi^{tea}(\cdot|q^T, o^S_{<t}))$（式 9）；
5. 总目标 $\mathcal{L}_{Dual-OPSD} = \mathbb{1}\{r\ge s\}\mathcal{L}_{preserve} + \mathbb{1}\{r<s\}\mathcal{L}_{distill}$（式 10）。

三个设计选择值得细看。

**为什么用反向 KL 而不用前向 KL？** 附录 Table 10：反向 91.5 vs 前向 91.0。反向 KL 是 mode-seeking：学生被推向 teacher 分布的峰值，压制低概率的"野路子"。在这个场景下是对的——teacher 的纠正分布里，峰值就是"该怎么改"的明确指令，长尾是噪声；前向 KL 会逼学生覆盖 teacher 的整个分布，把纠正指令稀释了。但反向 KL 有 mode collapse 的固有风险，这里靠 $O^+$ 路径的 self-preservation 对冲：好的行为不进蒸馏，也就不被压扁。

**为什么 teacher 的分布只取 top-64？** Table 9：k=16 时 90.2，k=64 时 91.5，k=256 和全词汇表都是 91.4–91.5。纠正信号集中在 teacher 分布的头部，截断到 64 既省计算又不丢信息。这是个工程上干净的结论。

**为什么阈值 s=0.85？** Table 11：s 从 0.65 升到 0.85，PDMS 从 91.1 涨到 91.5——更多次优轨迹被送去"返修"；但 s=0.95 时掉到 91.2——连本来开得不错的轨迹也被 teacher"指手画脚"，反而破坏。0.85 是"该修的都修、不该动的不动"的分界线。这个阈值是 PDMS 绝对分数，不是百分位——换数据集或评估器版本可能要重调，论文没讨论这点。

## Teacher 的三段式构造：解决"学生犯的错，老师没见过"

Self-distillation 有一个经典坑：teacher 只在离线数据上见过"别人犯的错"，学生 rollout 时犯的是"自己的错"，分布对不上，teacher 的纠正就是隔靴搔痒。作者用三段式训练把 teacher 一步步推向学生的分布：

1. **Driving-Knowledge Pretraining**：7 个驾驶 QA 数据集（LingoQA、DriveGPT4、Senna、DriveLM、NuScenes-QA、NuInstruct、DriveLMM），2 epoch——给 VLM 灌驾驶常识；
2. **Planning-and-Refinement SFT**：基础规划数据＋guidance-conditioned 的精化数据混训，2 epoch——让模型既会直接生成，也会"看 guidance 改轨迹"；
3. **Self-Refinement Rejection Sampling**：模型自己 rollout，自己挑有问题的，自己用 guidance 精化，只保留"精化后确实更好且合法"的样本再训一轮——让 teacher 见过"学生本体会犯的错"。

消融（Table 6，4B 模型）很说明问题：SFT 88.1 → +pretrain 88.5 → +RS **88.4（反而掉了 0.1）** → +OPSD 90.2 → +RS+OPSD 91.5。RS 单独用是负增益——作者解释为"在精化数据上继续训，部分遗忘了基础规划能力"。RS 的真正价值只在和 OPSD 联用时兑现（+1.3）。换句话说，**第三段不是独立有效的训练阶段，而是给蒸馏准备的 teacher 热身**。论文把它列为独立贡献略有包装之嫌，诚实读法是：它是 Dual-OPSD 的前置条件。

## 数字验证（全部来自原文表格）

**NAVSIM v1（PDMS，Table 1）**：0.8B 89.3 / 2B 90.0 / 4B 91.5。自家 GRPO 基线（Qwen3.5-RL）88.8 / 89.2 / 91.2——Reflect-VLA 分别高出 0.5 / 0.8 / 0.3。注意这个 margin 不大：0.3–0.8 个点，Table 1 的 caption 没给误差棒，4B 那 0.3 的优势是否显著不好说。

**NAVSIM v2（EPDMS，Table 2）**：0.8B 88.8 / 2B 89.5 / 4B 91.2。

**Bench2Drive（Table 3，闭环）**：4B 的 DS 80.15 / SR 59.09% / Effi 212.20 / Comf 21.92，是表中 VLA 方法里 DS 最高的（超 AutoVLA 的 78.84）。但 Comfort 21.92 偏低——和 ReCogDrive 一样，是用舒适性换进度和成功率的路子，正文对这个 trade-off 着墨不多。

**nuScenes 开环（Table 8，附录）**：4B 平均 L2 0.41m、碰撞率 0.14%，是表中 VLA 方法最好的；2B 0.52m / 0.20%；0.8B 0.73m / 0.33%。

**Teacher–学生差距（Table 4）**：guidance 加持的 teacher 比裸学生高 6.1 / 5.3 / 4.8 个点（0.8B/2B/4B）；蒸馏后学生追回 3.0 / 2.6 / 3.1。**差距没有完全闭合**：4B 的 teacher 93.2，蒸馏后学生 91.5，还差 1.7。这 1.7 就是"内化"的损耗——有些纠正能力确实需要 guidance 在场才能发挥。

**训练时间（Table 5，每 10k 样本小时数）**：这是需要最仔细读的一张表。
- 0.8B：GRPO 5.3 → OPSD 1.3，加速 4.1×，**降幅 75.5%**；
- 2B：GRPO 22.3 → OPSD 1.7，加速 13.1×，降幅 92.4%；
- 4B：GRPO 33.4 → OPSD 2.4，加速 13.9×，降幅 92.8%。

摘要里"reducing training time by over 90% compared with RL"这句话，**只在 2B 和 4B 上成立，0.8B 上只有 75.5%**。README 更直接拿 4B 的数字（33.4h → 2.4h）当标题宣传。另外注意两点：一是比较的是 post-training 阶段（per 10k samples），不是端到端全流程——Reflect-VLA 自己有 pretrain+SFT+RS+OPSD 四段，GRPO 基线的前置阶段未知，全流程账没算过；二是 GRPO 随模型增大显著变慢（5.3→33.4），而 OPSD 几乎不涨（1.3→2.4）——这才是"over 90%"背后真正的故事：**RL 的在线探索成本随模型变大爆炸，蒸馏的 teacher 前向成本几乎恒定**。这个 scaling 行为比 headline 数字本身更值得记住。

**训练配置**：基座 Qwen3.5-0.8B/2B/4B（Team 2026），8× NVIDIA H20。超参数见附录 Table 12。

## 怀疑式审读

**1. "Over 90%" 的适用范围。** 如上，0.8B 只有 75.5%。更重要的是比较口径：per-10k-samples 的 post-training 阶段对比。如果计入 Reflect-VLA 自己的四段流水线（pretrain 2 epoch + SFT 2 epoch + RS + OPSD 2 epoch），总账未必有 90%。论文没有给端到端 wall-clock 对比，这是一个缺口。

**2. Guidance 是模板，不是学出来的。** 这是理解本文分量的关键。Reflective Guidance 的四个组件是确定性规则生成的——它的"诊断"能力上限是模板作者的先验，不是模型涌现的。方法的真正创新在**路由＋蒸馏**（dual-path、on-policy、reverse KL），而不在 guidance 本身。标题把 "Reflective Guidance" 放前面有营销成分；如果把标题改成 "Dual-Path On-Policy Self-Distillation for VLA Planning"，更接近实际贡献。

**3. Teacher–学生差距没闭合。** 1.7 个点的残留说明部分纠正能力是 guidance-dependent 的，内化不完全。这对 RAMP 有启发（见下节）：有些验证能力可能注定要留在推理时。

**4. Rejection Sampling 的独立价值为负。** +RS 单独掉 0.1 个点。它存在的唯一理由是给 OPSD 准备 teacher。审稿人如果问"能不能砍掉第三段"，作者的诚实回答应该是"砍掉的话 OPSD 掉 1.3 个点"——Table 6 里 Pre+SFT+OPSD 是 90.2，+RS+OPSD 是 91.5。

**5. Comfort 的 trade-off。** Bench2Drive 上 Comf 21.92（4B），和 ReCogDrive 的 17.45 是同一路数：PDMS/DS 的增益部分来自"更敢开"。正文没有讨论这个交换是否可接受。

**6. 代码仓库是空的。** 官方 repo 建于论文提交前一天，至今只有 README 和图片。复现无从谈起，Table 5 的 GRPO 基线配置也无法核查。

**7. 做得好的地方也要说。** 消融做得干净（top-k、KL 方向、阈值、guidance 四组件、训练阶段，每个都有对照）；附录 Q&A 主动回答了"为什么不用 guidance 做推理""为什么双路径""RS 为什么必要"三个最可能被问到的问题；Figure 3/4 的 token 级可视化（teacher 在 <|x_208|> vs <|x_215|> 上的概率差）把"纠正分布"从抽象概念变成了看得见的东西——0.01m 量级的 token 选择差异，这正是 dense 监督相对标量奖励的信息量优势所在。

## 与 RAMP 的血缘/划界

**血缘**：Reflect-VLA 和 RAMP 在"评估必须产生可行动的纠正，而不只是一个分数"这一点上是同路人。Reflective Guidance（轨迹评估→显式诊断＋纠正指令）可以看作 RAMP 闭环风险校验的**训练时镜像**：RAMP 在推理时对每条候选做 rollout 风险验证并仲裁，Reflect-VLA 在训练时对每条 rollout 做评估诊断并蒸馏。两者都认为标量分数是不够的。

**划界**（三条）：

1. **训练时 vs 推理时**。Reflect-VLA 的全部 machinery（guidance、teacher、蒸馏）都发生在训练阶段，推理时是一个普通的单遍 VLA；RAMP 的 machinery（LLM 调度、世界模型复用、记忆检索、风险仲裁）全部发生在推理阶段。这是两者最本质的区别——一个把智能做进参数，一个把智能做进架构。Table 4 留下的 1.7 个点 gap 恰好说明：有些纠正能力内化不进去，必须留在推理时做，这正是 RAMP 存在的理由之一。

2. **单模型 vs 多智能体架构**。Reflect-VLA 没有规划器池、没有调度器、没有情景记忆、没有世界模型复用。它是"VLA＋训练范式"线上的一个数据点，可与 ReCogDrive 的 DiffGRPO、DriveAgent-R1 并列——三者都在回答"怎么把 VLA 训得更会开"，只是手段不同（RL 奖励 / 自蒸馏 / 推理激励）。

3. **评估器的角色不同**。Reflect-VLA 用 NAVSIM 评估器在训练时当老师（离线、可容忍）；RAMP 需要在推理时在线做风险验证（实时、不可容忍延迟）。这决定了两者对"评估成本"的敏感度完全不同。

**一句话定位（可直接进 related work）**：Reflect-VLA (Luo et al., 2026) 把轨迹评估翻译成显式的诊断＋纠正指令作为训练特权信息，用双路径 on-policy 自蒸馏（风险轨迹蒸馏 teacher 的 token 级纠正分布、可靠轨迹自模仿）内化进单遍 VLA；与 RAMP 的区别在于它的评估－纠正闭环完全发生在训练时，推理时无调度、无记忆、无独立验证。
