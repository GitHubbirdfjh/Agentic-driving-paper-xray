# 给 GRPO 造出它采不到的正样本：ELF-VLA 的失败诊断与反馈回灌

> **论文**：ELF-VLA: Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures
> **作者**：Yuechen Luo*、Qimao Chen*、Fang Li*（共同一作）、Shaoqing Xu（项目负责人）、Jiaxin Liu、Ziying Song、Zhi-xin Yang（通讯）、Fuxi Wen（通讯）；清华大学、澳门大学、北京交通大学
> **venue**：CVPR 2026（核实：CVPR Open Access 已收录该文，官方仓库描述亦标注 [CVPR 2026]；arXiv v1 自身未写录用信息）
> **arXiv**：2603.01063v1 [cs.CV]，2026-03-01
> **代码**：github.com/luo-yc17/ELF-VLA —— 仓库公开但无代码（2026-10-14 用 GitHub API 核实：仅 README + assets，0 个 Python 文件；2026-02-27 创建，2026-09-12 最后更新）
> **一句话核心洞见**：RL 卡住不是因为模型学不动，而是因为长尾场景里所有 rollout 都是零分、梯度没有方向；ELF-VLA 让一个能看见真值的 teacher 把"零分"翻译成结构化诊断报告，再让学生照着报告重新采样出高分轨迹、塞回 GRPO 的 batch——相当于给优化器凭空造出了它自己探索永远采不到的正样本，而 policy shaping 解决的是这批"借来的样本"在新策略下概率太低、梯度被方差淹没的问题。

---

GRPO 训练 VLA 到后半程会出现一个让人泄气的现象：某些场景下，不管采多少条轨迹，PDMS 全是 0。不是接近 0，是干脆的 0。梯度更新还在转，但每个样本的 advantage 都一样——全军覆没的 batch 里，分母的 std 把一切都抹平了，模型在这个场景上等于停止了学习。论文把这叫 persistent failures（持续性失败），而 Figure 1（论文第 1 页）就是拿这个现象当赌注：上半部分画的是普通 GRPO——VLA 采样、reward 全是 0、policy update 原地打转；下半部分画的是 ELF-VLA 的回答——teacher 模型介入，给出带 COT、failure、meta-action、correction 的结构化反馈，policy 照着反馈做 refinement，采出 reward=1 的样本，更新才重新有方向。

这个赌注押得准，因为它点出了 VLA+RL 这条线当时真正的痛点。SFT 之后模型的探索能力被 SFT 数据集的分布锁死了：常见场景占绝对多数，真正考验系统的长尾安全场景在数据里稀有。进 RL 阶段后，模型在这些场景上怎么采样都拿不到正奖励——探索退化成在零分区域里随机游走。更麻烦的是，就算拿到了 0 分，这个标量也说不清失败的根因：是 think 模块里高层规划错了，是对关键目标的推理错了，还是底层轨迹执行时动态跟不上？一个数字，三种完全不同的病因，对应三种完全不同的改法。标量奖励在这里不是弱监督，是几乎无监督。

![Figure 1：普通 GRPO（上）vs ELF-VLA（下）。上半部分：VLA 采样多条轨迹，reward 全是 0，policy update 原地打转；下半部分：teacher 模型给出带 COT / Failure / Meta-Action / Correction 的结构化反馈，policy 做 refinement 采出 reward=1 的样本，更新重新有方向。出自论文第 1 页。](/home/hatch/workspace/your_files/paper-xray-100/papers/180-VLA-elfvla/paper-fig1.png)

*看这张图时注意作者的措辞：上半部分的 reward 方块里是两个 0，下半部分是 0 和 1 并排——整篇论文的方法就是为了把那个 1 变出来。紫色 Refinement 那条回路是全文唯一的新增结构，其余都是标准 GRPO。*

## 作者的底牌是哪几张

从论文的组织方式看，作者手里至少有三张牌。

第一张是 LLM 文献里已经验证过的两招：Critique-GRPO（用自然语言反馈代替纯数字奖励，arXiv:2506.03106）和 LUFFY（off-policy 指导下的 shaping，arXiv:2504.14945）。这两篇一个解决"反馈太稀疏"，一个解决"借来的样本概率太低"。ELF-VLA 的方法部分几乎就是把这两招搬到驾驶 VLA 上，再加一个驾驶领域特化的 teacher 诊断格式。这张牌的价值在于：审稿人不需要相信任何新的数学，只需要相信"文本反馈在 LLM 推理任务上work"这件事能迁移到驾驶上。

第二张是实验设计里的两个对照基线：GT-GRPO（直接把真值轨迹塞进 batch）和 Rule-GRPO（用预设规则生成反馈再重采样）。这两个基线选得很老实——它们分别回答了两个最刁钻的质疑："你这增益是不是因为偷看了真值？"（GT-GRPO：直接给真值反而只有 89.2，因为真值轨迹在策略分布下概率太低、优化不动）和"你这 teacher 是不是还不如人工规则？"（Rule-GRPO：规则反馈只有 89.6，说明粗糙的自我修正不够）。ELF-VLA 的 91.0 落在这两个中间点的上方，故事就闭环了：既不是纯粹的真值蒸馏，也不是规则能替代的。这两个基线的 PDMS（89.2 / 89.6）离普通 GRPO（89.0）都很近，恰恰说明"往 batch 里加好样本"这个想法本身不稀奇，稀奇的是怎么加——加什么样的、用什么权重加。

第三张是 Figure 4（论文第 7 页）的 total-failure 比例。这张图是全文最有说服力的证据：它不比较最终分数，而是直接测量训练过程中"一个场景所有 rollout 全挂"的比例。GRPO 是 2.73%，ELF-VLA 压到 1.08%。这个指标紧扣论文的中心论点（persistent failures），而不是拿一个综合分数糊弄过去。

![Figure 4：RL 训练阶段 total-failure 样本比例。PDMS / DAC / NC 三个指标下，ELF-VLA（绿）都把"所有 rollout 全失败"的比例压到最低；PDMS 从 2.73% 降到 1.08%。出自论文第 7 页。](/home/hatch/workspace/your_files/paper-xray-100/papers/180-VLA-elfvla/paper-fig4.png)

*看这张图时注意：纵轴是百分比，PDMS 那组四个柱子是 2.73 / 2.14 / 1.88 / 1.08，分别对应 GRPO / GT-GRPO / Rule-GRPO / ELF-VLA。DAC 和 NC 组也有同样的单调下降，说明反馈机制同时改善了碰撞和可行驶区域两类硬失败，而不只是在刷综合分。*

## 两条输入线：模型既是答题者又是改错者

ELF-VLA 的 VLA 模型同时接受两种输入。base 输入 $q_{\text{base}}$ 是常规的：前视图像 $I_{\text{cam}}$、高层导航指令 $q_{\text{com}}$（Move Forward / Turn Left / Turn Right 之类）、自车状态 $q_{\text{ego}}$（速度、加速度）、过去三帧历史轨迹 $q_{\text{his}}=\{h_{t-3},h_{t-2},h_{t-1}\}$（2Hz）。feedback 输入 $q_{\text{fb}}$ 则把"上一次的作答"和"针对它的反馈"一起拼进 prompt：

$$
q_{\text{fb}} =
\begin{cases}
\langle q_{\text{base}}, o_c, f_{\text{rule}} \rangle & \text{if } o_c \quad (r_{\text{traj}} \ge s) \\
\langle q_{\text{base}}, o_w, f_{\text{teacher}} \rangle & \text{if } o_w \quad (r_{\text{traj}} < s)
\end{cases} \tag{1}
$$

$o_c$ / $o_w$ 是按 PDMS 阈值 $s$ 划分的答对/答错样本。对答对的样本，反馈是规则生成的肯定性反馈 $f_{\text{rule}}$（巩固正确行为）；对答错的，请外部 teacher（Qwen3-VL-32B）看三样东西——base 输入、错的轨迹 $o_w$、真值轨迹 $o_{\text{gt}}$——写一份结构化诊断 $f_{\text{teacher}}$。注意 teacher 是**能看见真值**的，这个细节后面会成为怀疑式阅读的重点。

这份诊断报告是全文最有"驾驶领域味道"的设计，分五段：(1) Meta Action Analysis——高层元动作对不对；(2) Think Process Analysis——think 模块里对关键目标的推理哪里错了；(3) Safety Failure Analysis——安全类失败（碰撞、可行驶区域）；(4) Efficiency Failure Analysis——效率类失败（TTC、ego progress）；(5) Actionable Correction——可执行的纠正，带横向和纵向分量。这五段恰好对应 PDMS 的子指标结构（NC / DAC / TTC / EP / Comfort），等于把标量奖励拆回了它本来的多维面孔。Figure 8（附录）把 Rule-GRPO 和 ELF-VLA 的反馈并排画出来：规则反馈是二值的"对/错 + 看真值重答"，teacher 反馈是实例相关的诊断 + 具体纠正策略。

为什么模型要同时学"答题"和"改错"？SFT 阶段的混合数据训练（式 (2)，标准的负对数似然，同时在 $(q_{\text{base}}, o_{\text{gt}})$ 和 $(q_{\text{fb}}, o_{\text{gt}})$ 上监督）就是为了让模型具备"看到反馈能改出更好轨迹"的能力。附录 6.2 交代了反馈数据集的构造：从生成轨迹里随机抽 4k 个答对的和 4k 个答错的，配上对应的反馈。没有这一步，RL 阶段的 teacher 反馈就是对牛弹琴——模型根本没学过"读诊断报告改轨迹"这个技能。

## 一个左转场景，走完全程

论文 Figure 5（第 8 页）给了一个具体的复杂左转场景，值得把数字摊开看。

基线行为：模型输出初始轨迹（红色），think 模块里写"Key Obstacle Analysis：应该注意前方 15.57 米、左侧 8.11 米处的一辆车"，meta action 是"减速并左转"。这条轨迹撞上了潜在碰撞——对关键障碍物的位置估计错得离谱。

teacher 介入：它看到真值，指出这是 "Think Process" 错误——障碍物的真实位置是**前方 11.43 米、左侧 4.11 米**。误差不是一点点：纵向差了 4 米多，横向差了 4 米。在左转场景里，4 米的横向误差就是"撞"和"过"的区别。teacher 的 Actionable Correction 给出具体纠正：减速的同时 gradual 左转、对齐车道曲率、每步横向目标位置 0.3 米、纵向速度上限 2 m/s。

Refinement：模型读到这份反馈，重新生成 think——"Key Obstacle Analysis" 改成注意 11.43 米 / 4.11 米处的车，输出修正轨迹（蓝色），绕开障碍物。

同一个场景如果走普通 GRPO：8 条 rollout 全是类似的红色轨迹，PDMS 全 0，advantage 归一化后全挤在一起，梯度≈0，模型在这个场景上什么都学不到，下一轮继续采样红色轨迹。这就是 persistent failure 的完整形态——不是学错了，是根本没在学。

## 训练管线的三步：先筛数据，再 RL

RL 阶段开始前有个容易被忽略但很关键的数据 curation。用 SFT 模型对每个 query 采样 N 条 rollout，估计均值奖励和奖励方差，**扔掉"均值高且方差低"的样本**（已经学会的简单场景），留下困难样本（均值低、方差低：稳定地失败）和模糊样本（方差高：模型最不确定的）。85k 训练条目筛到 24k。Table 5（第 7 页）显示：85k 全量 89.1 PDMS，24k 随机子集 88.9，24k 精选 91.0。作者的解释是全量数据里简单场景占主导，稀释了梯度信号。这个消融是干净的——随机 24k 和精选 24k 的唯一差别就是选择标准。

然后是带反馈的 GRPO rollout（Algorithm 1，第 4 页），三步：

1. **初始 rollout**：从 $q_{\text{base}}$ 采样 $n$ 条轨迹 $\{o_i\}$，算三项奖励。
2. **反馈 rollout**：按阈值 $s$ 把样本分成答对/答错，组装 $q_{\text{fb}}$（答错的调 teacher），模型再从 $q_{\text{fb}}$ 采样 $n$ 条修正轨迹 $\{o^{\text{fb}}_i\}$ 并算奖励。
3. **拼 batch**：从修正轨迹里随机选 $k$ 条满足 $r^{\text{fb}}_{\text{traj}} > \max(r_{\text{traj}})$ 的（比原始 batch 里最好的还好），不够 $k$ 条就复制原始最优的那条来凑。最终 batch 是 $n+k$ 个样本，在上面算 advantage 做联合优化。

第三步里"复制凑数"的细节论文一笔带过，但它有个隐含效果：被复制的原始最优样本在 batch 里出现两次，等于它的梯度权重翻倍。这是一种没被讨论的隐式 upweighting。

## 奖励函数：三项各管什么

$$r = r_{\text{traj}} + r_{\text{fmt}} + r_{\text{goal}} \tag{3}$$

$r_{\text{traj}}$ 就是 PDMS 本身，连续值 0 到 1。PDMS 的公式（附录式 (11)）：

$$\text{PDMS} = \text{NC} \times \text{DAC} \times \frac{5 \times \text{EP} + 5 \times \text{TTC} + 2 \times \text{C}}{12}$$

NC（无责任碰撞）和 DAC（可行驶区域）是乘性门控——撞了或者开出路了，整个分数直接归零；EP（自车进度）、TTC、Comfort 是加权平均。EPDMS（附录式 (12)）结构类似，门控更多（NC × DAC × DDC × TLC），加权项里加了车道保持 LK、历史舒适 HC、扩展舒适 EC。

$r_{\text{fmt}}$ 是二值的格式奖励：输出里有 `<think>...</think>` 和 `<answer>...</answer>` 两段得 0.5，轨迹点语法合法再得 0.5。它的作用是防止 RL 把输出格式训崩——VLA 的输出是文本，格式一崩后面全没法解析。

$r_{\text{goal}}$ 是按终点 L1 距离给的阶梯奖励（附录式 (10)）：距离 0–2 米给 1，2–4 米给 0.8，4–6 米给 0.6，6–10 米给 0.4，10–15 米给 0.2，超过 15 米给 0。它和 $r_{\text{traj}}$ 里的 EP 有重叠（都鼓励往前走），但它只看终点，不管过程——等于给稀疏的 PDMS 加了一个稠密一点的形状奖励（shaping）。

三项加起来最大 3。注意选择修正样本时只看 $r_{\text{traj}}$（PDMS 分量）是否超过原 batch 最大值，而 advantage 归一化时用的 $r_{\text{union}}$ 原文没有明确是总奖励 $r$ 还是 $r_{\text{traj}}$——这是个没说清的地方。

## GRPO 目标：clip 只给亲生的孩子

$$\begin{aligned}
J(\theta) &= \mathbb{E}_{D_{\text{final}} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{n}\sum_{i=1}^{n} J_i + \frac{1}{k}\sum_{j=1}^{k} J^{\text{fb}}_j - \beta D_{\text{KL}} \right] \tag{4} \\
J_i &= \min\left(c_i(\theta) A_i,\ \text{clip}(c_i(\theta), 1-\epsilon, 1+\epsilon) A_i\right) \tag{5} \\
J^{\text{fb}}_j &= f(c^{\text{fb}}_j(\theta)) A^{\text{fb}}_j \tag{6}
\end{aligned}$$

式 (5) 是标准 GRPO/PPO 的 clip 目标，只用在**原始 rollout**的 $n$ 个样本上。式 (6) 是反馈修正样本的目标：**没有 min、没有 clip**，直接用 shaped ratio 乘 advantage。为什么不对称？作者的逻辑是：修正样本本来就是"借来的"（在 $q_{\text{fb}}$ 条件下采样，却要在 $q_{\text{base}}$ 条件下优化），标准 importance ratio 在这里方差极大，clip 也救不回来，不如换一套专门的 shaping。KL 项的参考策略 $\pi_{\text{ref}}$ 是初始 SFT 模型，防止 RL 跑飞。

advantage 的计算（式 (7)(8)）有个关键选择：把原始奖励和反馈奖励**合并成一个集合** $r_{\text{union}} = \{r_j\}_{j=1}^{n} \cup \{r^{\text{fb}}_{j'}\}_{j'=1}^{k}$，用合并集的均值和标准差做归一化：

$$A_i = \frac{r_i - \text{mean}(r_{\text{union}})}{\text{std}(r_{\text{union}})}$$

这个设计的用意很直白：修正样本的高分会抬高均值，让原始的零分样本拿到更负的 advantage——"看看人家，再看看你"。归一化是相对的，修正样本的存在本身就在给原始样本提供学习信号。

## Policy shaping：把借来的样本扶上梯度

这是全文数学上最值得停下来看的地方。反馈修正样本有个先天问题：它是在 $q_{\text{fb}}$（带 teacher 诊断的 prompt）条件下生成的，但在优化目标里却要按 $q_{\text{base}}$（无反馈）条件算概率。一个在"开卷"条件下很容易采样到的好轨迹，在"闭卷"条件下概率可能极低。标准 ratio $c = \pi_\theta / \pi_{\text{old}}$ 在这里会剧烈震荡，导致梯度爆炸、训练不稳定。

受 LUFFY 启发，作者把反馈样本的 ratio 换成一个有界 shaping（式 (9)；原文中 $f$ 的定义式排版错乱，但式 (9) 右端形式明确）：

$$f(c^{\text{fb}}_j(\theta)) = \frac{\pi_\theta(o^{\text{fb}}_j \mid q_{\text{base}})}{\pi_\theta(o^{\text{fb}}_j \mid q_{\text{base}}) + \gamma}, \qquad 0 < \gamma < 1 \tag{9}$$

把 $p = \pi_\theta(o^{\text{fb}}_j \mid q_{\text{base}})$ 看成横轴，$f(p) = p/(p+\gamma)$ 的图像是这样：

![自绘图：policy shaping 曲线 f(p) = p/(p+γ)。γ 越小，对低概率样本的放大越强；高概率端被压到 1/(1+γ) 的上界。论文取 γ=0.1。](/home/hatch/workspace/your_files/paper-xray-100/papers/180-VLA-elfvla/fig-policy-shaping.png)

*这张图是按式 (9) 右端画的。注意它已经不是 importance sampling ratio（分母里没有 $\pi_{\text{old}}$），而是一个只依赖当前策略概率的有界单调变换。*

算两个数就明白它的作用。取论文的 $\gamma = 0.1$：

- 一个稀有但正确的修正 token，当前策略下概率 $p = 0.02$：shaped 权重 $= 0.02/0.12 \approx 0.167$，是原概率的 **8.3 倍**。梯度终于能"看见"它。
- 一个已经很熟练的 token，$p = 0.9$：shaped 权重 $= 0.9/1.0 = 0.9$，几乎没变，但上界是 $1/1.1 \approx 0.909$，再高的概率也冲不上去。

所以 shaping 做的是两件事：低概率区近似线性放大 $1/\gamma$ 倍（$\gamma=0.1$ 就是 10 倍），高概率区压到上界。Table 6（第 8 页）显示去掉 shaping（$k=1$）PDMS 从 91.0 掉到 89.3——**1.7 个 PDMS 点**（原文写作"1.7%"，实为点数差），说明没有这个机制，借来的好样本在优化里几乎等于不存在。这是全文最承重的设计之一：反馈机制负责"造出好样本"，shaping 负责"让好样本的梯度活下来"。

符号表（核心公式涉及的量，一次列清）：

| 符号 | 形状/含义 |
|---|---|
| $q_{\text{base}}$ | base 输入：前视图像 + 导航指令 + 自车状态 + 3 帧历史轨迹 |
| $q_{\text{fb}}$ | feedback 输入：$\langle q_{\text{base}}, \text{旧作答}, \text{反馈} \rangle$ |
| $o_c$ / $o_w$ | PDMS $\ge s$ 的答对样本 / $< s$ 的答错样本 |
| $f_{\text{rule}}$ / $f_{\text{teacher}}$ | 规则生成的肯定反馈 / teacher 生成的结构化诊断 |
| $o_{\text{gt}}$ | 真值轨迹（teacher 可见，policy 训练时不可见） |
| $r_{\text{traj}}, r_{\text{fmt}}, r_{\text{goal}}$ | PDMS 奖励 [0,1]、格式奖励 {0, 0.5, 1}、终点阶梯奖励 |
| $n$ / $k$ / $s$ / $\gamma$ | 每 batch rollout 数（8）/ 回灌修正样本数（1）/ 正误阈值（0.8）/ shaping 参数（0.1） |
| $r_{\text{union}}$, $A_i$ | 合并奖励集、相对 advantage |
| $\beta$, $D_{\text{KL}}$ | KL 惩罚系数、相对初始 SFT 模型的 KL 散度 |

## 数字核实：每个数出自哪里

**主结果**（第 5–6 页，表 1/表 2）：

- NAVSIMv1 **PDMS 91.0**（纯视觉设定）。之前最好的纯视觉方法是 DriveVLA-W0-3B 的 90.3，领先 **0.7**；GoalFlow 也是 90.3 但用了 lidar。相对自家基线：SFT-only（InternVL3-8B-SFT）87.4 → **+3.6**；普通 GRPO（InternVL3-8B-RL）89.0 → **+2.0**。子指标：NC 98.9、DAC 98.1、TTC 96.0、CF 100、EP 85.3。
- NAVSIMv2 **EPDMS 87.1**，之前最好 DriveVLA-W0-3B 86.1 → **+1.0**。子指标里 EC（扩展舒适）87.2，接近最高的 DiffusionDrive 87.7。

**反馈机制对照**（表 3，第 6 页）：GRPO 89.0 / GT-GRPO 89.2 / Rule-GRPO 89.6 / ELF-VLA 91.0。直接塞真值只 +0.2（分布偏移，真值在策略下概率太低），规则反馈 +0.6，teacher 结构化反馈 +2.0。

**高层规划准确率**（表 4，第 7 页）：ELF-VLA 80.3%（纵向速度 85.8 / 横向路径 92.5），GRPO 79.3%（+1.0）。注意这个指标的 GT 标签是从真值轨迹派生的（附录：纵向按滑动窗口加速度分四类 accelerate/decelerate/maintain/stop，横向按地图拓扑分五类），是个开环分类指标。

**数据 curation**（表 5，第 7 页）：85k 全量 89.1 / 24k 随机 88.9 / 24k 精选 91.0。

**k 与 shaping 消融**（表 6，第 8 页）：$k=4$ → 89.0（**掉回纯 GRPO 水平**，反馈样本一多增益被噪声吃光）；$k=2$ → 89.7；$k=1$ 无 shaping → 89.3；$k=1$ + shaping → 91.0。只试了 1/2/4 三个点，$k=1$ 是尖峰还是平坦区域未知。

**训练管线消融**（表 8，附录）：NAVSIM-only SFT 85.3 → +大规模驾驶 QA 预训练 87.4（+2.1）→ +Feedback-GRPO 91.0（+3.6）。**无预训练对照**（表 9）：ELF-VLA 90.0 vs 基线 86.9（+3.1），说明增益主要来自 GRPO 设计而非预训练数据。

**阈值 $s$**（表 10，附录）：0 → 89.4 / 0.5 → 90.1 / 0.9 → 89.8 / **0.8 → 91.0**。$s=0.9$ 反而掉，作者的解释是 [0.8, 0.9) 的样本已经够好，强行纠正引入噪声。只试了四个点。

**训练代价**（附录 8.1，表 7）：三阶段。预训练：16×H20，2 天（lr 1e-5，batch 1，2 epochs）；SFT：16×H20，8 小时（lr 4e-5，batch 2，2 epochs）；RL：32×H20，18 小时（lr 2e-6，RL batch 正文写 3、表 7 写 2——**两处不一致**，temperature 1.2，每 query 8 条 rollout，2 次迭代）。推理时延 0.1s（vLLM 加速，CoT+轨迹一起生成；未给 batch size 和精度条件）。

## 怀疑式阅读：哪些结论可以信，哪些还缺信息

**teacher 看见了真值，这是全文最大的诚实性考验。** Algorithm 1 第 3 行和 3.1 节都明确：teacher 的输入包含 $o_{\text{gt}}$。它的 "Actionable Correction"（"每步横向目标 0.3 米、纵向速度上限 2 m/s"）在多大程度上是诊断推理的产物，在多大程度上是把真值翻译成了自然语言？论文没有做这个区分。GT-GRPO（89.2）说明"直接给真值"不好优化，但它没有回答"teacher 的诊断里有多少信息量来自真值"。一个更干净的消融应该是：teacher 不看 $o_{\text{gt}}$ 只看 $o_w$ 和 metric 分数，PDMS 还剩多少。这个实验没做，所以 +2.0 里"诊断"和"特权蒸馏"各占多少，目前是分不开的。诚实地说，ELF-VLA 的机制更接近"一个能看见真值的强规划器在手把手带学生"，而不是"学生自己学会了从失败中诊断"。

**没有随机种子和方差报告。** 全文所有表格都是单次运行的点估计。RL 训练的 run-to-run 方差在 NAVSIM 这种基准上经常有 0.5–1 个 PDMS 点的量级，0.7（vs DriveVLA-W0）和 1.0（EPDMS）的领先是否稳健，现有信息判断不了。

**参数量不对等。** SOTA 对比里 DriveVLA-W0 是 3B 模型，ELF-VLA 是 8B（InternVL3-8B：300M 视觉编码器 + 7B Qwen2.5）。2.7 倍的参数差本身就能解释一部分差距。更干净的对比其实是表 2 里的 Recogdrive-8B（83.6 EPDMS，同 8B 级别，ELF 领先 3.5）——这个差距比 headline 的 +1.0 更有说服力，论文没有强调它。

**几个文本与表格的不一致**：(1) RL batch size 正文 8.1 节写 3，表 7 写 2；(2) Algorithm 1 第 3 行为全部 $n$ 条 rollout 调用 teacher，但式 (1) 和 3.1 节说 teacher 只处理答错样本——如果伪代码是忠实的，teacher 调用量比正文描述大得多（不过 $s=0.8$ 的阈值下答错占多数，差别可能不大；论文没给答错比例）；(3) "去掉 shaping 掉 1.7%"实为 1.7 个 PDMS 点。

**$k=1$ 的选择很脆。** $k=4$ 时 PDMS 掉回 89.0，恰好等于纯 GRPO——说明反馈样本的信噪比很低，多加几条就把增益全吃掉了。只试了 1/2/4，$k=1$ 是尖峰还是平台不知道。实际用起来，这个超参数大概率要跟着 teacher 质量和场景难度重调。

**基线选择有讲究也有水分。** 表 4 里拿 Qwen2.5-VL-72B（28.7%）这种没做过驾驶微调的通用 VLM 当基线，51.6 个点的差距没有任何信息量——不做驾驶微调的模型在这个指标上本来就该是这个水平。SFT（79.2）到 ELF-VLA（80.3）的 +1.1 才是真实的改进幅度，而且不大。

**评估范围就是作者自己承认的局限。** 全部实验在 NAVSIM 上——非反应式仿真（log-replay，其他交通参与者不会对自车行为做出反应）。结论部分原话承认两点：性能被 teacher 的分析能力上界 bound 住；没做闭环反应式评估。NAVSIM 的 PDMS 本来就是开环指标为主，这篇的"闭环"是打了引号的。

**做得漂亮的地方也值得点名。** 一是 GT-GRPO / Rule-GRPO 两个对照基线的设计——它们把"真值泄露"和"规则反馈"这两个最容易被质疑的点主动做成了实验，而不是藏在文字里。二是 Figure 4 直接测量训练中的 total-failure 比例——这个指标和中心论点（persistent failures）是同一件事，而不是拿最终分数倒推。三是数据 curation 的消融（随机 24k vs 精选 24k）把"选难样本"和"反馈机制"的贡献分开了，虽然两者是乘法关系，但至少证明了 curation 不是在搭便车。

## 和 DriveVLA-M0 的对照：失败学习发生在训练时还是测试时

（说明：DriveVLA-M0 的测试时设定来自任务背景，本文未引用 M0；本文引用的同系列基线是 DriveVLA-W0，即世界模型数据 scaling 那条线，在表 1/表 2 里是 PDMS 90.3 / EPDMS 86.1 的直接对手。）

两条路线处理"长尾失败"的时间位置完全相反。DriveVLA-M0 的路线是在**测试时**做自适应：模型上车之后遇到没见过的场景，用 LoRA 做 test-time training，现场学、现场改，推理不是 frozen 的。ELF-VLA 恰好反过来：**失败学习全部发生在训练时**——teacher 诊断、反馈回灌、shaping 加权，每一步都在训练循环里；一旦训练结束，teacher 就消失了，推理时模型是 frozen 的，只输出 think + 轨迹，不再有任何自我修正。

这个对照点出了两种"从失败中学习"范式的代价结构。M0 把代价付在推理时：每次遇到难场景都要现场梯度更新，有延迟和稳定性风险，但不需要训练时准备 teacher，也不受 teacher 能力上界限制。ELF-VLA 把代价付在训练时：32 块 H20 跑 18 小时 RL，每一步 rollout 还要调一个 32B 的 teacher 写诊断报告，训练开销巨大，但推理时就是一次 0.1s 的前向，无额外负担。两种路线对"失败"的定义也不同：M0 的失败是测试时遇到的分布外场景，ELF-VLA 的失败是训练集里模型稳定搞不定的长尾样本（靠 curation 筛出来的 24k）。一个向外（部署时），一个向内（训练时）。

## 与 RAMP 的血缘/划界

ELF-VLA 落在"VLA + RL/蒸馏"这条线上，和 ReCogDrive-DiffGRPO、Reflect-VLA、TakeVLA（Learning from Mistakes，同期 arXiv:2603.14972）并列，都是"失败学习"的数据点：ReCogDrive 用 RL 直接调扩散规划器，Reflect-VLA 做反思引导的自蒸馏，ELF-VLA 用外部 teacher 的结构化诊断合成纠正样本。三者的共同假设是：**失败样本里有信息，关键是怎么把信息提出来喂给优化器**。

和 RAMP 的关系是"同源、镜像、互补"三层：

- **同源**：ELF-VLA 的"失败 → 诊断 → 纠正"链条，和 RAMP 的情景记忆（从高风险经历里保留经验）是同一个思想——失败不是噪声，是最值得保留的信号。ELF-VLA 用 teacher 把失败写成自然语言报告，RAMP 用风险等级调制记忆的衰减速率（高风险忘得慢），都是给失败样本开绿灯。
- **镜像**：ELF-VLA 的诊断-纠正发生在**训练时**，推理时 frozen；RAMP 的闭环风险校验发生在**推理时**——候选轨迹独立 rollout、风险仲裁、多规划器调度，每一步都在车上实时发生。ELF-VLA 是"把纠正能力蒸馏进权重"，RAMP 是"把纠正能力留在运行时架构里"。
- **互补/划界**：ELF-VLA 依赖外部强 teacher（Qwen3-VL-32B），性能被 teacher 上界 bound 住，且 teacher 在推理时不存在；RAMP 不依赖外部 teacher，风险校验来自世界模型独立 rollout 的自包含验证。ELF-VLA 的记忆是训练数据的静态集合（curation 出来的 24k），RAMP 的情景记忆是在线的、测试时自适应的（非参数）。ELF-VLA 是单一 VLA 规划器 + RL 微调，RAMP 是 LLM 调度异构规划器池——架构上就没有重叠。

**一句话 related work 定位**：ELF-VLA 在训练时用能看见真值的外部 teacher 对失败做结构化诊断、合成纠正样本增强 GRPO，而 RAMP 把诊断与纠正搬到推理时闭环里，用世界模型独立 rollout 做风险仲裁、不依赖外部 teacher——前者是后者的训练时镜像，可在 RAMP 的 related work 里作为"失败学习"谱系的训练时代表与之对照。

## 带走什么

这篇论文留下的可迁移想法有三个。第一，**"给优化器造正样本"比"调优化器"更治本**：当探索退化到全零分区域时，再好的 advantage 估计、再精的 clip 范围都救不了梯度，因为 batch 里根本没有方向信息。ELF-VLA 的核心动作不是改进 GRPO 的数学，而是改变 batch 的内容构成——这个思路在任何稀疏奖励的 RL 场景里都成立。

第二，**policy shaping 是一个通用技巧**：凡是"在条件 A 下采样、拿到条件 B 下优化"的 off-policy 样本（反馈修正、人工示范、teacher 输出），标准 importance ratio 都会因为条件失配而方差爆炸；换成 $p/(p+\gamma)$ 这种有界变换，用 $1/\gamma$ 的放大换稳定性，是比直接 clip 更对症的处理。它和 GRPO 的 group 归一化是正交的，可以叠加。

第三，**诊断的粒度决定了纠正的质量**：Rule-GRPO（+0.6）和 ELF-VLA（+2.0）的差距，来自反馈是"二值对错"还是"定位到 think 推理哪一步错了"。这对 RAMP 有直接启发——运行时风险仲裁如果只输出"这条轨迹有风险"，调度器的改进空间有限；如果能定位到"是哪个规划器的哪类假设错了"，反馈才有指导性。

真正值得做的开放问题：teacher 能不能被蒸馏掉？ELF-VLA 的推理时模型已经内化了一部分诊断能力（SFT 阶段学过读反馈改轨迹），但诊断的**生产者**始终是外部 32B teacher。如果让模型自己给自己写诊断（self-critique），PDMS 还剩多少——这个实验没做，而它决定了这条路线是"方法"还是"对 teacher 的依赖"。作者在结论里把"探索不同 teacher 模型的作用"列为未来工作，算是诚实地把这个问题留了出来。
