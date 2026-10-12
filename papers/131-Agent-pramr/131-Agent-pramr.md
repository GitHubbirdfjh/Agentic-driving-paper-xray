# LLM 当传感器开关：PRAM-R 的模态路由与四层记忆

> **论文**：PRAM-R: A Perception-Reasoning-Action-Memory Framework with LLM-Guided Modality Routing for Adaptive Autonomous Driving
> **作者**：Yi Zhang, Xian Zhang, Saisi Zhao, Yinglei Song, Chengdong Wu, Nenad Petrovic, Alois Knoll —— 慕尼黑工大（TUM）机器人、人工智能与嵌入式系统教席
> **Venue**：arXiv:2603.04222v1（2026-03-04）
> **核心洞见（一句话）**：这篇论文的 LLM 不碰驾驶决策，它只做一道选择题——"现在开哪几个传感器"；所谓的四层记忆，本质是一个带过期时间的路由决策缓存，记住的是"上次这种场景开了哪个传感器"，而不是驾驶经验。

**先说最重要的判断**：任务简报里说"四模块与 RAMP 高度同构"，读完正文后我的结论是——**同构只存在于模块名字层面**。Perception–Reasoning–Action–Memory 这四个词和 RAMP 的"调度–世界模型–记忆–校验"在字面上能对上，但 PRAM-R 的每个模块干的事和 RAMP 的完全不是一回事。划界一节会把这四条差异钉死。先讲论文本身。

---

## 问题的形状：全模态常开是在烧钱

起点是一个工程事实：摄像头、激光雷达、毫米波雷达全开，每帧都做融合，计算和延迟都贵；而且在某些场景下，多开一个模态不仅没帮助反而添乱（比如大雾天激光雷达全是噪声，融进去拉低精度）。已有的自适应方法（AdaFusion、DynMM、SamFusion）用注意力或不确定性做门控，但都是"小模型+启发式"，没有高层推理。

PRAM-R 的回答：用 LLM（Qwen3-VL-8B）当**模态路由器**。注意这个定位——LLM 不输出轨迹、不输出动作，它输出的是"每个传感器的可靠性分数 + 场景复杂度 + 开关掩码"。驾驶决策本身由 reasoning/action 层负责，而这两层是**沿用**作者自家前作 PLA 框架 [8]（arXiv:2507.23540，同一作者群）的设计。所以这篇论文的真实增量是两块：LLM 路由器 + 分层记忆。标题里的四个模块，一半是继承的。

## 双环架构：快环执行、慢环思考

公式 (1)(2) 定义了异步双环：

$$O_t = f_{react}(S_t, M^{map}_t, E_t, P_t, M_\tau), \quad \tau \le t$$

$$M_t = f_{delib}(H_{t-\Delta t}, P_{t-\Delta t}, M_{t-\Delta t})$$

快环（reactive）每步都跑：拿当前传感器数据 $S_t$、地图、外部信息，**再加上慢环上次留下的记忆快照 $M_\tau$**（$\tau \le t$，可能是几帧之前的），输出驾驶指令 $O_t$。慢环（deliberative）攒一个时间窗 $H_{t-\Delta t}$ 的历史，更新记忆 $M_t$，里面包含 LLM 的路由决策。

关键细节在 $M_\tau$：快环用的记忆是**过期的**（stale by design）。论文没有讨论过期记忆的可用性边界——慢环 1Hz、快环 2Hz（ablation 里提到的配置），记忆最多过期 0.5 秒，问题不大。但如果慢环更慢（LLM 推理本来就慢），这个设计的天花板很明显。RAMP 的异步流水线+时间戳补偿正是要系统化地解决这个问题，PRAM-R 只是"先这么用着"。

## Shallow perception：诊断指标的手工清单

路由器的输入不是原始传感器数据，而是"shallow perception"提取的诊断指标（Table I）。用的：摄像头亮度/对比度/边缘密度、激光雷达点云密度/噪声比/反射率、雷达截面/检测概率、天气/光照/地图复杂度。没用的（标 ✗）：激光雷达回波比、Z 方差、雷达 SNR 等，"omitted for efficiency"。

这张表的诚实之处在于把"没用哪些"也列出来了。但选取标准是手工的：为什么留边缘密度不留回波比？没有消融。更深的问题是：**这些诊断指标本身就是启发式的**——"低亮度→低语义分割精度"这种映射是经验规则。LLM 路由器的输入如果是启发式指标，那 LLM 的"推理"只是在启发式之上再包一层语言模型，垃圾进、包装过的垃圾出。论文没有评估诊断指标本身的准确性。

## LLM 路由器：公式 (3)-(7) 一步步走

路由器每步输出两样东西（公式 3）：

$$\{u_i^t\}_{i=1}^N, \{r_i^t\}_{i=1}^N = f_{LLM}(I_t, C_t, M_{t-1})$$

$u_i^t \in \{0,1\}$ 是"场景复杂度建议开/关"（usage mask），$r_i^t \in [0,1]$ 是"可靠性分数"。输入是诊断指标 $I_t$、场景上下文 $C_t$、上一步记忆 $M_{t-1}$。LLM 在这里干的事：**把一堆数字映射成另一堆数字，外加一个 JSON 格式**。prompt 要求输出严格合法的 JSON，包含每个模态的可靠性分数、场景复杂度标量、二值开关。这本质上是一个结构化回归任务——用 8B 的 VLM 来做，相当于请米其林大厨来煮方便面。能煮，但贵，而且大厨的"厨艺"（常识推理）在这里完全没用上。

然后是选择逻辑（公式 4-6）：

$$R_t = \{i \mid r_i^t \ge \theta_i\}, \quad w_i^t = \frac{r_i^t}{\sum_{j \in R_t} r_j^t}$$

$$U_t = \{i \mid u_i^t = 1\}, \quad F_t = U_t \cap R_t, \quad \text{若} |F_t|=0 \text{ 则} F_t \leftarrow R_t$$

可靠集合 $R_t$（分数过阈值）和复杂度建议集合 $U_t$ 取交集，交集为空时回退到 $R_t$。权重是可靠性分数的归一化。这个"交集+回退"规则是纯启发式的：它隐含了"复杂度建议"和"可靠性"同等重要，但没有论证为什么是交集而不是加权。Fallback 的方向（听可靠性的）是合理的——安全兜底嘛，但论文没明说。

稳定化两件套。Hysteresis（施密特触发器）：

$$s_i^t = \begin{cases} 1 & s_i^{t-1}=0 \land r_i^t \ge \theta_i + \delta \\ 0 & s_i^{t-1}=1 \land r_i^t \le \theta_i - \delta \\ s_i^{t-1} & \text{otherwise} \end{cases}$$

开和关用不同的阈值（差 $2\delta$），防止可靠性分数在阈值附近抖动时开关来回跳。这是控制理论里用了几十年的东西，论文拿来用是对的。EMA 平滑权重：

$$\tilde{w}_i^t = \alpha w_i^t + (1-\alpha)\tilde{w}_i^{t-1}, \quad \alpha = 1 - e^{-\Delta t/\tau}$$

$\alpha$ 用时间常数 $\tau$ 参数化，这是连续时间一阶低通滤波器的精确离散化——工程上是标准做法，论文写对了。

Algorithm 1 把全流程串起来：LLM 打分 → 交集选择 → hysteresis 锁存 → EMA 平滑 → 更新记忆。注意第 16 行：记忆更新 `UPDATE MEMORY(M_{t-1}, {u_i^t}, {r_i^t}, {s_i^t}, {\tilde{w}_i^t})`——**记忆存的是路由决策本身**（开关、分数、权重），不是驾驶经验。这个细节是划界的关键证据。

## 四层记忆：名字唬人，实质是缓存

Fig.2 和 Sec III-D 描述了四层：

- **P-Routing Context**（短期）：存诊断指标 + 路由决策，感知更新时同步。
- **P-Semantic Cache**（短期）：压缩后的感知输出（检测、分割、地图块）。
- **R-Scene Record / R-Seed State**（中期）：场景理解、风险评估、抽象策略提示。
- **A-Policy Log**（中期）：控制指令、轨迹误差、约束激活。
- **Knowledge Repository**（长期）：每 episode 蒸馏的"高级场景描述符和最优策略"，号称支持"continual self-improvement"。

Table II 给了每个组件的生成/读取/更新/过期触发条件。读下来你会发现：**这是一个带过期时间的缓存系统**，不是记忆系统。P-Routing Context 在"下一轮路由决策后"过期，R-Seed State 在"每轮推理周期后"过期——过期条件就是"被用过一次"。真正的记忆应该有"什么值得长期保留"的选择机制（比如 RAMP 的风险加权），这里没有：Knowledge Repository 的"distill with quality threshold"提了一句，但阈值是什么、怎么蒸馏，正文没展开，实验也没测。

作者在 conclusion 里自己承认了："Memory Validation: due to space constraints, this work focuses on the core aspects of the memory mechanism; future work will include layer-wise memory ablations"。翻译：四层记忆里真正被实验验证的只有路由缓存那部分，R-Seed State、A-Policy Log、Knowledge Repository 都没有独立消融，"持续自改进"是个期货。

## 实验：数字逐个验算

**Table III（synthetic stress test）**：高频扰动下，纯阈值 vs hysteresis 的切换次数：Camera 11→2（-81.8%），LiDAR 18→2（-88.9%），Radar 18→2（-88.9%），总计 47→6（-87.2%）。验算都对。但 baseline 是"纯阈值"——在高频噪声下纯阈值本来就会疯狂抖动，这是一个**故意选弱的 baseline**。任何带迟滞的控制器都能赢 80% 以上。这个 87.2% 是工程常识的胜利，不是方法的胜利。

**Table IV（ablation，10 个 nuScenes 场景）**：

| Variant | RE | RC | RSI | MRR | ADE | FDE |
|---|---|---|---|---|---|---|
| Static Fusion (PLA) | 0.00 | 1.000 | 1.000 | 0.000 | 1.013 | 2.026 |
| w/o Memory | 6.02 | 0.992 | 0.662 | 0.000 | N/A | N/A |
| w/o Dual-loop | 5.67 | 0.998 | 0.641 | 0.313 | N/A | N/A |
| Full PRAM-R | 6.22 | 0.991 | 0.699 | 0.2000 | 1.124 | 2.182 |

三处值得停下来看：

第一，**ADE/FDE 变差了**：1.013→1.124（+10.9%），2.026→2.182（+7.7%）。论文正文说 "Perception metrics (ADE/FDE) remain comparable"——1.124 vs 1.013 叫 comparable 是在硬拗。关掉 6.22% 的传感器，轨迹误差涨 11%，这个 trade-off 论文没有正面讨论。

第二，**w/o Memory 和 w/o Dual-loop 的 ADE/FDE 是 N/A**——没测。于是我们永远不知道"记忆"对轨迹精度是帮助还是拖累（full 的 ADE 比 baseline 差，但 baseline 是 PLA 不是 w/o Memory，变量没隔离）。

第三，**记忆对 RE 的贡献只有 0.2 个百分点**：w/o Memory 6.02% vs full 6.22%。路由效率的增益几乎全来自"LLM 路由器+双环"，记忆的边际贡献微乎其微。而 MRR（记忆召回率）full 只有 20%，w/o Dual-loop 反而有 31.3%——论文解释为"stable activation avoids redundant recalls"，这个解释是循环的：召回率低被说成是"稳定所以不需要召回"，那要记忆何用？

**Fig.4（记忆召回率曲线）**：短期（5 秒滚动）召回随场景切换波动，长期累积收敛。曲线好看，但 20% 的绝对值不高，且没有和"不用记忆"的下游指标对比。

**Fig.5（nuScenes 全局统计）**：相机和雷达基本常开，激光雷达在低增益/噪声时自适应关闭；中等复杂度只开相机+雷达，高复杂度全开。这是全篇最有信息量的实证：它说明**路由器的行为是可解释的、符合直觉的**。但这也是"意料之中"的——任何合理的可靠性阈值都会这样。

## 雨夜路口 walkthrough

场景：夜间、雨、城市路口。Shallow perception 输出：摄像头亮度低、对比度低（雨+夜），激光雷达点云密度低、噪声比高（雨滴反射），雷达检测概率正常（雨对毫米波影响小）。

LLM 路由器收到这些指标，输出：相机可靠性 0.35、激光雷达 0.28、雷达 0.82；场景复杂度 0.8（路口+夜+雨）；usage mask 全 1（复杂场景建议全开）。阈值假设 $\theta=0.5$：$R_t$={雷达}，$U_t$={相机，激光，雷达}，交集 $F_t$={雷达}。权重归一化后雷达 1.0。Hysteresis：如果上一帧相机是开的（$s^{t-1}=1$），关闭阈值是 $\theta-\delta$=0.4，0.35<0.4，相机关闭；激光雷达同理。EMA 把权重平滑，避免下一帧雨小一点就反复横跳。

Deep perception 只用雷达做检测（+地图），输出 JSON 给 reasoning 层。Reasoning 层（沿用 PLA）做场景理解，Action 层输出轨迹。记忆更新：P-Routing Context 存下"雨夜路口→只开雷达"这条决策。

下一帧雨停了，相机可靠性回升到 0.6。Hysteresis 的开启阈值是 $\theta+\delta$=0.6，0.6≥0.6，相机重新打开——但注意是**恰好压线**，如果 $\delta$ 设得大一点，0.6 开不了机。$\delta$ 的选择论文没给（记为缺口），而它直接决定了"雨后多久恢复全模态"。

这个 walkthrough 暴露了系统的真实能力边界：**它是一个做得不错的传感器电源管理器**。驾驶决策（怎么过路口）完全由 PLA 的 reasoning/action 层负责，LLM 路由器不参与。把这篇论文放到 RAMP 的 related work 里，它属于"效率优化"一支，不属于"决策智能"一支。

## 怀疑式阅读

1. **LLM 做结构化回归是大炮打蚊子**。路由器的输入是十几个诊断数字，输出是 JSON 数字。用 Qwen3-VL-8B 干这个，延迟和显存开销都大，论文 future work 自己承认"introducing latency and memory overhead on embedded hardware"。一个两层 MLP 就能拟合"诊断指标→开关"的映射——论文没有和"MLP 路由器"对比，无法证明 LLM 的常识在这里有任何用处。我的判断：**没有**。prompt 里没有任何需要常识推理的内容，全是数值映射。

2. **记忆对下游精度的贡献为负**（ADE +10.9%，FDE +7.7%），且关键消融的精度指标是 N/A。"Comparable" 的说法站不住。

3. **Table III 的 baseline 是故意选弱的**。纯阈值在高频噪声下抖动是已知事实，87.2% 的"提升"是迟滞控制器的应得分数，不是 LLM 的功劳。论文把 hysteresis 和 LLM 路由器打包宣传，有混淆之嫌——Table III 测的只是 hysteresis，没有 LLM 什么事。

4. **10 个场景，选取标准不明**。"10 representative scenes"——怎么代表的？nuScenes 有 1,000 个场景，选 10 个的偏差风险很大。Fig.5 的全局统计如果是在这 10 个上做的，代表性更可疑。

5. **一半模块是继承的**。Reasoning/Action 沿用 PLA [8]（自家前作），perception 的 deep 部分也是标准检测分割。真实增量 = LLM 路由器 + 路由缓存。标题的四模块框架有包装成分。

6. **"持续自改进"是期货**。Knowledge Repository 没有实验，作者自己列为 future work。但摘要和结论的措辞（"supports long-term adaptation""continual self-improvement"）会让读者以为这是已验证的。

7. **诊断指标手工选取无消融**。Table I 里 ✗ 掉的指标（回波比、SNR 等）为什么不要？没有对比实验。

8. **交集规则和 fallback 是启发式**。$F_t = U_t \cap R_t$，空则回退 $R_t"——为什么是交集不是加权？$\theta_i$、$\delta$、$\tau$ 的取值论文都没给，复现不了。

## 作者的底牌

一张半。半张是**问题的工程真实性**："全模态常开浪费"是车厂真实面临的问题（功耗、散热、传感器寿命），审稿人里有工业界背景的会认可这个问题的重要性。一张是**TUM 的招牌 + 写作的完整性**：双环、记忆、hysteresis、EMA、synthetic+real 两阶段评估，故事是完整的，公式是自洽的。但底牌不硬：核心实验（10 个场景、ADE 变差、记忆无消融）撑不起"unified framework"的宣称。这篇更像一篇 workshop 级别的工作投成了 arXiv 全文。

## 与 RAMP 的血缘/划界（划界从严）

**先把"同构"的神话拆掉**。PRAM-R 的四模块和 RAMP 的四模块，名字能对上，干的事对不上：

| | PRAM-R | RAMP |
|---|---|---|
| LLM 的职责 | 传感器开关（开/关相机、激光、雷达） | 多规划器调度（选哪个规划器、给什么参数） |
| 记忆的内容 | 路由决策缓存（诊断指标→开关记录） | 情景经验（场景→动作→风险），风险加权+自适应衰减 |
| 记忆的机制 | 存→读→过期（Table II），无选择、无加权 | 风险加权检索、自适应衰减、分层索引 |
| 世界模型 | 无 | 有（单次前向复用） |
| 校验 | 无（最多算 hysteresis/EMA 信号平滑） | 闭环风险校验 |
| 优化目标 | 效率（省算力、省电） | 安全（降风险）+ 效率（降延迟） |

**四条划界，related work 里建议这样写**：

1. **LLM 的层级不同**。PRAM-R 的 LLM 是**感知层的开关**，输出不进入驾驶决策；RAMP 的 LLM 是**决策层的调度器**，输出直接决定车辆行为。一个管"眼睛睁不睁"，一个管"手往哪打"，不在同一层，不构成竞争。

2. **记忆的语义不同**。PRAM-R 的记忆回答"上次这种天气开了哪个传感器"（Algorithm 1 第 16 行存的就是 $\{u_i^t\},\{r_i^t\},\{s_i^t\}$）；RAMP 的记忆回答"上次这种场景下哪个动作危险"。前者是**配置缓存**，后者是**经验记忆**。PRAM-R 的记忆没有风险概念、没有衰减机制、没有检索泛化——它的"过期"就是"用完即焚"（Table II）。RAMP 的风险加权自适应衰减是 PRAM-R 完全没有的维度。

3. **PRAM-R 的记忆对驾驶本身没有帮助**。Table IV 显示 full PRAM-R 的 ADE 比静态基线差 10.9%，记忆对 RE 的边际贡献只有 0.2pp。如果审稿人问"PRAM-R 也有记忆，你们的区别是什么"，最狠的回答是数据：**PRAM-R 的记忆是成本中心（省算力），RAMP 的记忆是安全资产（降风险）**——目标函数都不一样。

4. **PRAM-R 没有闭环、没有校验、没有世界模型**。它的"稳定化"是控制理论的 hysteresis/EMA，不是风险校验；它的"双环"是快慢频率不同，不是感知-预测-验证的闭环。在 ICAPS 的 framing 下，PRAM-R 属于"资源自适应"一支，RAMP 属于"决策-记忆-校验"一支，两者在 planning 会议的分类里都不会被分到一组。

**一句话**：PRAM-R 和 RAMP 的相似是**脚手架的相似**（双环、记忆、LLM），不是**问题的相似**。RAMP 引用它时，应该放在"相关架构模式"里一笔带过，重点引用的是它的反面教材价值：**记忆如果只是缓存路由决策，对驾驶安全没有贡献；记忆必须记住风险，才有资格叫记忆**。
