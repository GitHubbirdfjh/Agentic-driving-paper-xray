# NAVSIM：非反应式仿真的"中间道路"，以及它为什么测不出真本事

> 一句话：NAVSIM 在"开环位移误差"和"闭环仿真"之间取了个中间点——智能体只在第一帧规划一次 4 秒轨迹、背景车按录像重放，用 PDM Score 打分。它比 L2 更接近闭环，但因为背景车不理你、你也不用持续修正，它测不出动态反应能力。RAMP 选 Bench2Drive 而不选 NAVSIM，理由全在这篇里。

## 基本信息

| 项 | 内容 |
|---|---|
| 标题 | NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking |
| 作者 | Dauner Daniel（Tübingen）、Hallgarten Marcel（Tübingen/Bosch)、Li Tianyu、Weng Xinshuo、Huang Zhiyu（NVIDIA）、Yang Zetong、Li Hongyang（OpenDriveLab/上海 AI Lab）、Gilitschenski Igor、Ivanovic Boris、Pavone Marco（NVIDIA/Stanford）、Geiger Andreas、Chitta Kashyap（Tübingen） |
| arXiv | 2406.15349（v1，2026-10-11 已读全文；另有官方 repo 的 v2.0 文档已读） |
| 链接 | https://arxiv.org/html/2406.15349v1（已验证可访问） |
| 代码 | https://github.com/autonomousvision/navsim（已验证；v2.0 tag 的 docs/metrics.md 已读） |
| 数据 | OpenScene（nuPlan 的重分发，120 小时 @2Hz）；navtrain 103k / navtest 12k 场景 |
| 版本线 | v1.0（2024-04，CVPR 2024 竞赛）→ v1.1（2024-09）→ **v2.0（2025-02-28，加伪闭环）** → v2.1（2025-04，加两阶段 reactive traffic） |
| 信源评级 | **A 级（同行评议论文 + 官方 repo 文档，全文精读）**——PDMS 公式、权重、v2.0 伪闭环机制全部来自原文/repo |

## 方法论拆解（一）：为什么搞"非反应式"仿真

论文开篇列了现有评测的四个毛病，个个都是冲着当时社区的痛点：

1. **nuScenes 是为感知建的**，不是为规划建的——75% 直行，AD-MLP 盲开刷 L2 拿 SOTA。
2. **位移误差（ADE）经常 misrepresent**：两条轨迹，一条贴着人类轨迹但撞了，一条偏离人类轨迹但安全，ADE 会判前者更好（论文图 1 的经典例子）。
3. **闭环仿真贵且有 domain gap**：CARLA 图形渲染和真实传感器差得远；nuPlan/Waymax 是数据驱动但不支持图像/LiDAR 输入的方法。
4. **各家指标实现不一致**，没法公平比。

NAVSIM 的核心 trick：**绕开传感器仿真**。智能体只在**初始帧**用一次真实传感器数据，规划出 4 秒轨迹后**冻结**；之后在一个简化的 BEV 抽象上，用 LQR 控制器 + 运动学自行车模型 @10Hz 把轨迹展开，算 progress、TTC 这类仿真指标。背景车按**记录的未来轨迹重放**——不理你。

大白话：它把"开车"简化成"**开卷考试**"——第一帧看一眼真实世界，然后闭眼开 4 秒，考官（仿真器）按你交的卷子打分。背景车都是"NPC"，按剧本走，不会因为你的出现而刹车或让行。

## 方法论拆解（二）：PDM Score（PDMS）指标定义（逐字抠原文）

PDMS 继承自 PDM-Closed（nuPlan 上的规则规划器）打分函数，论文 Equation (1)：

$$\text{PDMS} = \underbrace{\left(\prod_{m \in \{\text{NC}, \text{DAC}\}} \text{score}_m\right)}_{\text{惩罚项}} \times \underbrace{\left(\frac{\sum_{w \in \{\text{EP}, \text{TTC}, \text{C}\}} \text{weight}_w \times \text{score}_w}{\sum_w \text{weight}_w}\right)}_{\text{加权平均}}$$

结构：**乘子惩罚 × 加权平均**，所有子分数 ∈[0,1]，PDMS ∈[0,1]，按帧算再平均。

**惩罚项（硬）**：
- **NC（no at-fault collision）**：与 road user（车辆/行人/自行车）发生 at-fault 碰撞 → score=0，**整场景 PDMS 直接归零**；与静态物碰撞 → 0.5（软惩罚）；非 at-fault 碰撞忽略（比如自车静止时被追尾不算你的锅）。
- **DAC（drivable area compliance）**：开出可行驶区域 → score=0，同样整场景归零。

**加权平均项**（权重**沿用 PDM-Closed 的 cost weights**，不是调出来的）：
- **EP（ego progress），weight=5**：自车沿路线中心的进度 / 一个"安全上界"。这个上界是 PDM-Closed（有特权的规则规划器）用搜索式轨迹提案算出来的"无碰撞无出界前提下能走多远"。比值 clip 到 [0,1]；**上界 <5 米时直接丢弃该样本**。注意：人类 EP 也只有 87.5——纯模仿人类拿不满，因为上界是规则规划器定的。
- **TTC（time to collision），weight=5**：默认 1；4 秒内**任一仿真步**，自车按恒定速度+航向外推的碰撞时间 < 阈值 → 置 0。
- **C（comfort），weight=2**：轨迹的加速度/jerk 和预设阈值比。

**权重 5/5/2 的来源**：论文原话 "Following the cost weights used by the PDM-Closed planner"——是抄规则规划器的代价权重，不是超参搜索来的。引用 PDMS 时要知道这个权重没有"最优性"背书。

## 方法论拆解（三）：场景筛选（navtrain/navtest 怎么来的）

多数人类驾驶数据都是"无聊"的（静止、匀速直行）。NAVSIM 用两道过滤：

1. **删太简单的**：constant velocity 基线（保持当前速度航向）PDMS > 0.8 的场景删掉——连"闭眼直开"都能拿 80 分，说明场景没信息量。
2. **删无解/噪声的**：人类司机自己的轨迹 PDMS < 0.8 的删掉——人类都开不好，要么是标注错了，要么是场景本身无解。

效果：constant velocity 基线从 79%（人类 91%）掉到 **22%**（人类 95%），横向多样性（log 尺度）显著提升。得到 navtrain 103k / navtest 12k（450GB）。

这个筛选法的方法论价值：**用"简单基线做得太好"来定义'无聊'**，比人工定规则干净。RAMP 将来筛选 Bench2Drive 难例时可以借鉴这个思路。

## 方法论拆解（四）：它和闭环到底有多对齐（论文 4.1）

作者在 nuPlan 仿真器里（有完全反应式的真闭环）做了对照：114 个学习型规划器 + 37 个规则规划器，对比 PDMS vs OLS（位移误差类开环分）与 CLS（真闭环分）的 Spearman/Pearson 相关性。结论：**PDMS 比 OLS 更接近 CLS**，尤其 CLS 时长从 15s 降到 4s（和 PDMS 时域对齐）时相关性更高。

但同一节的 limitations 是诚实的：**high PDMS 不蕴含 high CLS**——因为 NAVSIM 不考虑反应性（reactiveness），也不考虑闭环里的误差累积。作者原话建议：开发规划算法时，**把 CARLA 这类图形闭环仿真器当互补 benchmark**。

## 方法论拆解（五）：v2.0 的"伪闭环"到底是什么（2025-02-28）

这是任务点名的重点，来自官方 repo v2.0 的 docs/metrics.md（不是论文 v1 的内容）：

**两阶段伪闭环聚合**：
1. **第一阶段**：初始场景，4 秒，算 EPDMS（和 v1 流程一样）。
2. **第二阶段**：测试集里预置了多个**从初始场景派生出的 follow-up 场景**——做法是：从初始状态出发，用**几个不同的 3 秒计划**预 rollout，得到一批"起点不一样"的后续场景（比如带横向偏移、不同速度）。在每个 follow-up 场景上再跑一遍 planner，4 秒 EPDMS；这一阶段的**背景车换成 reactive 的**（不再是纯重放）。
3. **加权聚合**：每个 follow-up 场景的权重，取决于它的起点**离 planner 在第一阶段实际开到的终点有多近**——越近权重越高。最后加权平均得到总分。

大白话：v1 是"看一眼、闭眼开 4 秒"；v2 是"看一眼开 4 秒，然后考官把你'瞬移'到几个'如果你当时稍微偏一点'的平行宇宙里，再各开 4 秒，按你第一段实际开到哪儿来决定哪个平行宇宙更算数"。它**近似**了闭环里的"状态会偏、要持续修正"，但 planner 在每个 4 秒窗口内依然是开环的——所以叫"伪"闭环，名副其实。

**EPDMS（扩展版 PDMS）** 的新东西：
- 新增权重项：**LK（lane keeping，weight=2）**——偏离车道中心线太久扣分，路口禁用（因为路口中心线标注和实际车道线对不上）；**EC（extended comfort，weight=2）**——对比**相邻两帧**的轨迹输出，看加速度/jerk 是否跳变（v1 的 C 只看单帧轨迹的光滑，EC 看帧间一致性）。
- 新增乘子项：**DDC（driving direction compliance，{0, 1/2, 1}）**——逆行；**TLC（traffic light compliance，{0,1}）**——闯红灯（v1 被诟病不管红绿灯，v2 补上）。
- **False-positive 过滤**：人类司机也违章的场景，planner 不罚（比如借道超车必须短暂压对向车道）。公式里写成 max(m(agent), 1−m(human))。
- 完整公式：EPDMS = Π_{m∈{NC,DAC,DDC,TLC}} max(m(agent), 1−m(human)) × (1/16)Σ_{s∈{TTC,EP,C,LK,EC}} w_s·s(agent)，分母 16 = 5+5+2+2+2。

## 已知坑：为什么 NAVSIM 分数不反映动态反应能力（RAMP 论证核心）

这一节直接服务报告"RAMP 选 Bench2Drive 而非 NAVSIM"的论证，每条都有原文依据：

1. **背景车不反应 → 交互能力测不出来**。v1 里背景车按录像重放：你别它，它不躲；它别你，你躲了它也不领情。"让行""博弈""协商"这类 RAMP 冲突仲裁要处理的核心场景，在 NAVSIM 里根本不存在。v2 第二阶段虽然加了 reactive 背景车，但那是在预置的 follow-up 场景里，不是 planner 真实开出来的连续交互。
2. **单次 4 秒开环 → 测不出误差累积和长时程行为**。真闭环里"这次偏 10cm、下次偏 20cm"的滚雪球，在 NAVSIM 里不存在——每次评估都是从真实状态重新开始。RAMP 的"闭环风险校验"恰恰是要在误差累积的链条上工作。
3. **EP 上界是特权规则规划器定的**。PDM-Closed 能看到真值（BEV 特权输入），它的"安全上界"人类都达不到（人类 EP 87.5）。用特权上界归一化，等于拿"开挂的"当满分线——传感器方法天然吃亏一截。
4. **at-fault 定义让车尾不重要**。被追尾不算你的锅 → 模型可以完全不看后视。真实驾驶里"被追尾"很多时候是自己急刹/变道不当造成的，这个归因在 NAVSIM 里被豁免了。
5. **权重 5/5/2 是抄的**，舒适度在 v1 里所有模型 ~100（模仿人类自然学到），区分度为零——v1 的 PDMS 实际是 EP+TTC+NC+DAC 四项在干活。
6. **v1 不管红绿灯/stop sign**（v2 用 TLC 补了，但 v1 的论文数字都是不管的）；燃油/通行效率不管。
7. **确定性是优点也是局限**：NAVSIM 评估是确定性的（TransFuser 三个种子 σ=±0.56，方差极小），可复现性好；但真实驾驶的随机性（其他司机的意图不确定性）被平均掉了——而这正是 RAMP 的世界模型+不确定性处理要面对的。

## 与方向的关系

NAVSIM 是规划评估方法论上的**参照系**：它用一套干净的实验回答了"开环位移误差为什么不行"（图 1 的 ADE 反例、4.1 的相关性分析），是任何做端到端规划的人都绕不开的 baseline 论证。CVPR 2024 竞赛（143 队、463 次提交）还带出了一个有意思的结论：**冠军是"采样+打分"（学 TransFuser 预测 PDMS 子分数再加权选轨迹），不是更大的端到端模型**——"proposal + scoring"这个老派路线的回潮，和 RAMP 的"多规划器生成候选 + 风险校验打分选择"是同一个家族的思想。

## 与 RAMP 的血缘/启发

1. **选型论证（报告第 6 章）**：RAMP 的核心——闭环风险校验、冲突仲裁、多规划器动态调度——**每一项都要求环境对自车动作有反应**。NAVSIM（v1 纯非反应、v2 伪闭环）测的是"单次 4 秒开环规划质量"，Bench2Drive 测的是"持续闭环驾驶能力"。RAMP 必须选 Bench2Drive 做主 benchmark，NAVSIM 可作**辅助**（比如快速验证规划器单步质量、消融实验），但不能当主证据。这条论证本篇已给全弹药。
2. **PDMS 子分数设计可借鉴**：NC/DAC 硬惩罚（乘子归零）+ EP/TTC/C 加权平均，这个"一票否决 × 加权打分"的结构，可以直接启发 RAMP **内部风险评分**的设计——碰撞/出界类事件做硬门槛，进度/舒适做软加权。权重 5/5/2 虽是抄的，但"安全项权重 >> 舒适项"的排序直觉是对的。
3. **v2 伪闭环的思路值得玩味**：用"预置 follow-up 场景 + 按终点距离加权"来近似闭环，本质上是**用开环评估去覆盖闭环的状态分布**。RAMP 的情景记忆检索有异曲同工之妙——都是"用历史/预置的状态分布来弥补单次评估的覆盖不足"。报告里可以点一句这个思想血缘。
4. **筛选难例的方法**：NAVSIM 用"constant velocity 基线 PDMS>0.8 则删"来定义无聊场景，RAMP 在 Bench2Drive 上做课程学习/难例挖掘时可直接套用（比如用 TCP-traj 当基线过滤器）。
5. **引用口径**：提 NAVSIM 分数必须注明版本（v1 PDMS vs v2 EPDMS 口径完全不同，v2.1 又加了 reactive traffic），不然数字对不上。RAMP 若用 NAVSIM 做辅助对比，建议锁定 v2.0+ EPDMS。
