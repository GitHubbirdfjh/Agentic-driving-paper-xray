# 156 | Tesla AI Day 2021（官方回放 + 交叉核实）

## 基本信息

- **发布方**：Tesla 官方活动（以招聘为目的的技术公开日）
- **发布日期**：2021-08-19（美国时间；北京时间为 2021-08-20 上午）
- **链接**：https://www.youtube.com/watch?v=j0z4FweCy4M（官方回放，链接已验证可访问；视频章节：46:54 开始，48:44 Tesla Vision，1:13:12 Planning and Control，1:24:35 Manual Labeling，1:28:11 Auto Labeling，1:35:15 Simulation，1:45:40 Dojo）
- **核实方式**：本拆解未逐帧观看 3 小时视频，技术主张经 4 家独立文字实录/报道交叉核实——CleanTechnica 观察笔记、Towards Data Science 两篇复盘（Part 1 感知规划 / Part 2 训练数据）、TeslaMotorsClub 网友详解、DriveTeslaCanada 简化版，外加 videohighlight 的 AI 文字实录。四家在关键技术点上口径一致。
- **信源评级**：**A-** —— 源头是 Tesla 官方一手披露（Andrej Karpathy、Ashok Elluswamy 亲讲），信息源头硬；但本拆解依赖转述层，凡 Tesla 自报的数字（如"1 万 clips 一周标注"）均按"官方自称"处理，不当独立事实引用

> 术语大白话：**HydraNets** = 一个"九头蛇"网络：共用一个主干看懂图像，然后分出几十个"头"，每个头只干一件事（认红绿灯、认锥桶、估计车速……），各头可单独微调；**向量空间（vector space）** = 把 8 个摄像头的视频融合成一张统一的 3D 俯视图式表达，之后所有的标注、预测都在这个空间里做，而不是在单张图片上做；**自动标注（auto labeling）** = 用算力换人力——让离线大模型自动给海量视频打标签；**MCTS（蒙特卡洛树搜索）** = AlphaGo 下棋用的那种"先靠神经网络猜个大概、再靠搜索精修"的规划方法。

## 技术主张拆解

**主张 1：HydraNets——共享主干 + 约 80 个任务头（红绿灯、锥桶、停止标志、车辆及速度等），各头共享特征但可独立运行与微调。**
依据：四家来源一致描述（TMC："around 80 other neural networks that Tesla calls 'tasks'"；DriveTesla："Each Hydranet can talk to each other and share features however they all run separately"；TDS 列为视觉系统三大改进之首）。
可信度：**高**。多源一致，且 HydraNets 是 Tesla 2019 年起就公开的技术路线，AI Day 2021 是系统性披露。

**主张 2：8 相机视频融合进统一的 3D 向量空间（BEV 式表达），并加入"feature queue / video module"（spatial RNN + 时序记忆）来处理遮挡、估计速度、记住几米前看到的路牌。**
依据：TDS（"feature queue…look up the memory record…to detect occluded cars"）、DriveTesla（video module 改善深度与速度估计）、TMC 详解一致。
可信度：**高**。这是 2021 年 Tesla 纯视觉路线的核心技术主张，后续 FSD 的 BEV+时序架构一脉相承。

**主张 3：自动标注流水线——人工多边形标注 → 向量空间标注工具（一处标注、多相机自动映射）→ GPS 复用（同一路口标一次，数千次经过自动复用）→ 离线大模型全自动标注（可"偷看未来帧"解决遮挡与运动学标注）；实战案例：为移除雷达，1 万个 clips 在一周内自动标注完成，人工则需数月。**
依据：CleanTechnica（"10,000 clips were collected and automatically labeled within a week…It would have taken humans several months"）、TDS Part 2（"looking ahead at future frames…generate perfect labels by cheating"）、TMC 详解一致。
可信度：**中高**。流水线四阶段的描述多源一致可信；"1 万 clips / 一周 / 人工需数月"是 Tesla 自报的战果数字，无第三方审计，引用时标注来源。

**主张 4：规划侧——目标是安全/舒适/效率三优；采用"粗搜索（物理模型找凸走廊）+ 连续优化"的混合方案（例：双车道变道做 2500 次仿真、1.5ms 量级）；学习型规划（神经网络+MCTS，类 MuZero 思路）当时仍在研发，明确"not yet in use"，未上线。**
依据：CleanTechnica（2500 simulations in 1.5ms 的变道案例）、TDS（"coarse search and continuous optimization…Learning-based planning…not yet in use in the current version"）、videohighlight 文字实录（search/planning/optimization + NN heuristics 章节）一致。
可信度：**中高**。这是理解 2021 年 Tesla 的关键：**规划仍是经典优化，神经网络规划只是路线图**——不要把 2021 年的 Tesla 误读成"世界模型/智能体编排"。

**主张 5：配套披露了神经仿真（simulation）与 Dojo 超算——仿真用于难采集/难标注/闭环场景（如停车场泊车），用向量空间的完美标注做渲染；Dojo 解决训练算力。**
依据：CleanTechnica（simulation 三节：1:35:15 起）、TMC 详解。
可信度：**中高**。存在性主张可信；具体能力（如 Cybertruck 停车场 demo）为发布会演示。

### 宣传口径 vs 技术实质

- **宣传口径**：活动本身是招聘会（Musk 开场即招人），"解决真实世界 AI"的宏大叙事；Tesla Bot（Optimus）占压轴。
- **技术实质**：**纯视觉+BEV+数据飞轮**工程路线的首次系统性公开。真正的干货是感知与数据 pipeline（HydraNets、向量空间、自动标注），规划侧反而保守（经典优化打底）。一句话：2021 年的 Tesla 是"感知革命、规划守旧"——这与 Wayve 同期"世界模型+策略联合训练"形成鲜明对照，也是报告里值得写的一组对比。

## 与"编排 + 世界模型"方向的关系

AI Day 2021 是"**端到端数据驱动**"路线的工业级宣言：不靠高精地图、不靠规则引擎，靠海量视频+自动标注+BEV 时序网络。它定义了后来 5 年行业的主战场。但它**没有**世界模型（显式的未来预测与想象），**没有**智能体编排（规划是单体优化器），语言更是完全缺席。可以说：Tesla 2021 解决了"看清世界"，Wayve 的 MILE/LINGO/GAIA 则在回答"理解世界、谈论世界、想象世界"——"编排+世界模型"方向正好站在两者的交叉点上。

## 与 RAMP 的血缘 / 启发

1. **思想呼应**：HydraNets 的"共享主干、多任务头"与 RAMP 的"共享世界模型先验、多规划器"在架构哲学上相通——都是"一份共享的世界理解，多种下游用法"。RAMP 可以在 related work 里写："正如 HydraNets 用共享表征支撑多感知任务，我们用共享的世界模型中间分布支撑多规划器"。
2. **反衬定位**：2021 年 Tesla 的规划是"粗搜索+连续优化"的单体优化器，连 NN+MCTS 都还在研发。RAMP 的"LLM 在线调度多规划器"恰好是"学习型规划"在 2026 年的学术形态——引言可以写："从 Tesla 2021 年路线图里的 learning-based planning，到今天 LLM 调度多规划器，在线学习型规划终于到了可落地的时点"。
3. **数据飞轮的启示**：Tesla 用自动标注+仿真解决长尾数据问题；RAMP 的"风险加权情景记忆"解决的是另一个长尾问题——**在线时如何快速想起高风险先例**。两者都是"用过去的经验覆盖长尾"，一个在训练侧、一个在推理侧，可在讨论章节形成呼应。
4. **诚实边界**：① 所有 Tesla 数字（1 万 clips/一周、2500 仿真/1.5ms）均为官方自报，引用标注来源；② 本拆解基于转述交叉核实、非逐帧看片，报告中如需精确引用原话（如 Karpathy 的 "Operation Vacation" 等），建议再核对视频时间戳；③ 不要把 2021 年的 Tesla 描述为已实现神经规划——TDS 明确写了 "not yet in use"，这是容易踩的坑。
