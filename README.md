# Agentic Driving Paper X-Ray · 自动驾驶跨领域前沿论文精读 40 篇

[![papers](https://img.shields.io/badge/papers-40-c1502e.svg)](./00-总览.md)
[![domains](https://img.shields.io/badge/domains-6-2e7d4f.svg)](./00-总览.md)
[![method](https://img.shields.io/badge/method-paper--xray-58C4DD.svg)](https://github.com/Wang-auspicious/paper-xray)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](./LICENSE)

以"以 Agent 为中心的自动驾驶"课题为圆心，向外画六条线的 40 篇顶会论文深度精读。

> **课题**：世界模型引导 + 自适应衰减情景记忆的闭环多规划器自动驾驶 Agent——LLM 调度多个专家规划器，执行开放式自然语言驾驶指令，在 Bench2Drive 闭环基准上验证。

这不是论文摘要的复述。每一篇都按 [paper-xray](https://github.com/Wang-auspicious/paper-xray) 的深度解读方法写成：**还原作者真实的思考过程**——前人方法在哪个具体场景撞墙、作者押了什么赌注、哪些设计是承重的、每个关键符号在具体例子上长什么样、作者在哪里有把握、在哪里心虚。

## 40 篇一览

### VLA 驾驶（51–60）——语言模型直接开车走到了哪

| # | 论文 | arXiv | 一句话 |
|---|---|---|---|
| 51 | EMMA | [2410.23262](https://arxiv.org/abs/2410.23262) | 几何量全翻译成纯文本塞进 MLLM；3–5 秒时域没崩、8 秒开始崩 |
| 52 | DriveMoE | [2505.16278](https://arxiv.org/abs/2505.16278) | 稀有行为的病根是 mode averaging；轻量 router 按上下文选专家 |
| 53 | SOLVE | [2505.16805](https://arxiv.org/abs/2505.16805) | VLM 与端到端从串联改成并联双轨：慢参考 + 实时精修 |
| 54 | AutoVLA | [2506.13757](https://arxiv.org/abs/2506.13757) | 轨迹离散成 2048 动作 token；GRPO 同时优化开得准和该快时少想 |
| 55 | ReAL-AD | [2507.12499](https://arxiv.org/abs/2507.12499) | 战略/战术/操作三层认知模型，每层各配一个老师 |
| 56 | MindDrive | [2512.13636](https://arxiv.org/abs/2512.13636) | 两套 LoRA 做在线 RL；Bench2Drive DS 78.04 / SR 55.09% |
| 57 | OmniDrive | [2405.01533](https://arxiv.org/abs/2405.01533) | 规则仿真批量算候选轨迹后果，GPT-4 写成问答造稠密监督 |
| 58 | CoVLA | [2408.10845](https://arxiv.org/abs/2408.10845) | CAN/GNSS/IMU 本身就是动作日志；83 小时真实驾驶数据 |
| 59 | DriveAction | [2506.05667](https://arxiv.org/abs/2506.05667) | 从驾驶员真实动作倒着出题，树状评测定位错误层级 |
| 60 | ADriver-I | [2311.13549](https://arxiv.org/abs/2311.13549) | vision-action 交错序列；在自己创造的世界里开车 |

📖 [领域导读](./vla-driving/导读.md)

### 驾驶世界模型（61–70）——除了 rollout 校验，世界模型还能干什么

| # | 论文 | arXiv | 一句话 |
|---|---|---|---|
| 61 | OccSora | [2405.20337](https://arxiv.org/abs/2405.20337) | 32 帧体素一次压成 4D 潜图，一次扩散画出整段未来 |
| 62 | DOME | [2410.10429](https://arxiv.org/abs/2410.10429) | 把可控配方搬到 occupancy 连续潜空间 |
| 63 | DriveWorld | [2405.04390](https://arxiv.org/abs/2405.04390) | 4D 预训练的本质是把未来预测当预训练任务 |
| 64 | Copilot4D | [2311.01017](https://arxiv.org/abs/2311.01017) | 点云离散成 token，离散扩散 10 步解完一帧 |
| 65 | Scenario Dreamer | [2503.22496](https://arxiv.org/abs/2503.22496) | 向量图元潜空间做扩散；return-conditioned 拧出对抗行为 |
| 66 | SMART | [2405.15677](https://arxiv.org/abs/2405.15677) | 地图+轨迹全离散成 token，多智能体运动变 next-token 分类 |
| 67 | GenAD | [2403.09630](https://arxiv.org/abs/2403.09630) | 瓶颈在采集成本；2000+ 小时 YouTube 教"世界怎么动" |
| 68 | ReconDreamer | [2411.19548](https://arxiv.org/abs/2411.19548) | 世界模型微调成专修重建鬼影的"在线修复工" |
| 69 | Epona | [2506.24113](https://arxiv.org/abs/2506.24113) | 共享潜变量 F：规划与渲染并行消费同一个 F |
| 70 | Panacea | [2311.16813](https://arxiv.org/abs/2311.16813) | BEV 标注画成控制图喂 ControlNet，生成六视角一致视频 |

📖 [领域导读](./world-model-driving/导读.md)

### 视觉语言导航 VLN（71–75）——"听懂指令并执行"隔壁是怎么做的

| # | 论文 | arXiv | 一句话 |
|---|---|---|---|
| 71 | LM-Nav | [2207.04429](https://arxiv.org/abs/2207.04429) | GPT-3 拆路标 + CLIP grounding + ViNG，零微调走真实街道 |
| 72 | NaviLLM | [2312.02010](https://arxiv.org/abs/2312.02010) | schema-based instruction 把五个具身任务统一成"生成下一个词" |
| 73 | NavGPT-2 | [2407.12366](https://arxiv.org/abs/2407.12366) | 推理归 LLM、动作归策略网；导航时 LLM decoder 整个删掉 |
| 74 | MapGPT | [2401.07314](https://arxiv.org/abs/2401.07314) | 在线拓扑地图转写成 prompt，给 GPT-4 全局视野 |
| 75 | NaVid | [2402.15852](https://arxiv.org/abs/2402.15852) | 单目 RGB 视频流即时空记忆，不要地图/里程计/深度 |

📖 [领域导读](./vln/导读.md)

### 多智能体协同驾驶（76–80）——通信/协商思想往单车仲裁迁移

| # | 论文 | arXiv | 一句话 |
|---|---|---|---|
| 76 | V2XPnP | [2412.01812](https://arxiv.org/abs/2412.01812) | 本地融好历史帧再发一次，传得少、精度反而高 |
| 77 | UniV2X | [2404.00717](https://arxiv.org/abs/2404.00717) | 稀疏 query + 稠密概率图混合传输，把协同一路做到规划 |
| 78 | HEAL | [2401.13964](https://arxiv.org/abs/2401.13964) | 异构新智能体接入不重练：统一特征空间 + 后向对齐 |
| 79 | CodeFilling | [2405.04966](https://arxiv.org/abs/2405.04966) | 传整数码而不是特征图；等 AP 下带宽降 1333× |
| 80 | Co-MTP | [2502.16589](https://arxiv.org/abs/2502.16589) | 把自车规划写进预测的条件里 |

📖 [领域导读](./multi-agent-driving/导读.md)

### Agent 迁移应用（81–85）——通用 Agent 机制搬到驾驶上，哪些水土不服

| # | 论文 | arXiv | 一句话 |
|---|---|---|---|
| 81 | AsyncDriver | [2406.14556](https://arxiv.org/abs/2406.14556) | LLM 不必每帧参与：慢思考与快执行解耦 |
| 82 | KoMA | [2407.14239](https://arxiv.org/abs/2407.14239) | 不开口、只读心：LLM 互相揣测意图，排序反思做仲裁 |
| 83 | SurrealDriver | [2309.13193](https://arxiv.org/abs/2309.13193) | 24 名人类司机的"内心独白"移植为 CoT 示范 |
| 84 | LangCoop | [2504.13406](https://arxiv.org/abs/2504.13406) | 自然语言是车队最省带宽的通用语，但幻觉污染零讨论 |
| 85 | AgentsCoDriver | [2404.06345](https://arxiv.org/abs/2404.06345) | 五模块拼出完整 agent 循环 |

📖 [领域导读](./agent-migration/导读.md)

### 主流端到端（86–90）——不看语言不看 Agent，纯端到端的基准线

| # | 论文 | arXiv | 一句话 |
|---|---|---|---|
| 86 | UniAD | [2212.10156](https://arxiv.org/abs/2212.10156) | 规划是整条管线的组织原则；统一 query 接口 |
| 87 | VAD | [2303.12077](https://arxiv.org/abs/2303.12077) | 安全约束写成训练时可微损失，推理零开销 |
| 88 | Hydra-MDP | [2406.06978](https://arxiv.org/abs/2406.06978) | 一台规划器、五个考官：把评测指标蒸进网络 |
| 89 | SparseDrive | [2405.19620](https://arxiv.org/abs/2405.19620) | 不需要稠密 BEV；预测与规划是同一个并行解码问题 |
| 90 | DiffusionDrive | [2411.15139](https://arxiv.org/abs/2411.15139) | 别从纯噪声去噪：先验多模态写进扩散起点 |

📖 [领域导读](./e2e-mainstream/导读.md)

## 阅读路线（给新手）

1. **主流端到端（86→90）**：先知道纯端到端能做到哪——这是所有讨论的基准线。
2. **VLA 驾驶（51→60）**：语言进来之后改变了什么。重点 52、54、56。
3. **驾驶世界模型（61→70）**：重点 69、62、61，直接对应"中间分布复用"和"rollout 校验"。
4. **Agent 迁移（81→85）**：重点 81、82、83，对应异步流水线、风险仲裁、记忆冷启动。
5. **VLN（71→75）**：重点 73、75，拿"调度器+专家"架构论据和记忆尺度论据。
6. **多智能体协同（76→80）**：为多车扩展留接口，重点 79、76 的通信思想。

每个领域目录下的 `导读.md` 有该领域的 landscape、论文关系图谱和细化阅读顺序，读该领域前先看它。全局总览见 [00-总览.md](./00-总览.md)。

## 每篇文档的写法

严格按 [paper-xray](https://github.com/Wang-auspicious/paper-xray) skill（MD 分支）执行：

- **信息块开头**：编号、标题、作者、会议/年份、链接、代码状态、一句话核心洞见；文档标题用自己的话点出洞见，不照抄论文标题。
- **章节标题是只属于这篇论文的具体问题**，不用"背景/方法/实验/总结"这类通用标签。
- **还原作者思考**：前人方法在哪个具体场景撞墙、作者押的赌注、承重设计 vs 装饰设计、关键符号配具体形状的例子。
- **怀疑式审读**：基线选择、消融表解读、数字口径、作者哪里心虚——有一说一。
- **结尾**：一两个可迁移的想法 + "与本课题的关联"，不写"总结与展望"八股。
- **语言**：Hinton 式白话讲机制，3Blue1Brown 式先看见再计算；不堆砌形容词。

## 目录结构

```
├── README.md（本文件）
├── 00-总览.md（六条线横向对比 + 阅读路线 + 未确认汇总）
├── vla-driving/导读.md + 10 个 <slug>-xray/<slug>.md
├── world-model-driving/导读.md + 10 个 <slug>-xray/<slug>.md
├── vln/导读.md + 5 个 <slug>-xray/<slug>.md
├── multi-agent-driving/导读.md + 5 个 <slug>-xray/<slug>.md
├── agent-migration/导读.md + 5 个 <slug>-xray/<slug>.md
└── e2e-mainstream/导读.md + 5 个 <slug>-xray/<slug>.md
```

每篇 `*-xray/` 目录下是独立 Markdown 文档（含自绘机制图）。论文 PDF 与官方代码仓库未收录，请点击上表 arXiv 链接获取原文。

## 诚实性说明

- 所有实验数字均要求出自论文原文；核实不到的在各篇标注"未确认"，未编造。
- 部分论文 venue 未独立确认（如 81/82/83/85、84 的 workshop 身份），引用前请以 arXiv 页面为准；数字口径存疑处（如相对百分比 vs 绝对值）已在各篇挑明。
- 86–90 及部分 VLA 论文的漂亮数字建立在开环/回放式评测上，引用时不要写成闭环能力。

## 参与补充

新增论文从 **91** 开始编号，放在对应领域目录下，并在这份 README 的表格中登记。欢迎提 Issue / PR 补充精读或纠错。

## License

MIT — 论文解读为原创学习笔记，论文本身版权归原作者所有。
