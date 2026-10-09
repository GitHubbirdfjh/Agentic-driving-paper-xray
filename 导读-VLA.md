# 导读 · VLA 驾驶（001–030）

> 对应总览：`00-总览.md`｜论文目录：`papers/001-VLA-*/` ～ `papers/030-VLA-*/`

## 一句话定位

让视觉-语言模型真正"动手开车"的路线——核心矛盾是**语言推理的慢 vs 驾驶决策的快**，以及 **VQA 的"嘴" vs 动作的"手"**之间的鸿沟。

## 必读的 5 篇

1. **012 · DriveLM**（ECCV 2024 Oral）：VLM 驾驶的奠基框架。把驾驶问答组织成图结构（Graph VQA），模拟人类多步推理。读它是为了建立坐标系：之后所有"VLM+驾驶"的工作都在回答它留下的问题。
2. **020 · DiLu**（ICLR 2024）：知识驱动的 LLM 驾驶 agent，首创"记忆—反思—工具调用"循环。单车 agent 范式的源头，不读它就看不懂 061 AgentsCoDriver、071 DriveAgent-R1 在干什么。
3. **011 · SimLingo**（CVPR 2025 Highlight）："VQA 对齐的是嘴，Action Dreaming 对齐的是手"。用离线仿真的多指令轨迹逼模型把语言听进动作里，1B 小模型在 CARLA 与 Bench2Drive 双 SOTA——"语言→动作"对齐问题的标杆答案。
4. **002 · AutoVLA**（NeurIPS 2025）：自适应推理 + 强化微调。代表最新判断：推理步数本身应该是可学的、按场景分配的，而不是固定模板。读它时对照 030 NoRD 一起看。
5. **030 · NoRD**（CVPR 2026）：反方辩手——"不推理也能开"。它的价值不在结论，而在逼问：VLA 的 chain-of-thought 到底是必需品还是奢侈品？读完 002 再读它，效果最好。

## 阅读顺序建议

```
奠基：012 DriveLM → 020 DiLu → 013 DriveVLM
对齐动作：011 SimLingo → 017 RDA-Driver → 014 ORION → 009 ReAL-AD
快与慢：062 AsyncDriver（在 Agent 区，但必读）→ 018 DiMA → 015 DriveGPT4-V2
新范式：002 AutoVLA → 007 MindDrive → 030 NoRD（对照读）
安全规则：023 SafeAuto → 026 Driving with Regulation
数据与评测：003 CoVLA → 008 OmniDrive → 004 DriveAction
其余按兴趣：005 DriveMoE / 006 EMMA / 010 SOLVE / 016 RAG-Driver /
           019 Reason2Drive / 021 VLP / 022 TOKEN / 024 DME-Driver /
           025 Driving with Advice / 027 Driving Everywhere /
           028 LeapAD / 029 PromptTrack
```

## 领域内关键争论点

1. **推理 vs 直接动作**：CoT 思维链（011/014/017）vs NoRD（030）的"零推理"。中间派是自适应推理（002）和混合思考（071）。争论本质：推理的边际收益能否覆盖它的时延与幻觉风险？
2. **数值决策走不走文本解码**：DriveGPT4-V2（015）的 DeciHead 主张"数字不走语言"，用专用回归头；ADriver-I（001）则把控制信号写成"数字词"走语言模型。两条路线在 Bench2Drive 上的对决还没分胜负。
3. **开环 VQA 指标 vs 闭环驾驶**：很多 VLA 论文用问答准确率证明自己，但 SimLingo（011）已经指出嘴和手是两回事。读任何 VLA 论文先看评测：有没有闭环结果？没有的话，结论打几折？
4. **蒸馏还是原生**：DiMA（018）把 MLLM 蒸馏进端到端规划器，试图兼得推理与实时；反对者认为蒸馏丢掉了语言泛化这个 VLA 存在的根本理由。
5. **RL 会不会是必经之路**：AutoVLA（002）、MindDrive（007）、DriveAgent-R1（071）都在往 RL 走。但驾驶的奖励函数设计（安全 vs 效率 vs 舒适）远比围棋复杂，RL 在驾驶上到底是银弹还是新坑，2026 年还没定论。
