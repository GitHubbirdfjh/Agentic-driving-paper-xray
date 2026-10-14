# 5.52 秒：端侧 LLM 的延迟真相，以及一篇论文如何诚实地绕开它

> **信息块**
>
> - **标题（原文）**：Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models
> - **作者**：Nicolas Baumann, Cheng Hu, Paviththiren Sivasothilingam, Haotong Qin, Lei Xie, Michele Magno, Luca Benini（ETH Zurich + 浙江大学）
> - **会议**：RSS 2025（Robotics: Science and Systems XXI, Los Angeles；经 roboticsproceedings.org 官网程序册 p140 核实；arXiv:2504.11514v1，2025-04-15）
> - **链接**：https://arxiv.org/abs/2504.11514
> - **代码**：https://github.com/ForzaETH/LLMxRobot（已 clone 全库核对：训练配置、prompt 文件、llama.cpp 推理脚本与论文一致；该仓库同时承载了后续 CoRL 2025 的 RobotxR1 工作）
> - **一句话核心洞见**：端侧 LLM 的延迟决定了它永远进不了控制闭环——这篇论文最有价值的不是"LLM 能开车"，而是诚实地接受 5 秒级延迟、用"LLM 只做慢环监督、MPC 保底安全"的分层把延迟包起来，再用 RAG + LoRA 蒸馏 + Q5 量化三件套把 7B 小模型在 Jetson Orin 上跑到可用。

---

## 先看那个决定一切的数字：5.52 秒

表 IV（计算性能）是全文第一张该看的表。Jetson Orin AGX 上，Qwen2.5-7B 用 Q5_k_m 量化 + llama.cpp 跑，22.12 tokens/s，生成 121 个 token 需要 **5.52 秒**。FP16 不量化是 2.11 tokens/s，64 个 token 要 30.48 秒——半分钟。Phi3-mini-3.8B 量化后快一些：32.47 tokens/s，154 个 token 4.80 秒。

这就是端侧 LLM 的物理现实：哪怕 7B 模型、哪怕量化到 5.5 bit、哪怕 Orin AGX（275 TOPS），一次像样的推理也要 5 秒。自动驾驶的控制闭环是 10–50Hz（F1TENTH 的 MPC 跑 50Hz），5 秒在控制眼里是永恒。所有"LLM 直接控车"的方案，撞上的第一堵墙就是这 5 秒。

论文的架构是被这个数字逼出来的，而不是设计出来的。III-C 藏着全文最诚实的一句话："the inference latency of the LLM is decoupled from the control frequency of the MPC"（LLM 的推理延迟与 MPC 的控制频率解耦）。翻译成人话：**LLM 慢就让它慢，控制环自己转，LLM 偶尔伸手调一下参数**。这个"慢环监督 + 快环保底"的分层，是整篇论文真正的骨架，后面所有的 RAG、LoRA、量化都是让这个骨架在 Orin 上立得住的工程。

## 两级 LLM + 一级 MPC：谁负责什么

系统叫 LLM-enhanced ADS，三层：

**DecisionxLLM（裁判）。** 每 2 秒看一次机器人状态快照（位置、速度、离墙距离，4–5 个采样点），对照人类的一句自然语言指令（"Drive normally!"），做二分类：a) 继续当前行为，b) 改变行为 + 一句自然语言修正意见。注意它**不输出任何控制量**，只输出判断和文本意见。

**MPCxLLM（翻译）。** 把裁判的自然语言意见翻译成 MPC 的参数（cost 权重、约束边界），通过 ROS dynamic reconfigure 发给 MPC。

**MPC（保底）。** 经典的运动学 MPC，固定频率跑，约束满足保证还在。LLM 再怎么幻觉，也只是在调权重和边界——车不会因为 LLM 胡说就撞墙，因为 MPC 的约束集 X、U 还在起作用。这是全文安全论证的全部：**LLM 的犯错半径被限制在参数空间里**。

这个分层的代价在 VI Limitations 里作者自己说了：慢（抓不住高频行为细节）、纯文本（看不见图像）。收益是：它是 2025 年少有的**在真车上跑起来**的端侧 LLM 系统，而不是仿真 demo。

## MPC 公式：LLM 到底在调什么

理解这篇论文必须先看懂 MPC 在优化什么，否则 "LLM 调参"就是空话。状态是 Frenet 坐标系下的五维向量：

$x = [s,\ n,\ \Delta\phi,\ \delta,\ v]^T$

s 是沿参考轨迹走了多远，n 是横向偏离（离赛车线多远），Δφ 是航向角误差，δ 是前轮转角，v 是速度。输入是两个增量：$u = [\Delta\delta,\ a]^T$（转角变化量、纵向加速度）——用增量模型是为了让控制输入平滑。

运动学自行车模型（式 1）：

$\dot{s} = \frac{v\cos(\Delta\phi)}{1-\kappa_r(s)\,n},\quad \dot{n} = v\sin(\Delta\phi),\quad \dot{\Delta\phi} = \frac{v\tan\delta}{L} - \kappa_r(s)\frac{v\cos(\Delta\phi)}{1-\kappa_r(s)\,n}$

κr(s) 是参考轨迹的曲率，L 是轴距。直觉：车速 v 越大，横向偏离 n 长得越快（ṅ = v·sinΔφ）；转弯时分母 1−κn 修正内侧/外侧的弧长差异。

MPC 的 cost（式 4），逐项翻译：

$J = \sum_{i=0}^{N-1}\big[q_n n^2 + q_v(v-v_{ref})^2 + q_\alpha\Delta\phi^2 + \|\Delta u\|_{q_R}\big] + \text{终端项}$

- $q_n n^2$：别偏离赛车线（横向）
- $q_v(v-v_{ref})^2$：速度跟上参考值
- $q_\alpha\Delta\phi^2$：车头别歪
- $\|\Delta u\|_{q_R}$：方向盘和油门别乱抖（qΔδ、qac 两个对角元）
- 终端项是同样三项在预测 horizon 末端再加一次，逼轨迹收敛

约束：速度/转角限幅、横向误差 n 不许出赛道边界（再加一个 **boundary inflation ε**，边界内缩系数，越大越保守）、输入变化率限幅。

LLM 能调的参数就是 Listing 1 那张表（代码 `prompts/mpc_base_memory.txt` 与论文完全一致，我逐行对过）：qv∈[0,2]、qn∈[0,100]、qalpha∈[0,100]、qac∈[0,1]、qddelta∈[0,100]、alat_max∈[0,20]、a_min∈[−20,0]、a_max∈[0,20]、v_min∈[−2,5]、v_max∈[−1,10]、track_safety_margin∈[0,1]。每个参数有名字、范围、默认值，prompt 里用大写警告 "STRICTLY ADHERE TO THESE NAMES, DONT INVENT NEW ONES!!!"。

看一个真实例子（Bubble 2c）。人类指令 "Reverse the car!"（倒车），MPCxLLM 输出：

`{qv:0.1, qn:40, qalpha:50, ddelta_min:-5, ddelta_max:0, dv_min:-50, dv_max:-1, v_min:-1, v_max:-1, boundary_inflation:0.1}`

核心动作是 `v_min:-1, v_max:-1`——把速度上下界都钉死在 −1 m/s，MPC 被迫倒车。qn=40、qalpha=50 是让倒车时也别偏出赛道。这就是"自然语言→参数"的全部机制：**LLM 不需要懂优化，它只需要知道"想倒车就把 v_min/v_max 设成负的"这种映射**——而这种映射正是 RAG 里写好的（见下节）。

## RAG：两份"驾驶学校小抄"，全文最大的杠杆

RAG 在这里不是检索维基百科，而是两份手写的领域知识小抄，在 prompt 里按相似度取 top-k 塞进去。

**Decision RAG**（Listing 3，附录 C）：把"正常开车"的量化定义写死——"没特别说明时，车速正常范围 5–7 m/s""离墙 <0.4 m 算近""d-speed >0.5 m/s 算高""d 坐标 >0.3 m 算偏离赛车线""d 坐标正负振荡超 0.3 m 算振荡"。**MPC RAG**（Listing 2，附录 B）：11 条 scenario→action 映射，比如"想强制前进就把 v_min 设正，想倒车就设负""v_max 必须大于 v_min""想减小横向加速度就把 alat_max 设小"。

表 II 的数字说明 RAG 是全文最粗的一条腿：平均 **+7.35 个百分点**（GPT-4o +10.8、Phi3 +6.54、Qwen +4.72）。为什么 RAG 这么管用？因为对于 3.8B/7B 的小模型，"从 2 秒的状态数字里判断车开得正不正常"本来是弱项——它没有自动驾驶的先验。但 RAG 把判断标准**直接写成了 if-then 规则**，小模型从"推理"降级成"查表+套用"。这是端侧部署的一个通用教训：**小模型的知识缺口，用 prompt 里的显式规则补，比把模型做大便宜得多**。

但这里有个必须指出的循环论证。附录 C 说评测标签是**程序化生成**的：比如指令 "Reverse the car!" 的标签就是"速度是否为负"。而 RAG hints 里写的正是同一套阈值。于是"RAG 提升 7.35pp"里面，有一部分是"答案的阈值就写在提示里"——模型照着抄就能对。这不是作弊（RAG 本来就是这么用的），但读者要明白这个数字的含义：它测的是"小模型套用显式规则的能力"，不是"小模型理解驾驶场景的能力"。

## LoRA 蒸馏：150 步的轻量微调

小模型光有 RAG 还不够（Phi3+RAG 78.69% vs GPT-4o+RAG 92.48%，差 14 个点），于是用 GPT-4o 当老师蒸馏。流程：程序化随机生成机器人状态/RAG 参数 → GPT-4o 生成"状态描述+判断" → 得到 626 对 Decision 数据、150 对 MPC 数据 → unsloth 框架 LoRA 微调（rank 16、alpha 16、lr 2e-4、batch 2×4、**只训 150 steps**，代码 `train/config/sft_train.yaml` 已核对）。

150 steps 是个很轻的微调——按 batch 8 算，总共只看了 1200 个样本步。效果：平均 **+4.38pp**（Phi3 +3.91、Qwen +4.85），RAG+LoRA 合计相对 base 模型 +10.45pp（Phi3）/+9.57pp（Qwen）——摘要里的 "up to 10.45%" 就是这么来的。

两个诚实注脚。第一，论文原话："Although the correctness of the synthetic data is not formally guaranteed"（合成数据的正确性没有形式化保证）——老师 GPT-4o 自己也可能错，蒸馏的是"GPT-4o 的风格"，不一定是真理。第二，表 V 最后一行有个漂亮的负结果：DeepSeek-R1-distill-Qwen2.5-7B（带长 CoT 推理的版本）+RAG 能到 85.52%，反超普通 Qwen，但一做 LoRA（用 GPT-4o 数据 SFT）**掉到 83.28%**——SFT 把推理模型的长思考链压没了，学是学会了"简洁"，代价是推理能力下降。作者没有藏这个结果，还分析了原因（该用 R1 生成的数据做 SFT）。这种"我们的方法在某处失效"的写法，是全文可信度最高的地方之一。

## 量化：Q5_k_m 的 10.5 倍与它的代价

表 IV 逐项读（60 次连续推理取平均，同一 prompt）：

| | RTX 3090 FP16 | RTX 3090 Q5 | Orin FP16 | Orin Q5 |
|---|---|---|---|---|
| Phi3-mini 3.8B | 25.23 tok/s（72 tok / 2.85s） | 148.36 tok/s（5.9×） | 4.71 tok/s（15.29s） | 32.47 tok/s（6.9×，4.80s） |
| Qwen2.5 7B | 11.14 tok/s（50 tok / 4.51s） | 107.52 tok/s（9.7×） | 2.11 tok/s（30.48s） | 22.12 tok/s（**10.5×**，5.52s） |

摘要的 "10.5×" 就是 Qwen 在 Orin 上 22.12/2.11。精度代价：决策任务上 Phi3 +2.35pp（量化后反而涨了，噪声）、Qwen −0.3pp——基本无损；控制适应性上掉 1–6%（Phi3-Q5 的平滑项 ES 从 0.8 退到 1.1，相对基线 −22.3%）。**这是端侧部署最硬的一条经验：Q5_k_m + llama.cpp 是 10 倍吞吐换 <1 个点的精度，必做。**

但有个反常细节：内存占用 FP16→Q5 只从 4.3→3.9GB（Phi3）、7.8→5.6GB（Qwen），降幅远小于"16bit→5.5bit"该有的 3 倍。论文没解释。合理的推测是 KV cache、context（n_ctx=2048）和 llama.cpp 的运行时开销占了大头，权重本身只是一部分——但这只是推测，论文没给内存 breakdown，**这是表 IV 缺的一块信息**。

**模型选型结论**：Qwen2.5-7B 在所有设置下都压过 Phi3-mini-3.8B（决策 87.32% vs 82.60%，控制 52.2% vs 45.1%），作者明说 Qwen 是首选。3.8B 不是不能用，是在"无 RAG"时会出大事（见下节）。

## 表 III：52.2% 是怎么来的，以及 −229.6% 的灾难

控制适应性实验在 F1TENTH **仿真**里做（作者明说仿真优先于真车：省时间、有真值）。四个场景、四个 RMSE：中心线偏离 EC、参考速度 EV、倒车 ER、平滑度 ES。默认 MPC 基线：EC 0.7m、EV 1.6m/s、ER 4.6m/s、ES 1.5m/s²。

"52.2%" 是 Qwen2.5-7B（RAG+LoRA）在四个场景相对基线的**平均**改进：EC 0.4（+37.2%）、EV 0.4（+71.9%）、ER 1.1（+76.4%）、ES 1.1（+23.2%）。GPT-4o+RAG 是 46.3%。

最值得看的是**失败行**。Phi3-mini 无 RAG：EC 直接 N.C.（没跑完），ES 从 1.5 炸到 11.8（**−711.4%**），平均 −229.6%。Qwen 无 RAG：平均 +0.5%，基本没用。翻译：**小模型裸调 MPC 参数是灾难**——它不知道参数含义，乱设一气，车开得比默认参数还糟十倍。RAG 把这条命救回来（Phi3+RAG 从 −229.6% 到 +39.9%）。这个对比是全文对"为什么需要 RAG"最有力的论证：不是锦上添花，是生死线。

三个诚实瑕疵。第一，作者原话："the closed-loop experiment is not deterministic"——闭环仿真不确定，取决于 MPC 求解质量和地图。**没有随机种子、没有误差棒**，52.2% vs 46.3% 这种差距很可能在噪声带里。第二，表 III 里 Phi3-Q5 行的平均改进印成了 "44.%"——缺了个数字，排版事故。第三，GPT-4o 无 RAG 行标了 †（"average over completed runs, excluding N.C."）——EC 那格 N.C. 被排除在平均之外，20.2% 这个数是挑过子集的。

## 完整推演：一次"开得不正常"的判定与一次撞墙恢复

**场景一（Bubble 1）：** 人类说 "Drive normally!"。2 秒 4 个采样点：s 坐标 19→22（往前走），d 坐标 −0.6/0.6/−0.65/0.61（**左右横跳**），d-speed ±1.2（RAG 说 >0.5 算高），s-speed 1.0（RAG 说正常是 5–7），右墙距离 0.1m（RAG 说 <0.4 算近）。DecisionxLLM 的输出把每条都对上 RAG："速度低于正常范围、d 坐标振荡、离墙太近"→ 选 b)，指令："把 s-speed 提到 5–7、减小振荡、回中线"。注意它的推理**完全是 RAG 规则的套用**，没有超出小抄的信息——这就是 RAG 版 DecisionxLLM 的真实能力边界。

**场景二（附录 D，真车撞墙恢复，三轮对话）：** 第 1 轮，车头顶墙（crashed=True，右墙 0.15m，速度≈0）。Qwen 判 b)"倒车脱困"，MPCxLLM 输出 `{..., v_min:-2.0, v_max:-1.0, ...}`——钉死负速度，车开始倒。第 2 轮，车还在墙边但已在倒，Qwen 判"继续倒"（正确：别中途停）。第 3 轮，crashed=False、车已远离右墙（左 0.37m/右 2.8m），Qwen 判"停止倒车、恢复前进"，MPCxLLM 输出 `{v_min:1.5, v_max:1.5, qn:100, qalpha:100}`——速度钉回正，横向/航向权重拉满（100 是范围上限）让车 aggressively 回线。三轮里 Decision 和 MPC 的配合是严丝合缝的。

但第 1 轮有个作者自己承认的 hallucination：Qwen 写"d-speed is high"，而实际 d-speed 是 0.09（很低）。Human Narration 原话："albeit a reasoning error on stating that the d-speed is high, but this did not change the overall decision-making."——**解释错了，结论蒙对了**。这个细节比所有成功数字都重要：它说明 DecisionxLLM 的"推理过程"不可信，可信的只是它套 RAG 规则的那部分。做安全论证时，只能把 MPC 的约束当依靠，不能把 LLM 的解释当依靠。

**参数名幻觉（论文展示了但没讨论）：** Bubble 2c 里模型输出了 `ddelta_min:-5, ddelta_max:0, dv_min:-50, dv_max:-1`——这四个名字**不在** base memory 的允许列表里（代码里大写警告不许发明新名字）。ROS dynamic reconfigure 收到未知参数名会怎样？论文和代码都没说。这是全文最大的未闭合回路：LLM 偶尔会编参数名，系统靠什么兜住？可能是解析器静默丢弃，可能是报错——**未知**，审稿人该问，读者要知道这是个坑。

## 怀疑式阅读：这篇论文的边界

**评测的循环性。** 决策准确率的标签是程序化生成的（8 条指令 × 200 状态），标签规则和 RAG hints 是同一套阈值。于是"RAG +7.35pp"测的不是驾驶理解，是"按小抄办事"的能力。结论依然成立（端侧小模型就需要小抄），但数字的含义要打折扣。

**仿真与真车的鸿沟。** 表 III 的 52.2% 全是仿真数字，且作者自声明非确定、无误差棒。真车部分（图 5/6、附录 D）**全部是定性的**：before/after 照片、速度曲线图、聊天记录。没有真车上的 RMSE、没有成功率统计。RSS 接受"真机定性 + 仿真定量"的组合，但读者要清楚：这篇论文没有证明"端侧 LLM 在真车上把控制指标提高了 X%"。

**纯文本的局限（作者自认）。** DecisionxLLM 的输入是数字快照，不是图像。Limitation 节说得很直白：下一步是多模态，但 Orin 的算力逼他们先用纯文本 LLM。这意味着这套系统**看不见**红绿灯、行人、锥桶——它只知道自己离墙多远、速度多少。它的适用域是"行为风格调整"（开稳一点、离墙远一点、倒车），不是"场景理解"。

**训练量的诚实度。** LoRA 只训 150 steps、626+150 对合成数据，老师是 GPT-4o（正确性无保证）。这个配置下 +4.38pp 是"蒸馏 GPT-4o 的输出风格"能拿到的，不是"学会了驾驶知识"。附录 E 的 R1-distill 负结果反过来证明了这一点：SFT 压的是风格，不是推理能力。

公道话：附录 E 敢放 R1-distill 被 LoRA 搞坏的负结果、附录 D 敢承认 d-speed 误判、VI 敢写"慢、纯文本、偶尔推理错"——这种**不藏拙**的写作，在 LLM4AD 论文里是少数。表 IV 的延迟数字是全领域端侧部署都能引用的硬数据。

## 作者的底牌

ETH 的 ForzaETH 车队（F1TENTH 1:10 真车 + 完整的 race stack，JFR 2024 有论文打底）是第一张牌：别人还在仿真里调参，他们直接上真车，这是 RSS 最吃的证据形式。第二张是**时机**：2025 年 4 月，端侧 LLM 部署在机器人领域刚起步（llama.cpp + Q5_k_m 这套做法还没被写烂），"第一个把 7B LLM 塞进 Orin 跑真车调 MPC"的身位被他们占了。第三张是**写作策略**：把"LLM 慢"这个最大弱点提前摊牌（decouple 延迟），反而让审稿人没法用"实时性"攻击它——攻击点被作者自己写进了 Limitation。

## 后续：这条线没停

同一仓库的 README 显示，作者团队的 **CoRL 2025 论文 RobotxR1**（arXiv:2505.03238）在这套系统上继续做了闭环 RL 训练 LLM 开车：Qwen2.5-3B 达到 63.3% 控制适应性，超过 GPT-4o 的 58.5%。说明"小模型+RL"正在接替"蒸馏+prompt"的路线。RAMP 若借鉴 124 的工程结论，应同时跟踪 RobotxR1 的 RL 部分。

## 与 RAMP 的血缘 / 划界

**血缘（124 是 RAMP 时延优化部分最直接的工程参照，逐条列可借鉴点）：**

1. **延迟预算表（直接抄作业）。** Orin AGX 上 Qwen2.5-7B-Q5 ≈ 22 tok/s、120 token 约 5.5 秒；Phi3-mini-3.8B-Q5 ≈ 32 tok/s。这给 RAMP 的 LLM 调度器定了**延迟天花板**：任何需要 LLM 在线的决策，粒度不能小于秒级。RAMP"LLM 调度多规划器"的分层思想与 124 的"LLM 解耦于控制频率"是同一条设计线——124 是这条线在真车上的存在性证明。
2. **量化选型结论。** Q5_k_m + llama.cpp：10.5 倍吞吐、决策精度损失 <1pp、控制适应性损失 1–6%。RAMP 端侧部署 LLM 调度器时，这是经过验证的默认选项（配合 temperature=0、top_k=1 贪婪解码保证可复现，见 `inf_gguf.py`）。
3. **模型选型结论。** 同代 7B（Qwen2.5）全面压过 3.8B（Phi3），尤其在无 RAG 时 3.8B 会灾难性失效（−229.6%）。RAMP 选调度器底座时，7B 是下限，3B 档需要更强的护栏。
4. **RAG vs 情景记忆的对照。** 124 的 RAG 是**静态小抄**（手写阈值+映射规则），RAMP 的情景记忆是**动态经验库**（风险加权衰减）。两者都是"给小模型补知识"的手段，但 124 证明了静态规则的天花板（循环论证问题）——这正是 RAMP 用"从闭环经验里学"而非"手写规则"的动机来源，写论文时可作为对比论据。
5. **"LLM 可以犯错、底层必须兜底"的思想同构。** 124 用 MPC 的约束集兜底 LLM 的参数幻觉；RAMP 用闭环风险校验兜底 LLM 调度器的选择。这是同一安全哲学的两个实例，related work 里可以并置。
6. **负结果的警示。** SFT 会压垮推理模型的长 CoT（R1-distill 85.52%→83.28%）；LLM 会编造参数名（Bubble 2c）。RAMP 若对调度器做微调/蒸馏，必须保留推理链的评测，不能只看最终选择准确率。

**划界（本质不同的四处）：**

1. **调度的对象不同。** 124 的 LLM 调的是**连续控制器参数**（qv、qn、速度边界）；RAMP 的 LLM 调度的是**离散规划器**（选哪个 planner、旁路哪个模块）。前者是参数自适应，后者是结构自适应——RAMP 的调度粒度更粗、但对实时性要求更高（100ms 级 vs 124 的秒级）。
2. **延迟差两个数量级。** 124 证明了"5 秒慢环可行"，但 RAMP 的在线调度需要 100ms 级响应。124 的数字**不能**直接论证 RAMP 的实时性，只能论证"分层解耦"这条路走得通——RAMP 的快环（规划器本身）必须不经过 LLM，这是 124 架构给 RAMP 的硬约束。
3. **无世界模型、无记忆、无 Bench2Drive。** 124 的"知识"是手写 RAG，没有未来预测、没有经验积累，评测在 F1TENTH 仿真+真车定性。它不是 RAMP 的竞品，是 RAMP **工程侧**的参考文献（部署、量化、延迟），不是方法侧的。
4. **输入模态不同。** 124 是纯文本状态快照（看不见图像）；RAMP 是多模态输入。124 的 Limitation 恰好是 RAMP 的起点——但 RAMP 要为此付出延迟代价，124 的表 IV 就是那张代价表。

**一句话定位**：124 是 RAMP 的"工程部"参考文献——延迟预算、量化选型、模型选型、慢环分层架构都可以直接引用；但它不碰规划、不碰世界模型、不碰记忆，在 RAMP 的 related work 里应放在 "On-board LLM deployment and latency" 小节，与方法主线的多规划器调度、世界模型分开。
