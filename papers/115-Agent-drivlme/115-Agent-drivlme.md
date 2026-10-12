# DriVLMe：让开车 agent 既会看路，也会跟人聊天

> **信息块**
> - 论文标题：DriVLMe: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences
> - 作者：Yidong Huang, Jacob Sansom, Ziqiao Ma, Felix Gervits（ARL）, Joyce Chai —— 密歇根大学，通讯作者 Ziqiao Ma（marstin@umich.edu）
> - 会议：IROS 2024
> - arXiv：2406.03008v2（v1 2024-06-05，v2 2024-10-15）
> - 链接：https://arxiv.org/abs/2406.03008 ｜ 项目页 https://sled-group.github.io/driVLMe/
> - 代码：官方 https://github.com/sled-group/driVLMe（本目录 `code/`；clone 过程因网络中断多次重试，最终以 `git fetch` 取回，见下文说明）
> - PDF：本目录 `2406.03008.pdf`
> - 一句话核心洞见：**“具身经验”（在仿真器里开车攒的经历）和“社交经验”（跟真人对话学来的表达）是两种不同的训练信号，要分开喂**——前者教会模型“路是什么样的”，后者教会模型“人话怎么说”；而这篇论文最值钱的部分不是它的指标，是它诚实地记录了 LLM 开车 agent 在闭环真人实验里是怎么失败的。

---

## 为什么这篇论文要同时做“开车”和“聊天”两件事

先看它要解决的问题。2024 年上半年，LLM 进驾驶的主流做法分两类：一类是 DriveGPT4、DriveMLM，把驾驶视频喂给多模态 LLM，让它预测油门和方向盘；另一类是 DiLu、Agent-Driver，用纯文本场景描述做推理和规划。两类都有一个共同的盲区：**它们只处理单轮、短视界的指令**——“左转”“超车”这种说完就忘的命令。

真实世界里乘客是怎么跟车说话的？论文的起点是 SDN（Situated Dialogue Navigation）这个 benchmark：人在 CARLA 里坐“自动驾驶车”，以为车是自动的，其实背后是个真人 wizard 在操作；还有一个捣乱的 wizard，时不时改天气、在前面摆障碍物、让乘客临时改目的地。8,000 多句对话、18.7 小时的交互录下来，乘客说的是这种话：“我好像到 IKEA 了，接下来去哪？带我去 Annabel 家吧……Annabel 家在哪？……算了直接沿 Duffield 走，到 Upland 右转。” 这是**多轮、目标会变、环境会变**的对话，单轮指令模型处理不了。

所以 DriVLMe 的设计目标是：一个 video-LLM agent，既能看视频理解路况（embodied），又能跟人多轮对话、理解意图变化（social），还能在对话里**主动调用一个路线规划工具**。它回答的问题是：foundation model 到底能不能同时干好“开车”和“聊天”这两件事？论文的诚实之处在于，答案是“部分能”，而“不能”的部分被完整记录了下来——这在 LLM 驾驶论文里很少见。

## 架构：三个部件，拼法很直白

整个系统就是 Video-ChatGPT 的驾驶特化版，三个部件：

**1. Video tokenizer（眼睛）。** 输入是历史视频帧，窗口 $T_{max}=40$、步长 $\Delta t=2$（也就是从过去 80 帧里每隔一帧抽一帧，10Hz 采样下覆盖约 8 秒）。每帧过 frozen 的 CLIP ViT-L/14 得到特征图 $f \in \mathbb{R}^{T \times h \times w \times D}$。然后做两件事：沿时间维平均池化得到空间表示 $v_s \in \mathbb{R}^{(h \cdot w) \times D}$（“每一块地方在这段时间里平均长什么样”），沿空间维平均池化得到时间表示 $v_t \in \mathbb{R}^{T \times D}$（“每一时刻整幅画面平均长什么样”），拼起来 $v = [v_t; v_s] \in \mathbb{R}^{(T + h \cdot w) \times D}$，再过一个**单层线性投影** $g$ 映射到 LLM 的词嵌入空间，维度 $K$。

这个拼法值得停一下：它没有用任何时序建模（没有 3D 卷积、没有视频 transformer），就是“时间平均 + 空间平均 + 拼接”。为什么这样能 work？因为驾驶视频里真正需要时序精度的信息（旁边那辆车是在加速还是减速）其实被这个粗糙表示丢掉了——论文后面自己也承认，视觉理解是短板之一。作者的赌注是：**高层决策不需要像素级时序**，知道“前面有车、红灯、我在第三车道”就够了。这个赌注对一半：开环指标上够用，闭环里遇到前车急停就露馅（后面细说）。

**2. Route planner（工具）。** 一个图最短路规划器，输入是 LLM 给出的地标名 $lm$，输出是从当前位置到地标的一串转弯方向 $[dir_1, dir_2, \dots]$，每个 $dir_i \in \{\text{left, right, straight, uturn}\}$。LLM 通过输出 `plan(ikea)` 这种文本调用它，返回的路线再喂回 LLM。这是全文最“agent”的一笔：**LLM 不自己算路线，它知道自己算不好，把符号化的部分外包给工具**。闭环实验里最亮眼的发现就是模型真的学会了在目标变更时重新调用 planner（`plan(shell)` → `plan(kfc)`）。

**3. LLM backbone（脑）。** Vicuna-7B v1.1 + LoRA（rank 128，alpha 256）。系统提示词规定了固定的交互流程：先描述看到的环境 → 调用规划工具 → 选导航动作 + 回复乘客。动作空间是 SDN 定义的 8 个高层离散动作：LaneFollow、LaneSwitch、JTurn、UTurn、Stop、Start、SpeedChange(±5km/h)、LightChange。注意：**LLM 只输出高层离散动作，真正的油门方向盘由一个 local motion planner 翻译**——这和 Drive As You Speak 的“脑-手分层”是同一个思想，只是这里落成了代码。

## 训练：三段式，每段喂不同的经验

训练分三段，对应标题里的三个词：

**第一段：domain video instruction tuning（看懂驾驶视频）。** 冻结 CLIP 和 LLM，只训练那个线性投影层。用 BDD-X 的 5 万个视频-文本对（真实驾驶视频 + 环境描述和动作解释标注），训 3 个 epoch，学习率 2e-5，batch size 4。这一步只做模态对齐：让 LLM 的词嵌入空间能“读懂”视频 token。

**第二段：social instruction tuning（学会跟人说话）。** 在 SDN 数据集（1.3 万个视频-对话对）上用 LoRA 微调 LLM + 投影层，2 个 epoch，batch size 1（batch 这么小是因为视频 token 太长）。SDN 的数据是真人对话，包含目标变更、路线重规划，作者还手工标注了 plan change（根据轨迹和对话判断车什么时候换了目标）。这一步喂的是**社交经验**。

**第三段：embodied instruction tuning（攒开车经历）。** 把 SDN 训练时的 CARLA 回放重跑一遍，从仿真器里抽真值（到路尽头的距离、所在车道、能不能变道、前方障碍物距离、可见交通标志、天气），填进**手写模板**生成描述文本，比如“I'm on the {lane} lane from the left of the road”“There is a {sign} that is {distance} meters from me, showing {state}”。这一步喂的是**具身经验**——但注意，它是模板生成的，不是模型自己从交互里学到的。

三段训完，BDD-X 上又加训了 9 个 epoch（LoRA rank/alpha 256，lr 5e-5）做真实场景适配。

## 开环评测：在 SDN 上确实赢了，但赢在哪里要细看

两个任务：**NfD**（Navigation from Dialogue，根据对话预测下一步物理动作，指标是动作准确率 Act、参数准确率 Arg、移动相关 Move）和 **RfN**（Response for Navigation，生成对话回复，指标是 CIDEr、BERTScore、METEOR）。

主结果（seen/unseen 地图）：DriVLMe 在绝大多数指标上超过 TOTO（episodic transformer 基线）、GPT-4、GPT-4V 和 DriveVLM。一个细节：unseen 地图（Town02，训练没见过）上 NfD 的动作准确率，DriVLMe(72.9) 反而略高于 DriveVLM(70.8)，但 seen 地图上 DriveVLM 的 Move 指标（61.4）远高于 DriVLMe(-social) 的 19.1——说明**动作预测能力主要来自 planner 工具和具身数据，而不是 LLM 本身的推理**。

消融表（Table 2）是全文信息量最大的表，逐行读：

- **去 video（-video）**：RfN 全线掉（CIDEr 0.45→0.33 seen），说明对话回复的质量依赖视觉输入——模型不是在纯背话术。
- **去 planner（-planner）**：NfD 的 Move 指标从 62.7 掉到 21.3（seen），掉得最狠。**路线规划工具是导航动作预测的第一功臣**，比 LLM 本身重要。
- **去 social 数据（-social）**：RfN 掉得多（CIDEr 0.45→0.17），NfD 动作准确率基本不动。社交经验管“说话”，不管“开车”。
- **去 embodied 数据（-embodied）**：NfD 动作准确率几乎不动（68.4 vs 68.7），只在 Move 上略有影响。作者的解释是：模板生成的具身描述主要帮助的是变道这类非路线动作，而 Town02 这种简单地图用不上。

这张表的诚实之处在于，它证明了一个反直觉的结论：**这个系统里 LLM 的边际贡献主要在“说话”和“调用工具”上，“开车”的活基本是 planner 工具和模板数据干的**。这对 RAMP 是个提醒：agent 架构里，要分清能力到底来自 LLM 还是来自工具，消融要做到工具级。

BDD-X 真实数据上的结果：DriVLMe 的 justification（动作解释）CIDEr 132.17，超过 DriveGPT4 的 101.55；但控制信号预测（speed RMSE 1.59 vs 1.30，turning angle RMSE 33.54 vs 8.98）明显不如 DriveGPT4。作者没有回避：DriVLMe 的强项是**解释和对话**，弱项是**精确控制**——因为它输出的是离散高层动作，DriveGPT4 直接回归连续控制量。这是架构选择带来的固有 trade-off，不是调参能解决的。

## 闭环真人实验：20 次里成了 6 次，失败案例比成功案例值钱

这是全文最有价值的一节。20 次 CARLA 闭环真人实验，人在回路里用自然语言指挥，指标是“最终到达目标”。成了 6 次。作者没有把 14 次失败藏起来，而是分类记录：

1. **多轮指令跟丢**：对话一长，模型会忘记之前定的长期目标，把一句低层指令（“左转”）误解成“放弃原目标”，重新预测一个错误的目标。这是 LLM 上下文管理的经典失败，也是**情景记忆**要解决的问题——RAMP 的自适应衰减记忆正好对着这个坑。
2. **前车急停没刹住**：timestamp 7 的记录里，前车突然停下，agent 没有输出 Stop，直接撞上去（或险些撞上）。根因是视频 tokenizer 的粗糙时序表示 + 5 秒一次的推理频率（2Hz 交互，单次推理约 5 秒），根本反应不过来。
3. **路口熄火**：最后一个路口没预测出转弯，车停在路口中间不动。planner 给了路线，但 LLM 在动作选择这一步掉了链子。
4. **推理时延不可接受**：单次推理约 5 秒，仿真里可以靠步进仿真等它，真车上这是致命的。论文明确说这是部署前必须解决的问题，方向是蒸馏、硬件加速、缩短 CoT。

成功的 6 次里，最漂亮的是**目标变更重规划**：乘客先说“去 Shell”，开到一半改口“不去 Shell 了，去 KFC”，agent 重新调用 `plan(kfc)`，更新目标并继续开。这个案例证明了工具调用范式在动态任务里的价值——**目标变了，不需要重训，只需要重新调一次工具**。

还有一个作者诚实记录的细节：模型生成的语言回复过于简单（yes/no 式），**不会主动向人提问**（比如“前面路况复杂，你要我慢点还是绕行？”）。一个不敢问人的 agent，在不确定场景下只能瞎猜，这是信任度的硬伤。

## 怀疑式阅读

**数据层面**：embodied 数据是模板生成的，模板是作者手写的 5 类句式。这意味着“具身经验”这个听起来很 embodied 的东西，实际上是**结构化真值 → 固定句式**的翻译，信息量完全被模板的天花板锁死。模型从中学到的不是“开车经验”，而是“把仿真器真值复述成英文”。这解释了为什么 -embodied 消融几乎不掉点。

**评测层面**：开环评测是 teacher-forcing 的（每一步都喂真值历史），闭环只有 20 次 pilot、没有统计显著性、没有对照组（比如同一个任务让人直接开车对比）。6/20 的成功率，作者定性为“promising”，这个定性是慷慨的——但考虑到这是 2024 年中 LLM agent 闭环真人实验的早期尝试，记录失败本身的价值大于数字。

**基线层面**：GPT-4/GPT-4V 基线只在每个 strata 抽 200 个点评测（作者注明是算力限制），而且 prompt 是作者自己设计的。DriveVLM 基线的数字没有注明是复跑还是引用。跨表对比时要留心。

**一个没有说破的假设**：整个系统假设高层离散动作（8 个）足以覆盖驾驶需求，底层 local motion planner 能把离散动作翻译成安全轨迹。但闭环失败案例 2（前车急停）恰恰发生在这个假设的缝隙里：Stop 动作的触发时机需要连续感知支撑，离散动作 + 5 秒推理周期的组合，在紧急场景下是结构性不够用的。

## 代码核对

官方仓库 https://github.com/sled-group/driVLMe（基于 Video-ChatGPT 二次开发），已完整 clone 到本目录 `code/`（单 commit：`4bdcc6b init commit`；clone 过程因网络中断多次重试，最终用 `git fetch --depth 1 origin <commit>` 取回）。仓库结构与论文对应：`drivlme/` 下有 `single_video_inference_SDN.py`（开环推理入口，第 188 行解析模型输出中以 `plan(` 开头的工具调用）、`single_video_inference_bddx.py`、`video_conversation.py`；`drivlme/model/video_chatgpt.py` 为 Video-ChatGPT 骨架；`drivlme/train/` 下 `train_drivelm.py` 为训练入口；`evaluation/` 下有 `physical_action_acc.py`（NfD 动作准确率）和 `diag_action_acc.py`（RfN 对话评估），与论文 5.2 节的指标定义对应。README 披露复现需要三件套权重：LLaVA-Lightning-7B 基座 + BDD-X 预训练的 projector（`mm_projector.bin`）+ SDN 微调的 LoRA 权重，权重托管在 Dropbox 外链——复现门槛主要在权重和 CARLA 环境，不在代码本身。

## 与 RAMP 的血缘/划界

**血缘有三条**，都很实在。第一，**工具调用范式**：DriVLMe 的 `plan(landmark)` 是 Agent-Driver 工具库思想在导航场景的实例化——LLM 不自己算路线，调工具。RAMP 的多规划器调度是同一范式的放大：调度的对象从“一个路线规划器”变成“多个异构规划器”，调度的依据从“对话意图”变成“世界模型风险预测 + 情景记忆”。第二，**分层控制**：LLM 只出离散高层决策，连续控制下沉给传统模块——RAMP 的“LLM 调度 + 底层规划器执行 + 安全盾校验”继承了这一分层。第三，**失败模式的清单**：DriVLMe 诚实记录的四个坑（多轮跟丢目标、前车急停、推理时延、不敢主动提问），RAMP 的四个模块恰好一一对着——情景记忆对跟丢、世界模型+风险校验对急停、异步流水线+中间分布复用对时延。读这篇，等于提前看到了 RAMP 要解决的问题清单。

**划界也有三条**。第一，DriVLMe 的 agent 是**对话驱动**的，没有对话就没有决策；RAMP 的调度器是**每帧在线**的，不依赖人类指令。第二，DriVLMe 没有记忆检索机制（对话历史是全量塞进 prompt 的，多轮就爆），RAMP 的情景记忆是检索式的、可衰减的。第三，DriVLMe 没有安全校验层，LLM 输出的动作直接进执行器；RAMP 的闭环风险校验是独立于 LLM 的否决权。这三条划界，写进 related work 就是“prior art 做到了哪，RAMP 往前多走了哪”。
