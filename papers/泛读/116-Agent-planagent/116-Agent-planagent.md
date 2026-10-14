# PlanAgent：MLLM 不直接开车，它给 IDM 规划器“调参”

> **信息块**
> - 论文标题：PlanAgent: A Multi-modal Large Language Agent for Closed-loop Vehicle Motion Planning
> - 作者：Yupeng Zheng*, Zebin Xing*（共同一作）, Qichao Zhang（通讯）, Bu Jin, Pengfei Li, Yuhang Zheng, Zhongpu Xia, Kun Zhan, Xianpeng Lang（理想汽车）, Yaran Chen, Dongbin Zhao —— 中科院自动化所 + 理想汽车（第一作者实习期间完成）
> - venue：**arXiv 预印本 arXiv:2406.01587v2（2024-06-04），稿件首页注“This work has been submitted to the IEEE for possible publication”；截至 2026-10-11，未发现正式发表 venue**，后续引用该工作的论文（如 AAMAS 2025、arXiv 2502.18600 等）均仍以 arXiv preprint 引用。引用时按预印本处理。
> - 链接：https://arxiv.org/abs/2406.01587
> - 代码：官方 https://github.com/ucaszyp/planagent（本目录 `code/`，已 clone；注意 README 中 installation / training / evaluation 脚本标注未上传，实际只放了 planner 相关代码与可视化脚本）
> - PDF：本目录 `2406.01587.pdf`
> - 一句话核心洞见：**MLLM 最大的不确定性来自让它直接输出轨迹——PlanAgent 的解法是根本不让它碰轨迹，只让它从 27 组预设参数里“点菜”（选 IDM 的速度/加减速/车道参数），再用仿真打分做二次校验；“生成规划器代码”这个说法，代码实现层面要打折扣。**

---

## 先把这篇论文的真实位置摆正

2024 年 6 月，LLM 驾驶 agent 的主流还是开环：Agent-Driver 在 nuScenes 上做开环轨迹预测，DiLu 在 Highway-env 里跑。PlanAgent 要回答的问题是：**MLLM 能不能进闭环**——在 nuPlan 这种大规模真实数据闭环仿真里，每 2 秒做一次决策，连续开完整段路。

它的答案是“能，但有条件”：条件就是**把 MLLM 的输出空间压到最小**。Agent-Driver 让微调后的 LLM 直接生成轨迹 waypoint 序列（连续、高维、LLM 最不擅长的输出）；PlanAgent 反其道而行，让 MLLM 只输出 5 个 IDM 参数（跟车道中心线 c、跟随目标 la、限速 v0、最大加速度 acc、最大减速度 dec），轨迹由确定性的 IDM 规划器算出来。这是一个关键的设计哲学差异：**把不确定性关进笼子里，而不是祈祷 LLM 不犯错**。

## 三个模块：环境翻译、推理引擎、反思校验

**1. Environment Transformation（把场景翻译成 MLLM 能读的两种语言）。** 场景信息被分成全局和局部。全局是一张 BEV 图：8 种元素用不同颜色画——车道中心线、人行横道、车道、自车、其他车辆、自行车、行人、静态障碍物；自车位置画一个白方块，每辆车带一个红色箭头，箭头方向是运动方向、长度是速度。局部是一段基于 lane-graph 的文本描述：把车道中心线切成等长小段当“节点”，每个节点有 8 种相邻关系（前、左前、左、左后、后、右后、右、右前），自车所在的节点染红，周围 8 个方向的节点按颜色编号；文本里写“你在红色节点上，前方黄色节点有一辆车 004，距离 11.61 米，速度 2.02 m/s”这类话。

为什么搞两套？论文的论点是：纯文本坐标数字又冗余又不对 LLM 的语义空间（LLM 不擅长读一堆浮点数），BEV 图给全局语义（“这是个环岛”），lane-graph 文本给局部精确关系（“左前方那辆车离我多远”）。Token 效率的对比数据：描述一个场景平均用 141 个 token，GPT-Driver 要 449，LLM-ASSIST 要 426——**三分之一**。这个数字是实测的（Table III），是论文最硬的工程贡献之一。

**2. Reasoning Engine（分层 CoT：场景理解 → 运动指令 → 参数选择）。** 系统提示词分三块：任务定义（告诉模型你是规则规划器的大脑、输出格式是什么、给一个示例）、常识（别撞车、尽量往前开、别逆行）、CoT 引导（按步骤来）。推理分三层：

- 场景理解：问答形式，“这是什么场景？（多车道环岛）”“自车状态？（最右车道、在动）”“红绿灯？（绿灯）”“同车道前方有车吗？（有）”。
- 运动指令：纵向（加速/减速/保持）+ 横向（左变道/右变道/保持）+ 理由。
- 规划器生成：输出 `Generate_IDM_Planner(la=vehicle_001, c=change_left, v0=9.0, acc=1.5, dec=3.0)`。

IDM 本体是经典跟驰模型，纵向加速度公式就是论文里的式 (1)(2)：

$$a = acc\left[1 - \left(\frac{v}{v_0}\right)^\delta - \left(\frac{s^*}{s}\right)^2\right], \quad \frac{dv}{dt} = \min(acc, \max(a, dec))$$

其中 $v_0$ 是期望速度（MLLM 给的“限速”参数，注意论文特别提示模型：v0 是上限不是目标），$s$ 是与前车距离，$s^*$ 是期望跟驰距离。MLLM 实际只定 5 个数，物理方程保证输出的轨迹满足车辆运动学——**LLM 再幻觉，也幻觉不出违反物理的轨迹**，因为轨迹不是它算的。

**3. Reflection（仿真打分，不行就重想）。** 每个 MLLM 选出的参数组合，先在 PDM-Closed 的短时仿真里跑一遍，算一个 driving score $s$（碰撞、舒适度等）。如果 $s < \lambda$（$\lambda = 0.75$），打回 Reasoning Engine 重想；最多重想 $max_{exec} = 3$ 次，超了就用最后一次的结果。注意这是**短时仿真**，不是 PDM 那种长时 rollout——作者的理由是短时仿真里其他车辆用匀速模型预测，累积误差小，打分更准。

## 实验：在 nuPlan 闭环上，规则方法和学习方法第一次被同一个 MLLM 系统同时超过

评测在 nuPlan Val14（常规场景）和 Test14-hard（PlanTF 定义的长尾场景集）上，指标是官方的 NR-CLS（非反应式闭环分数）和 R-CLS（反应式闭环分数），0–100 分。

主结果（Table II）：Val14 上 PlanAgent NR-CLS 93.26 / R-CLS 92.75，超过 PDM-Closed（92.51/91.79）、DTPP（89.64/89.78）、PlanTF（84.83/76.78）；Test14-hard 上 NR-CLS 72.51（PlanTF 72.68 略高）、R-CLS 76.82（全场最高）。论文的解读是：规则方法（PDM-Closed）在常规场景强、长尾弱；学习方法（PlanTF）在长尾上强、常规场景掉；PlanAgent 两边都好——**MLLM 的常识泛化补了规则方法的短板，又没有学习方法的过拟合**。

消融（Table IV/V，Test14-hard）：
- 去掉 BEV 图：NR-CLS 72.51→71.23；去掉 lane-graph 文本：→70.97。两个模态各贡献约 1.5 分，缺一不可。
- 去掉场景理解：-2.42；去掉运动指令：-1.91；去掉 Reflection：-2.67。**Reflection 是掉分最多的**——MLLM 裸输出的参数确实需要仿真校验兜底。

还有两个值得注意的实验：
- **MLLM 调用间隔**（Table VI）：每 1 秒调一次 71.85，每 2 秒 72.51（最好），每 3 秒 71.35。调太勤，MLLM 的噪声决策反而坏事；调太疏，跟不上场景变化。2 秒是 nuPlan 10Hz 仿真下每 20 个 planning iteration 调一次。
- **换开源 MLLM**（Table VII）：LLaVA-13B 69.40、CogVLM-chat 68.37、LLaVA-7B 67.73，都明显超过 PDM-Closed 的 65.07（Test14-hard NR-CLS）。说明这套架构不绑定 GPT-4V，开源模型也能 work——但注意 GPT-4V 单次推理要 5.6 秒，LLaVA-7B 也要 2.9 秒，**实时性是所有配置的共同短板**，论文在 limitations 里承认了。

## 怀疑式阅读：代码说出来的和论文说的不一样

这是本篇 x-ray 最重要的部分，来自对官方代码的实际阅读。

**第一，“生成规划器代码”在实现层面是字符串匹配选参数，不是生成代码。** 论文 3.3 节和 Fig. 6 展示的是 `Generate_IDM_Planner(la=vehicle_001, c=change_left, v0=9.0, acc=1.5, dec=3.0)` 这种漂亮的函数调用。但 `code/tuplan_garage/planning/simulation/planner/pdm_planner/GPT_utils_rest.py` 里的 `generate_gpt_decision()` 实际做的是：拿到 MLLM 的自由文本回复，做 `casefold()` 后**关键词匹配**——找“lateral”段落里有没有 "left"/"right"（定 lane_mode），找“speed”段落里有没有 "3.0"/"9.0"/"15.0"（定 speed_mode 三档），找“acceleration”段落里有没有 "1.5"/"2.5"/"3.5"（三档），找“deceleration”段落里有没有 "1.0"/"2.0"/"3.0"（三档）。然后在 `abstract_pdm_closed_planner_rest.py` 第 649 行，用 `gpt_choice_index = speed*9 + accel*3 + decel` 生成一个 mask，**从 PDM 原有的 27 种 IDM 参数组合 proposal 里选一个**。

什么意思？MLLM 的真实输出空间不是连续的 $(v_0, acc, dec)$，而是 $3 \times 3 \times 3 = 27$ 个离散档位；它也不是在“写代码”，而是在做 27 选 1 的选择题，判卷方式是字符串包含匹配。这比论文描述的机制**更弱、但更鲁棒**——弱在表达能力被锁死在 27 档里，鲁棒在 LLM 无论怎么胡说八道（只要包含几个关键词），最终执行的都是预设的安全参数组合。这是一个“论文讲故事、代码讲保险”的典型例子。我的判断（推断，依据是代码）：27 档离散化很可能是为了让 Reflection 的仿真校验可行——连续参数空间没法穷举仿真，27 个固定组合每个都预先可验证。

**第二，代码里调的是 gpt-4o，不是论文写的 GPT-4V。** `generate_gpt_decision` 里 `model="gpt-4o"`。论文实验用的是 GPT-4V（Table VII 明确列了 GPT-4V 5568ms）。代码是论文之后更新的（README 说“we update the latest version of our code”），模型换代了，但论文里的数字是 GPT-4V 跑出来的。引用性能数字时要注明模型版本。

**第三，Table I 里对竞品的标注有水分。** 表格把 Agent-Driver 的闭环评估标成“开环”（准确，Agent-Driver 闭环只在 CARLA Town05-Short 上做了很小规模的实验，论文自己都放在次要位置），把 DiLu 的场景标成 Highway-env（准确）。但把 LLM-ASSIST 的 CoT 输出标成“Waypoints or Parameters”而 PlanAgent 标成“Hierarchical CoT”——LLM-ASSIST 本质上也是给 PDM 选超参数，和 PlanAgent 的 27 选 1 是同一类事情，表格的措辞制造了差异大于实际的印象。

**第四，$\lambda = 0.75$ 和 $max_{exec} = 3$ 没有敏感性分析。** Reflection 的阈值是拍脑袋定的（论文只说“超参数”，没给搜索过程）。阈值设高了会频繁重想（每次重想都是一次 GPT-4V 调用，5.6 秒），设低了校验就形同虚设。这个 trade-off 论文没展开。

**第五，比较基线里 LLM-ASSIST 的数字是引用的**（论文自己脚注承认了：not open-source，只报公开结果），而且 LLM-ASSIST 只有 Val14 的数，没有 Test14-hard 的数——Table II 里那一行的横杠要看清楚。

公平地说：这篇论文漂亮的地方在于**诚实面对了 MLLM 的不确定性**，并且用三层机制（离散参数空间 + 短时仿真校验 + 重试上限）把它关起来。消融表证明每一层都在干活。这比那些让 LLM 直接输出轨迹、然后靠 prompt 祈祷它别撞墙的工作，在工程上成熟得多。

## 这篇论文的时间坐标和血缘

PlanAgent（2024-06）和 LLM-ASSIST（2024-01，arXiv:2401.00125）是“MLLM 调 PDM 参数”这条线的两篇代表作。LLM-ASSIST 在先，做法是让 LLM 在 PDM 的 proposal 里选一个更好的（两阶段：uncertainty 检测 + 参数选择）；PlanAgent 在后，加了 BEV+lane-graph 的环境翻译、分层 CoT、Reflection 校验，评测也从 Val14 扩展到 Test14-hard。VLM-MPC（2024-08）是同一思想在纵向控制 + MPC 上的对应物。三篇合在一起，说明 2024 年上半年领域已经形成共识：**MLLM 的正确位置是参数调度器，不是轨迹生成器**。

和 Agent-Driver 的关系：PlanAgent 的 Table I 明确把 Agent-Driver 列为对比对象，批评点是“开环 + 直接输出 waypoint 有安全风险”。这个批评是成立的——Agent-Driver 的自反思（self-reflection）虽然也有碰撞检查优化，但它的 motion planning 本体还是微调 LLM 直接生成轨迹，Reflection 只是事后修补。PlanAgent 的架构从根本上不让 LLM 碰轨迹。

## 与 RAMP 的血缘/划界

**血缘**：PlanAgent 是 RAMP 调度器模块最直接的先例之一，亲缘关系甚至比 Agent-Driver 更近。RAMP 的“LLM 在线多规划器调度”就是 PlanAgent “MLLM 选 IDM 参数”的推广：调度的对象从 27 组 IDM 参数变成多个异构规划器，调度的输入从 BEV+文本变成世界模型 rollout + 情景记忆，调度的校验从短时仿真打分变成闭环风险校验。PlanAgent 证明了这条路在 nuPlan 闭环上走得通，这是 RAMP 立项的实证底气之一。

**划界**：三处关键差异，写 related work 时可以直接用。第一，PlanAgent 的调度空间是**离散 27 档**（代码实锤），RAMP 调度的是异构规划器——连续参数和离散规划器选择的混合空间，调度粒度完全不同。第二，PlanAgent 的 Reflection 是**开环短时仿真**（前车匀速假设），RAMP 的风险校验是**世界模型 rollout**（带交互预测），校验的信息量不在一个量级。第三，PlanAgent 没有记忆——同样的场景每次都要重新问一遍 MLLM（2 秒一次，5.6 秒推理），RAMP 的情景记忆正是为了解决这种重复计算和经验复用问题。一句话：PlanAgent 回答了“MLLM 能不能做闭环调度”，RAMP 要回答“调度能不能又快、又有记忆、又安全”。
