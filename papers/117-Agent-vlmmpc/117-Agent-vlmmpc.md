# VLM-MPC：VLM 只管“调参”，MPC 只管“开车”——异步双层是怎么咬合的

> **信息块**
> - 论文标题：VLM-MPC: Model Predictive Controller Augmented Vision Language Model for Autonomous Driving（arXiv 预印本标题曾为 "Vision Language Foundation Model (VLM)-Guided Model Predictive Controller (MPC) for Autonomous Driving"）
> - 作者：Keke Long, Haotian Shi, Jiaxi Liu, Chaowei Xiao, Xiaopeng Li*（通讯）—— 威斯康星大学麦迪逊分校土木与环境工程系
> - venue：**Transportation Research Part C: Emerging Technologies, vol. 183, 2026, Article 105487**（DOI: 10.1016/j.trc.2025.105487）。arXiv 预印本 arXiv:2408.04821（2024-08-09 发布，v2 为最新版）。注意：TR-C 是交通领域的期刊，不是 AI 会议；2026 年卷期意味着从投稿到见刊走了一年多。
> - 链接：https://arxiv.org/abs/2408.04821
> - 代码：官方 https://github.com/keke-long/vlm-mpc（本目录 `code/`，已 clone 核对）
> - PDF：本目录 `2408.04821.pdf`（下载的是最新版 v2）
> - 一句话核心洞见：**VLM 的输出不是轨迹、不是动作，而是一组 MPC 的代价权重和期望值** $[N, Q, R, Q^h, v^d, h^d]$**；VLM 每几秒调一次参，MPC 每 0.1 秒解一次优化——“编排”在这里有精确的数学含义：上层定优化问题的参数，下层解优化问题。**

---

## 这篇论文到底在回答什么问题

引言里作者把 FM（foundation model）进驾驶的现有工作分成两类，这个分类是全文的纲：

- **FM to action**：VLM 直接输出动作——waypoint、油门、方向盘（GPT-Driver、DriveGPT4、LMDrive 都属此类）。问题有两个：VLM 不懂车辆运动学，输出的轨迹可能根本开不出来；VLM 太慢，近碰撞场景需要约 10Hz 的控制频率，VLM 给不了。
- **FM to commands/codes**：VLM 只做上层决策，给下层控制器发指令或参数（DiLu 的 5 个离散动作、LLM-ASSIST 选 PDM 参数、PlanAgent 调 IDM 参数、languageMPC）。

作者要回答的问题是：**在第二类里，如果把下层换成 MPC（而不是 IDM 或 PID），VLM 的决策质量会不会变好？** 这不是一个显而易见的问题——MPC 比 IDM 复杂得多（要解带约束的优化），VLM 给的参数稍微不靠谱，MPC 可能直接无解或解出危险动作。整篇论文就是围绕“VLM 调 MPC 参数”这个接口设计实验。

## 接口的精确形态：VLM 输出 6 个数，MPC 拿去组装优化问题

这是全文技术上最值得细读的部分，因为“编排”在这里不是比喻，是数学。

**上层 VLM 的输出** $\theta_t = [N, Q, R, Q^h, v^d, h^d]$，6 个数：

| 符号 | 含义 | 大白话 |
|---|---|---|
| $N$ | 预测时域 | MPC 往前看多少步 |
| $Q$ | 速度保持权重 | 多在乎“开到期望速度”（固定为 1） |
| $R$ | 控制 effort 权重 | 多在乎“别猛踩油门”（舒适性） |
| $Q^h$ | 跟驰距离权重 | 多在乎“跟前车保持距离”（安全性） |
| $v^d$ | 期望速度 (m/s) | 想开多快 |
| $h^d$ | 期望跟驰时距 (s) | 跟前车隔几秒 |

**下层 MPC 解的优化问题**（式 15，有前车/停止线时）：

$$J = \sum_{k=0}^{N-1}\|\Delta v_{t_0+k\Delta t^l}\|_Q^2 + \sum_{k=0}^{N-1}\|u_{t_0+k\Delta t^l}\|_R^2 + \sum_{k=0}^{N-1}\|\Delta x_{t_0+k\Delta t^l}\|_{Q^h}^2 + \epsilon^T\rho\epsilon$$

三项分别罚：速度偏离期望、控制量过大、跟驰距离偏离期望，外加松弛变量 $\epsilon$ 保证约束（速度上下限、控制量上下限，式 17-18）软化后仍可解。车辆纵向动力学带发动机时滞（式 9-13）：$\dot a_t = (u_t - a_t)/\tau^A$，$u_t$ 是 MPC 决定的期望加速度，$\tau^A$ 是发动机响应时滞——反馈线性化后，MPC 只需要决定 $u_t$。

**“编排”的精确含义**：VLM 定的是**优化问题的参数**（权重和期望值），MPC 定的是**优化问题的解**（每一时刻的加速度 $u_t$）。VLM 的幻觉被限制在参数空间里——它最多把 $v^d$ 定得离谱，但 MPC 的约束（$v_{min} \le v \le v_{max}$）和动力学模型还在，车不会做出物理上不可能的动作。这是和“VLM 直接输出轨迹”最本质的区别：**优化问题本身是一个安全壳**。

**异步**：VLM 每 $\Delta t^u$（秒级，Llava 实测平均 3.42 秒）更新一次 $\theta$，MPC 每 $\Delta t^l = 0.1$ 秒解一次优化。在两次 VLM 更新之间，MPC 沿用上一组参数接着开。论文的论点是：驾驶参数（期望速度、跟驰时距）本来就是慢变量，不需要 10Hz 更新；快变量（加速度）由 MPC 以 10Hz 处理。这个快慢分离是整套架构成立的前提。

## VLM 的输入是怎么拼出来的：四个部件

1. **Environment encoder（环境编码器）**：前视图像过 CLIP ViT-B/32，图像特征和预设的文本描述做余弦相似度，输出 6 类环境描述 $E_t = \{E_t^{Wea}, E_t^{Lig}, E_t^{RT}, E_t^{RC}, E_t^{Obs}\}$（天气、光照、道路类型、路况、障碍物）。天气/光照/道路类型**不管置信度直接输出**，路况和障碍物要超过阈值（0.3/0.2）才输出——防止低置信度的描述污染 prompt。
2. **Scene encoder（场景编码器）**：从 nuScenes 数据集里抽自车历史状态（$\gamma=5$ 步、间隔 0.5 秒的位置/速度/加速度）、前车状态（相对距离 $\Delta x$、速度、加速度）、停止线位置。注意：**感知是假设完美的**，直接用数据集真值，论文明确说“不考虑感知阶段误差”。
3. **Reference memory（参考记忆）**：这是论文里最有意思的一个部件。作者把 nuScenes 的场景按三个二元特征（雨/夜/路口）分成 8 组，对每组**标定**出一套 MPC 参数（怎么标定的：对每个场景做参数辨识，让 MPC 复现人类轨迹，见 `code/calibrate_MPC/`）。然后做统计显著性检验（p 值热图 Fig. 4）：如果两组在某个参数上差异显著，就分开存；不显著就合并。最后得到 Table 2 的 8 组参数，比如晴天白天非路口：$[N=9, Q=1, R=1.68, Q^h=2.75, v^d=6.44, h^d=2.60]$；雨夜路口：$[9, 1, 1.68, 1.99, 5.09, 2.55]$。VLM 第一次被调用时，prompt 里带上当前场景对应的这组平均参数当“参考答案”。
4. **Prompt generator**：把上面三样 + 前视图像拼成 prompt，末尾加 CoT 引导（5 步：理解环境 → 定预测时域 → 定代价权重 → 定期望速度 → 定期望跟驰时距）。首次调用 $ \theta_{t=0} = f^{VLM}(M(E_{t=0}), img_{t=0}, E_{t=0}, S_{t=0})$；后续调用把上一轮的 $\theta_{t-\Delta t^u}$ 也喂回去，让 VLM 做增量调整而不是从头定。

## 代码核对：VLM 的文本输出是怎么变成数字的

这是任务点名要讲清的关键。`code/controller_llm/fun_get_parameter_from_response.py` 里的 `extract_mpc_parameters()`：

```python
pattern = r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[^,\]]*\s*,\s*(\d+\.?\d*)\s*[^,\]]*\s*,\s*(\d+\.?\d*)\s*[^,\]]*\s*\]"
match = re.search(pattern, response_text)
```

VLM 被要求在回复末尾输出形如 `[10, 1.0, 2.0, 3.5, 6.5, 2.8]` 的 6 元组（可带单位如 `m/s`、`s`，正则里用 `[^,\]]*` 跳过单位文字），代码用正则从全文里**搜第一个匹配的方括号**，转成 6 个 float，存 CSV，喂给 MPC。注意这个正则的脆弱性：如果 VLM 在 CoT 中间举了个例子也写了方括号数字，会被误抓——代码里注释掉的老版本还处理过“带单位”和“不带单位”两种格式，说明作者实际被格式问题坑过。`get_parameter_from_response` 里如果没匹配到，直接打印 `Can NOT get parameters`——**没有重试、没有 fallback**，这一帧的 MPC 就沿用上一组参数（靠异步设计兜底）。

而 MPC 侧（`code/controller_MPC/MPC_fun.py`，cvxpy 实现）与论文式 (15)(16) 完全对应：

```python
def mpc_fun(v_ego, lead_info, N, dt, Q, R, Q_h, tau, desired_speed, desired_headway):
    ...
    cost = sum([Q * (v[i] - desired_speed) ** 2 + R * u[i] ** 2 + ... for i in range(N)])
    ...
    cost += Q_h * (expected_distance)**2   # 有前车时才加这一项
```

VLM 给的 6 个数原样进入二次代价函数——这就是“编排”的代码形态：上层只碰参数，下层只解优化。另有一个论文没写的细节：代码里硬编码了 `min_headway = 1`（最小跟驰时距 1 秒）作为安全约束，VLM 再怎么调参也突破不了这条底线——等于在 MPC 层埋了一个 VLM 不知道的“隐形安全盾”。

另外 `fun_generate_prompt.py` 里硬编码了 `parameters_info`——三组特征（rain/intersection/parking_lot）各自对 $Q^h$ 和 $v^d$ 的 p 值和均值（rain 对 $Q^h$：p=0.01889 显著；intersection 对 $v^d$：p=0.03214 显著），和论文 Fig. 4 / Table 2 对应。标定脚本在 `code/calibrate_MPC/` 下（含 `mpc_for_one_scene.py` 等），说明 Table 2 的数字是真实标定跑出来的，不是编的。

## 实验：在 303 个 nuScenes 场景上，VLM-MPC vs 三个基线

评测场景：nuScenes 中 303 个场景（波士顿右行 + 新加坡夜间），按雨/夜/路口分成 8 组。三个指标：

- **安全性 PET**（post encroachment time，两车占用同一空间的时间间隔，越大越安全，1 秒是临界阈值）
- **舒适性 RMS-a**（加速度均方根，越小越平顺）
- **稳定性 completion rate**（VLM 输出有效参数的比例）

基线：纯 MPC（固定平均参数）、LLM-to-Action（GPT-3.5 直接生成动作，复现 Agent-Driver 范式）、LLM-MPC（LLM 调参但无图像输入、无 reference memory）。

主结果：
- **PET**（Table 4）：VLM-MPC 在 8 组场景里全部高于 1 秒阈值（1.03–3.53）；LLM-to-Action 有多组跌破阈值（0.45、0.75、0.64），Fig. 6 展示了两个具体危险案例——白天差点追尾 parked cars 旁的工作区、夜间加速冲向路口停住的车。纯 MPC 的 PET 在 1.1–1.9 之间，安全但保守。
- **RMS-a**（Table 5）：VLM-MPC 0.33–0.43，**比人类真实轨迹（0.51–0.68）还平顺**；LLM-to-Action 高达 0.43–3.13——直接生成动作的 VLM 输出了 -23.2 m/s² 的极端减速度（人类轨迹最极端是 -10.53，作者注是定位误差导致的），说明它完全没学到车辆动力学约束。
- **Completion rate**（Table 6）：Llava-1.6（7B）99.7%，GPT-4o/GPT-4-mini/GPT-3.5 100%。作者把这归因于参数量，但这个归因是草率的（见怀疑式阅读）。
- **响应时间**（Fig. 8/9）：Llava-1.6 平均 3.42 秒（满足系统 0.2Hz 的要求——注意这个“要求”是作者自己定的），GPT-4o/mini 平均约 10 秒。Llama-3.1（纯文本 LLM-MPC）1.5 秒。

消融（Table 7/8）：
- **去 reference memory（-M）**：completion 99.7%→99.3%（多了 2 次无效输出），但有意思的是**安全性和平顺性基本没掉**——个别场景 PET 反而更高（场景 7：3.53→14.95）。作者的解释是：没了参考参数，VLM 倾向于定更高的速度和更大的跟驰距离，开得更“怂”反而更安全，但雨夜场景平顺性变差。我的解读：reference memory 的主要价值是**输出稳定性**（减少无效回复），而不是性能上限。
- **去 environment encoder（-E）**：completion 暴跌到 **87.5%**，12.5% 的场景 VLM 根本不知道自己在干嘛（不输出参数或格式错误），还有个别响应超过 200 秒。**环境描述是 VLM 理解任务的前提**，不是锦上添花。这个消融是全文最有说服力的一个数字。

可解释性分析（3.3.3）：用 VLM-Visualizer 看 Llava 生成 token 时对图像的注意力——前 50–60 个 token（CoT 第一步“理解环境”）约 14% 的注意力在视觉 token 上，后面定参数的步骤主要依赖文本。这个分布和 CoT 的设计意图一致，算是“模型确实按我们设计的步骤在走”的旁证。

## 怀疑式阅读

**第一，评测是“闭环”吗？** 论文通篇说 closed-loop，但实验是在 nuScenes 数据集上做的——nuScenes 是离线数据集，没有仿真器。所谓的闭环是：用数据集的真值场景初始化，MPC 按 VLM 参数开，其他车辆按 log 回放（non-reactive）。这和 CARLA/nuPlan 那种真闭环（自车动作影响他车）不是一回事。他车不反应，PET 指标会系统性偏乐观——因为真实世界里你别前车，前车会刹车或变道，这里的前车只会按 log 走。论文没有讨论这个局限。

**第二，reference memory 有数据泄露嫌疑。** 8 组场景的平均参数是从 nuScenes 标定出来的，评测场景也是 nuScenes 的 303 个场景。论文没有明确说明标定集和评测集是否 disjoint。如果标定用的是同一批场景，那么“纯 MPC 基线（固定平均参数）”和 VLM-MPC 的 reference memory 都在**用测试集的统计量**——基线强不奇怪，VLM-MPC 的提升里有多少来自“见过测试集的参数分布”，说不清。这是一个应该在审稿时被问到的问题，论文没提。

**第三，completion rate 归因草率。** “GPT 系列 100% 是因为参数量大”——但 GPT 是云端 API（temperature=0 以外的参数全默认），Llava 是本地部署；prompt 对长文本的遵循能力、指令微调的质量都可能是原因。更重要的是，99.7% vs 100% 在 303 个场景上只差 1 个样本，这个差异没有任何统计显著性。

**第四，Table 2 的参数合并逻辑。** 8 组场景里大量参数是相同的（比如 N 全是 9，Q 全是 1），真正有差异的只有 R、Qh、vd、hd 中的几个。8 组这个数字是 $2^3$ 凑出来的，p 值检验只是事后合理化。实际起作用的可能是 3–4 组有效参数。

**第五，纵向 only。** 全文只做纵向控制（跟驰、过停止线），没有横向（变道、转弯）。MPC 的 cost 里没有横向项，VLM 也不输出任何横向参数。标题里的 autonomous driving，实际是 car-following。这个范围限制在方法节是诚实的（2.3 开头明确说 focus on longitudinal），但摘要和标题没有限定。

公平地说：这篇论文的**接口设计**（VLM 定参数、MPC 解优化、异步快慢分离）是扎实的，代码和论文对得上（正则解析、标定脚本、p 值硬编码都能在 repo 里找到），消融 -E 的 87.5% 是一个诚实的负面结果。它的局限主要在评测的闭环成色和数据划分的透明度。

## 这篇论文的时间坐标

arXiv 2024-08-09，v1 标题还是 "Vision Language Foundation Model (VLM)-Guided Model Predictive Controller (MPC)"，v2 改成现在的 "Model Predictive Controller Augmented Vision Language Model"。TR-C 2026 年 183 卷见刊（article 105487）。从投稿到见刊一年多，符合 TR-C 这种交通期刊的审稿周期。

它的直接前辈是 languageMPC（Sha et al., arXiv:2310.03026，IdSim 开环，LLM 输出 MPC 参数）——VLM-MPC 的增量是：(1) 从纯文本 LLM 换成 VLM（加了图像输入）；(2) 从开环 IdSim 换成 nuScenes 数据集上的“准闭环”；(3) 加了 reference memory 和 environment encoder 两个稳定器；(4) 系统地对比了 FM-to-action vs FM-to-parameters。Table 1 把相关工作按输出类型（action vs commands/codes）和开/闭环列出来，是 related work 写作的一个好模板。

## 与 RAMP 的血缘/划界

**血缘**：这是 5 篇里和 RAMP 的“编排”二字咬合最紧的一篇。RAMP 的调度器本质上也是“上层定参数、下层执行”——只不过调度的对象从 MPC 的 6 个代价参数变成多个异构规划器，调度的依据从 VLM 的场景理解变成世界模型 rollout + 情景记忆。VLM-MPC 证明了三件事 RAMP 可以直接继承：(1) **参数级接口比动作级接口安全**——优化问题/规划器本身就是安全壳，LLM 的幻觉被关在参数空间里；(2) **快慢分离是解时延的正道**——VLM 秒级调参 + MPC 10Hz 执行，RAMP 的异步流水线是同一思想在多规划器上的展开；(3) **参考记忆能稳定 LLM 输出**——reference memory 让 completion 从 87.5% 回到 99.7%，RAMP 的情景记忆有同样的稳定器作用，只是记忆的内容从“平均参数”变成“风险加权的情景”。

**划界**：三条。第一，VLM-MPC 的 VLM 是**开环调参**——参数定完就不管了，没有对参数效果的闭环校验；RAMP 的风险校验是每帧对调度结果做世界模型 rollout 验证，是**闭环**的。第二，VLM-MPC 只有一个执行器（MPC），没有“多规划器选择”问题；RAMP 的核心恰恰是多个规划器之间的在线仲裁。第三，VLM-MPC 的参考记忆是**静态**的（离线标定、按场景查表），RAMP 的情景记忆是**在线更新、风险加权、自适应衰减**的。写 related work 时，VLM-MPC 适合放在“上层模型调下层控制器参数”这一支的最新代表，然后转折：静态查表记忆 + 单执行器 + 无闭环校验，正是 RAMP 要超越的三点。
