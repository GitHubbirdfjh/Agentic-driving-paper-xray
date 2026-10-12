# Bench2Drive：给端到端自动驾驶的"分科考试"

> 一句话：Bench2Drive 把闭环评测从"跑一条 10 公里的长路线看总分"改成"220 条 150 米的短路线、按 5 大技能分科打分"，并配了一套官方统一训练集——解决的是长路线方差爆炸和各家自采数据不公平这两个老毛病。这是 RAMP 的目标 benchmark，本篇把评估协议、指标口径和坑全部拆细。

## 基本信息

| 项 | 内容 |
|---|---|
| 标题 | Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving |
| 作者 | Jia Xiaosong*、Yang Zhenjie*、Li Qifeng*、Zhang Zhiyuan*、Yan Junchi（通讯），上海交大 ReThinkLab（ School of AI / CSE） |
| arXiv | 2406.03877（v1，2026-10-11 已读全文 1310 行，含附录 A–H） |
| 链接 | https://arxiv.org/html/2406.03877v1（已验证可访问） |
| 代码/数据 | GitHub: Thinklab-SJTU/Bench2Drive；HuggingFace: rethinlab/Bench2Drive；Apache 2.0 |
| 专家模型 | Think2Drive（arXiv:2402.16720，同组 RL 世界模型工作），用于采集官方训练数据并提供 value/特征蒸馏 |
| 信源评级 | **A 级（同行评议论文，全文精读）**——公式、惩罚表、基线数字全部来自原文，附录矛盾处已标注 |

## 它要回答的三个问题（动机）

论文开篇把现有评测的毛病数了三遍，每一遍都对应 Bench2Drive 的一个设计：

1. **开环评测（nuScenes L2 误差）不行**。社区已公认：distribution shift（开环训、闭环跑，分布变了）、causal confusion（模型学的是"自车历史状态→未来轨迹"的捷径，不是真在看路）、nuScenes 验证集 75% 是直行——AD-MLP 只用自车状态、不看传感器，L2 也能刷得很好。结论：L2 只能验证"模型收敛了没"，不能比谁开得好。
2. **闭环长路线方差太大**。Town05Long、Longest6、Leaderboard v2 的路线都是几公里起，Driving Score 用指数连乘惩罚违章。论文算了一笔账：三条都完成 90% 的路线，闯红灯 0/1/2 次，得分 90 / 90×0.7=63 / 90×0.7×0.7=44.1，**标准差 18.9**——一次偶然违章就天翻地覆，方法之间没法比。LB v2 榜上方法普遍 <10/100 分，低到失去区分度。
3. **各家自采数据，算法级不公平**。闭环方法都要自己用 CARLA 采数据训，数据集大小和分布直接影响成绩，比的其实是"数据工程"不是算法。

Bench2Drive 的回答：**官方统一训练集**（200 万帧、10000 clips）+ **220 条短路线分技能评估**。

## 方法论拆解（一）：官方训练集

- **规模**：200 万全标注帧，10000 个短 clips，覆盖 44 个交互场景 × 23 种天气 × 12 个城镇，CARLA v2 采集。
- **采集方式**：用专家模型 Think2Drive 跑预设路线文件，**只保留无违章的**；设计遍历算法保证场景×城镇覆盖，人工检查天气/城镇/场景均衡，人工过滤 CARLA 渲染 bug 的坏 clips。
- **传感器配置**（对齐 nuScenes，方便社区复用模型）：1×64 线 LiDAR（85 米、60 万点/秒）、6×相机（环视、900×1600、JPEG quality 20）、5×雷达（100 米、30° 水平/垂直视场）、IMU+GNSS（位置/偏航/速度/加速度/角速度）、1×BEV 相机（调试可视化用）、HD-Map（车道/中心线/拓扑/动态灯态/灯和 stop sign 的 trigger 区）。
- **标注**：3D 包围盒、深度、语义/实例分割、HD-Map，外加 **Think2Drive 的 value 估计和特征**——专门给蒸馏用的（DriveAdapter/ThinkTwice 类方法吃这个）。
- **三档**：mini（10 clips，调试可视化）、base（1000 clips，对标 nuScenes 量级，8×RTX3090 可训）、full（10000 clips，大规模研究）。基线实验用的是 base（950 训 / 50 验，验证集保证 44 场景每种至少 1 个、天气均衡）。
- **压缩细节**（附录 A，复现时要注意）：图像 JPG quality=20，**推理时也要在内存里做 JPG 压缩解压**，避免 train-val gap；语义分割/深度存 PNG；点云用 laszip；JSON 用 GZIP。
- **CARLA API 的 bug 清单**（附录 A，RAMP 跑实验备查）：① 所有行人速度 API 返回 0，要自己差分算；② 速度计/IMU 偶发返回 None，填 0；③ 部分 stop sign 在地面上没有包围盒，用矩形记录其 trigger volume 补；④ 部分静止车辆位姿 API 给错，用 center+extent 重算 3D 框。⑤ 坐标系是 UE 的 Z-up 左手系，和 nuScenes 的 Y-down 右手系不一样，转的时候注意（compass yaw = 标准 yaw + π/2 的关系，偶发 NaN 要过滤）。

## 方法论拆解（二）：220 条短路线 + 5 大技能（评估协议）

这是全文的核心设计：

- **44 场景 × 5 条路线 = 220 条**，每条约 150 米、只含**单个**场景，5 条之间换天气和城镇。
- **5 大技能**（表 2，把 44 场景归纳成 5 类能力）：
  - **Merging（汇入）**：16 个场景——CrossingBicycleFlow、EnterActorFlow、HighwayExit、InterurbanActorFlow、HighwayCutIn、InterurbanAdvancedActorFlow、MergeIntoSlowTraffic(V2)、NonSignalizedJunctionLeftTurn/RightTurn(+EnterFlow)、ParkingExit、LaneChange、SignalizedJunctionLeftTurn/RightTurn(+EnterFlow)
  - **Overtaking（超车/绕行）**：9 个——Accident(TwoWays)、ConstructionObstacle(TwoWays)、HazardAtSideLane(TwoWays)、ParkedObstacle(TwoWays)、VehicleOpenDoorTwoWays
  - **Emergency Brake（紧急制动）**：12 个——BlockedIntersection、DynamicObjectCrossing、HardBreakRoute、OppositeVehicleTakingPriority、OppositeVehicleRunningRedLight、ParkingCutIn、PedestrianCrossing、ParkingCrossingPedestrian、StaticCutIn、VehicleTurningRoute(+Pedestrian)、ControlLoss
  - **Give Way（让行）**：2 个——InvadingTurn、YieldToEmergencyVehicle
  - **Traffic Sign（交通标志）**：18 个——各种有/无信号灯路口转弯、TJunction、Vanilla 系列、stop sign 等
- **为什么用短路线**（论文 3.2 的论证）：① 单条路线只测一个技能，能力解耦，谁强谁弱一目了然；② 短路线里违章次数天然少，指数惩罚的方差被"物理截断"了——长路线跑 10 公里可能闯 5 个红灯，150 米撑死 1~2 个。

## 方法论拆解（三）：指标定义（逐字抠原文）

两个指标，公式是论文 Equation (1)：

**Success Rate（SR，成功率）**：
$$\text{Success Rate}=\frac{n_{\text{success}}}{n_{\text{total}}}$$
一条路线算成功：**限时内到达终点、且零违章**。SR 就是成功路线数 / 总路线数。——这是最硬的指标，二值，非黑即白。

**Driving Score（DS，驾驶分）**：
$$\text{Driving Score}=\frac{1}{n_{\text{total}}}\sum_{i=1}^{n_{\text{total}}}\text{Route-Completion}_{i}\times\prod_{j=1}^{n_{i,\text{penalty}}}p_{i,j}$$
Route-Completion_i 是第 i 条路线的完成百分比；p_{i,j} 是第 i 条路线上第 j 个违章的惩罚系数（连乘）。分技能报告时，分母是该技能组内的路线数。

**惩罚系数表**（Appendix E，**沿用 Leaderboard v2 口径**，注意这是 2.0 指数版、不是 2.1 线性版）：

| 违章 | 系数 | 备注 |
|---|---|---|
| 撞行人 | 0.50 | 每次都罚 |
| 撞车 | 0.60 | 每次都罚 |
| 撞其他 | 0.65 | 每次都罚 |
| 闯红灯 | 0.70 | 每次都罚 |
| 场景超时 | 0.70 | 4 分钟没通过某场景 |
| 太慢 | 0.70 | 跟不上周围车流速度 |
| 不给紧急车辆让行 | 0.70 | |
| Off-road | — | 不计入完成度（白开） |
| 偏离路线 >30m | — | 直接 shutdown |
| 180 秒无动作 | — | 直接 shutdown |
| 路线超时 | — | 直接 shutdown |

**口径细节**（RAMP 写论文时必须写对的）：
- DS 是"完成度×惩罚连乘"再平均，不是"平均完成度×平均惩罚"。
- CARLA 崩溃导致没分的路线，**计 0 分**（论文 4.1 原话）。
- 基线数字是 base 子集上训的（不是 full 10000），引用时要注明。

## 方法论拆解（四）：基线实验的结论

论文在 base 上训了 7 个基线（AD-MLP、UniAD-Tiny/Base、VAD、TCP/TCP-ctrl/TCP-traj、ThinkTwice、DriveAdapter），220 条路线闭环跑。关键结论：

1. **开环 L2 与闭环不相关**（再一次）：UniAD-Base 的 L2（0.73）比 VAD（0.91）低，闭环 DS 却更差（37.72 vs 39.42）。AD-MLP L2 3.64、闭环 DS 9.14 / SR 0——盲开模型现形。
2. **专家特征蒸馏是最大红利**：带 * 的方法（TCP/ThinkTwice/DriveAdapter，用 Think2Drive 特征蒸馏）大幅领先不带的（UniAD/VAD）。DriveAdapter DS 42.91 / SR 30.71 全场最好。但论文也诚实地说：真实世界很难拿到专家特征。
3. **轨迹头 >> 控制头**：TCP-traj（36.78/26.82）碾压 TCP-ctrl（18.63/5.45）——直接预测控制量不如预测轨迹再跟。这对 RAMP 的规划器输出形式选择有参考价值。
4. **交互技能全军覆没**：所有方法的 Merging/Overtaking/Emergency Brake 分数都难看（最好的 DriveAdapter mean 也只有 38.23，Emergency Brake 34.71）。论文归因于两点：① 长尾——一个 clip 里真正交互的帧很少；② 模仿学习范式本身不教"博弈思考"。——这正是 RAMP 这类"带推理的 agent"要切的蛋糕。

## 已知坑（RAMP 实验前必读）

1. **DS 的指数惩罚只是被缓解、没有被根除**。150 米路线里连乘次数少了，但"闯一次红灯 ×0.7"的狠劲还在。**DS 天然奖励保守驾驶**（论文 4.2 原话：heavy punishment encourages over-conservative driving）。RAMP 的风险校验模块会让车更谨慎——DS 上可能不占便宜，**必须同时报 SR 和分技能分**，故事才讲得圆。
2. **"太慢"（Too Slow）惩罚 0.70 很主观**：定义是"跟不上周围车流速度"。保守策略容易在这里被扣分，和上一条是同一枚硬币的两面。
3. **基线训练量打折**：附录 C 写明，UniAD/VAD/ThinkTwice/DriveAdapter 只训了原论文 **1/10 的 epochs**（因为 Bench2Drive 10Hz 采样、帧数是 nuScenes 10 倍但冗余大，按训练步数对齐）。引用基线数字时要知道这个前提；RAMP 如果训满，数字可能更好看，但也要注明公平性。
4. **ThinkTwice/DriveAdapter 被改过**：为公平对比，论文把两者改成了 6 相机无 LiDAR（原论文有用 LiDAR 的版本）。别拿 Bench2Drive 里的数字去和原论文的数字直接比。
5. **数据分布不均**：城镇分布被 Town12/13 主导（新城镇大、新场景只支持新城镇）；天气倒是接近均匀。跨城镇泛化结论要谨慎。
6. **44 场景里有 5 个是自造的**：超出 LB v2 的 39 个，是 Bench2Drive 团队为行为多样性新设计的（附录 D）。和 LB v2 做对比时注意这 5 个的差异。
7. **交互帧的长尾**：场景数均衡了，但每个 clip 里真正需要交互的帧只占少数——模仿学习吃不饱。这也是 RAMP 用"情景记忆+风险校验"而非纯模仿的论据。
8. **版本时效**：v1 是 2024-06 的；Leaderboard 2.1（2025-03）已改线性记分，Bench2Drive 用的是 2.0 口径。写论文时注明"following Bench2Drive v1 / LB 2.0 metric"。

## 与方向的关系

Bench2Drive 是 E2E-AD 社区在**评测方法论**上的一次修正：从"一条长路线、一个总分"转向"多场景、短路线、分技能"。它不发明新算法，但它定义了"驾驶能力"在 2024 年后学界的**操作定义**——Merging/Overtaking/Emergency Brake/Give Way/Traffic Sign 这 5 个技能，正在成为论文 related work 和实验表的默认切分。

对"智能体组织自动驾驶流程"方向的意义：分技能评估天然适配**多规划器架构**——每个技能可以对应/评估不同的规划器，RAMP 的"LLM 调度多规划器"正好能在这个考场上讲出"哪个规划器管哪个技能"的故事。

## 与 RAMP 的血缘/启发

1. **目标考场，口径已抠细**：RAMP 的第 6 章（实验标准）直接用本篇的 SR/DS 定义和惩罚表；写论文时 DS 公式、惩罚系数、"CARLA 崩溃计 0 分"这些口径细节建议照抄本篇附录 E 的表述，避免审稿人挑口径。
2. **分技能报告是 RAMP 的叙事主战场**：RAMP 的两大创新（风险加权记忆、世界模型复用）在 Emergency Brake 和 Give Way 这类强交互技能上最该出彩——基线在这两项全军覆没（<35%），正是 RAMP 讲"我们不一样"的对比位。Merging 则对应多规划器调度的价值。
3. **DS 保守 bias 的应对**：RAMP 带闭环风险校验，行为会偏保守。要在论文里**主动承认并拆解**这个 bias：同时报 SR（硬指标）、DS、以及"违章次数/公里"和"平均速度"——证明保守没有滑向"停着不动"（early stopping 的反面教材见 165 篇）。
4. **基线复现的起点**：DriveAdapter（DS 42.91 / SR 30.71）是 RAMP 要超过的公开最强基线；TCP-traj vs TCP-ctrl 的对比提示 RAMP 的规划器输出用轨迹形式更稳。
5. **数据策略**：官方 base（1000 clips）是 8×3090 可训的量级——RAMP 的 GPU 预算规划（云 GPU 租赁方案）可以直接对标这个数字；full（10000）是冲 SOTA 才需要的。
6. **论证"为什么不用 NAVSIM"**：Bench2Drive 是真闭环（模型动作改变下一帧输入），NAVSIM 是非反应式——RAMP 的闭环风险校验和冲突仲裁**必须**在真闭环里测。这条对比论证见 167 篇，两篇合在一起构成第 6 章"benchmark 选择"小节。
