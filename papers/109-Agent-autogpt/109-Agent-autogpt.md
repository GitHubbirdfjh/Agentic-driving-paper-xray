# 一个 while 循环点燃的 Agent 狂热：AutoGPT 到底证明了什么

> **信息块**：AutoGPT · Significant Gravitas（作者 Toran Bruce Richards）· **非论文，GitHub 开源项目**，仓库创建 2023-03-16，3 月底公开发布引爆社区（本次核对的是 v0.1.0 tag，2023-04-12，commit 16b37ff）· https://github.com/Significant-Gravitas/AutoGPT · **规模（2026-10-11 经 GitHub API 实测）：187,516 stars，45,926 forks**，GitHub 历史上 star 数最高的项目之一 · 代码已核对（v0.1.0 的 `scripts/` 全量）· **核心洞见一句话**：它证明的不是"autonomous AGI 实现了"，而是"GPT-4 + 一个 while 循环 + 一份工具清单"就足以让全世界相信 autonomous agent 触手可及——**它证明的是范式，不是能力**。

## 它到底证明了什么：三件事，多一件都不多

AutoGPT 的全部代码（v0.1.0）是 28 个 Python 文件、约一千行。它的"智能"全部来自三样东西的组合，没有一样是新的：

**第一，目标可以被"编译"成循环。** 用户给名字、角色、最多 5 个目标（默认那三个很说明问题："Increase net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously"），`main.py` 把它们拼进 system prompt，然后进入一个死循环：调 LLM → 解析出 JSON（含 `thoughts` 和 `command`）→ 执行命令 → 把结果塞回历史 → 下一轮。每轮的 user_input 固定就是一句话："Determine which next command to use, and respond using the format specified above:"。**整个"自主性"就是这句 prompt 被重复调用**。这件事在 2023 年 3 月是震撼的，因为此前所有人都在用 ChatGPT 做"一问一答"，没人把 LLM 放进 while True 里跑。

**第二，LLM 可以自己调用自己。** `commands.py` 里有三组今天看来极具预见性的命令：`evaluate_code` / `improve_code` / `write_tests`（AI 给自己写代码、改代码、写测试——self-improvement 的雏形），以及 `start_agent` / `message_agent` / `list_agents` / `delete_agent`（**子智能体**：主 agent 可以 spawn 子 agent 并给它派任务，这是 2023 年 4 月的代码，比"多智能体"成为热词早了半年）。

**第三，记忆可以外挂。** 短期记忆就是 `full_message_history`，按 token 数从旧到新往 prompt 里塞，超了就截断；长期记忆是 `memory/` 目录下的 LocalMemory / Pinecone / Redis 三种后端，每轮取最近 5 条消息做 query，`get_relevant(..., 10)` 拿回 10 条相关记忆，以"This reminds you of these events from your past:"注入 system prompt。向量检索记忆的工程形态，在这里已经齐了。

注意它**没有证明**的事：没有任何一个公开、可复现的 benchmark 显示 AutoGPT 能稳定完成多步任务。它的"战绩"全是 Twitter 上的录屏——而录屏是可以 cherry-pick 的。

## 架构三件套：循环、命令、记忆

### 第一件：思维循环（the loop）

`scripts/chat.py` 的 `chat_with_ai` 是全文最值得逐行读的函数，因为它暴露了 2023 年 agent 工程的全部妥协：

1. 组装 context：system prompt（含目标）+ 当前时间 + 相关记忆 + 从新到旧塞历史（`send_token_limit = token_limit - 1000`，留 1000 token 给回复）。
2. `if current_tokens_used > 2500: relevant_memory = relevant_memory[1:]`——**记忆不够 2500 token 就一条条砍**，简单粗暴，没有重要性排序（对比本报告 111 的 recency/importance/relevance 三因子，这里的记忆检索是裸奔的）。
3. 调 `create_chat_completion`，模型默认是 `cfg.fast_llm_model`（GPT-3.5），注意 main.py 里的 TODO 注释："This hardcodes the model to use GPT3.5"——**思考用的是 3.5，不是 GPT-4**，和宣传口径有出入。
4. 只 catch `openai.error.RateLimitError`，睡 10 秒重试。其他所有异常——JSON 解析失败、命令不存在、执行报错——都变成 `"Error: ..."` 字符串塞回给 LLM，让 LLM 自己看着办。**错误处理就是"把报错文本再喂回去"**，这是它频繁死循环的技术根源。

`main.py` 的交互循环里还有一处决定命运的设计：**默认每步都要人工确认**（`Enter 'y' to authorise command`），`--continuous` 模式会打出红色警告："potentially dangerous and may cause your AI to run forever"。安全和自主在这里是互斥的，作者选了安全，于是"全自主"从第一天起就是个需要用户坐在旁边按 y 的演示。

`thoughts` 字段（text / reasoning / plan / criticism / speak）是 prompt 工程的杰作：让模型在输出命令之前先写出思考过程，既是给用户看的"戏剧效果"，也是强迫模型做一步自我检查。但 `criticism` 只是文本自省，没有任何外部验证——模型批评自己三句话，然后照样执行错误的命令。

### 第二件：命令执行层（the tools）

v0.1.0 的命令清单（`execute_command`）：`google`（搜索，DuckDuckGo 或官方 API）、`browse_website`（抓取+摘要+取链接）、文件五件套（read/write/append/delete/search）、`execute_python_file`（**直接执行 Python，沙箱约等于无**）、`evaluate_code`/`improve_code`/`write_tests`（调 LLM 改代码）、`generate_image`、`memory_add`、子 agent 四件套、`get_text_summary`、`do_nothing`、`task_complete`（直接 `quit()`）。

这份清单定义了此后三年所有 agent 框架的工具本体：搜索、浏览、文件、代码执行、记忆、子 agent，一个不少。它的问题也定义了此后三年的工程课题：命令是字符串匹配调度的（`if command_name == "google"`），没有 schema、没有类型检查、没有权限分级。OpenAI 两个半月后推出的 Function Calling（本报告 110）本质上就是把这份"字符串清单"变成了"JSON Schema 清单"。

### 第三件：记忆（the memory）

`scripts/memory/local.py`（114 行）的实现值得逐行看，因为它暴露了 2023 年"向量记忆"的原始形态：记忆就是内存里的两个并行数组——`texts`（字符串 list）+ `embeddings`（OpenAI ada embedding，1536 维），落盘是一个 JSON 文件。`get_relevant(text, k)` 就三行：对 query 取 embedding、和记忆矩阵做**点积**（注意是原始点积，没做归一化，长文本天然占优）、`argsort` 取 top-k。和 111 Generative Agents 的 memory stream（recency 指数衰减 + importance 打分 + relevance 余弦，三因子加权、min-max 归一化）对比，这里的检索没有任何时间衰减、没有重要性、没有归一化——**记得一切，等于什么都没记住**，长任务跑到后面全是噪声。

还有一处耐人寻味的细节：`add()` 里 `if 'Command Error:' in text: return ""`——**执行报错的文本根本不进记忆**。错误只在当轮以 system message 形式喂回去，下一轮如果滑出历史窗口就彻底消失。agent 会反复踩同一个坑，因为它"记得"自己从没犯过错。这就是为什么 AutoGPT 的死循环往往不是"同一种错"，而是"同一种错换着花样"。

## 为什么火：天时、地利、人和

**天时**：2023 年 3 月 14 日 GPT-4 发布，3 月 30 日 AutoGPT 发布。GPT-4 的 API 刚开闸，全世界都在问"这东西到底能干什么"，AutoGPT 给出了第一个可运行的答案：把它放进循环里，它自己会定计划、会上网、会写代码。ChatGPT Plugins（3 月 23 日）是 OpenAI 官方的"连网"方案，但要排 waitlist；AutoGPT 是野生的、开源的、今晚就能跑的。

**地利**：README 的 demo 视频和默认目标"Entrepreneur-GPT"（让 AI 自主赚钱）精准击中了投机叙事。star 数在几周内冲破 10 万，媒体跟进，fork 数 4.5 万——它是 GitHub 历史上最快达成 10 万 star 的项目之一（此为社区共识表述，GitHub 官方未发布正式排名）。

**人和**：代码极简，一千行，任何人都能读完、改完、fork 出自己的变体（BabyAGI、GodMode 等一周内涌现）。**它的传播靠的不是性能，是可 fork 性**。

## 为什么迅速沉寂：四个结构性原因

**第一，可靠性是纸糊的。** 错误处理 = 把报错文本喂回去，LLM 经常陷入"报错→重试→再报错"的死循环；JSON 格式一错整轮作废（`fix_and_parse_json` 是裱糊匠）；没有任何状态机、没有 plan 的显式表示，"计划"只存在于 `thoughts.plan` 的文本里，执行一偏就回不来了。Twitter 上的成功录屏背后是大量"跑了 20 轮烧掉 5 美元 API 费然后卡死"的沉默案例（此为社区普遍反馈的归纳，非论文数据）。

**第二，成本结构不可持续。** 每轮一次 GPT-4（或 3.5）调用，context 里塞着全部历史 + 记忆 + 命令清单，token 烧得飞快。2023 年 GPT-4 的价格是 $0.03/1K input token，一个中等任务轻松烧掉几美元。没有缓存、没有小模型分流（`--gpt3only` 模式是后来加的妥协）。

**第三，安全设计杀死了"自主"。** `execute_python_file` 直接跑本地代码、文件操作无沙箱、联网无限制——默认的人工确认（每步按 y）是唯一的安全阀。这意味着"全自主"和"安全"不能兼得，而 continuous 模式的红色警告等于官方承认"这玩意儿可能失控"。企业级场景没法用，个人用户按 y 按到手酸。

**第四，被工程化更好的东西取代了。** 2023 年 6 月 Function Calling 把"字符串命令"变成"JSON Schema 函数"，8 月 AutoGen 给了多智能体对话框架，年底 OpenAI Assistants API 把"循环+记忆+工具"做成了托管服务。AutoGPT 的每一块都被更可靠的组件替换了。仓库本身也在 2024 年底彻底转型为 AutoGPT Platform（从 tag 名 `autogpt-platform-beta-*` 可见），原来的单文件 agent loop 不再是主角——**项目没死，但"AutoGPT"这个概念作为技术路线已经终结**。

## 与 RAMP 的血缘/划界

**血缘**：AutoGPT 是"LLM + 工具循环"这个大范式的**点火者**，RAMP 的"LLM 在线调度多规划器"在架构族谱上是它的后代。具体可继承的有两处：一是它的命令清单就是 RAMP 规划器接口设计的参考——搜索、文件、代码执行、子 agent 调度，RAMP 的"规划器"无非是把这些通用工具换成了驾驶专用的规划器（IDM、MPC、diffusion planner 等），调度逻辑同构；二是它的失败模式（死循环、幻觉命令、错误处理靠"把报错喂回去"）正是 RAMP 坚持"闭环风险校验独立于 LLM"的原因——**调度器会犯错，所以执行层必须有不依赖调度器的刹车**，AutoGPT 用人工按 y 当刹车，RAMP 用安全盾当刹车，这是同一问题的两种答案。

**划界**：AutoGPT 是通用、开环、无记忆管理的 prompt 循环；RAMP 是领域专用（驾驶）、闭环、有世界模型先验和情景记忆的系统。AutoGPT 的"自主性"是"没人管着 LLM"，RAMP 的"自主性"是"LLM 被世界模型和风险校验管着"。两者对"自主"的理解恰好相反——这也是 RAMP 在 ICAPS 这种 planning 会议上要讲清楚的：**我们不是在造一个更聪明的 AutoGPT，我们是在造一个被形式化约束管住的调度器**。
