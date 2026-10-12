# 从"猜文本"到"调函数"：tool-use 范式被工程化的两个月

> **信息块**：两个合并的里程碑，非论文。
> ① **ChatGPT Plugins**（OpenAI，2023-03-23 发布，TechCrunch 2023-03-23 报道）：ChatGPT 首次官方接入外部世界——联网浏览、代码解释器、第三方插件；2024-03-19 宣布废弃，2024-04-09 起插件对话逐步下线，由 GPTs 接替（此为 2024 年 3 月多家科技媒体报道确认的时间线）。
> ② **Function Calling**（OpenAI，2023-06-13 随 gpt-4-0613 / gpt-3.5-turbo-0613 发布）：Chat Completions API 新增 `functions` 参数，模型可输出结构化函数调用；同批还有 gpt-3.5-turbo-16k、embeddings 降价 75%、turbo 输入降价 25%。
> **核心洞见一句话**：这两个月里，tool-use 从"让模型在文本里猜 JSON"变成了"平台提供的类型化协议"——Plugins 回答了"LLM 怎么连外部世界"（产品形态），Function Calling 回答了"开发者怎么让 LLM 调函数"（API 形态）；**agent 从此有了地基**。

## 2023-03-23：Plugins——ChatGPT 第一次长出手脚

背景：ChatGPT 发布时知识截止于 2021 年 9 月，不会联网、不会算数、不会碰外部数据。3 月 23 日 OpenAI 推出 plugins（alpha，先上 waitlist，Plus 订阅用户优先），TechCrunch 当天的报道抓住了三个重点。

**第一，官方的 web-browsing 插件**：用 Bing Search API 检索，把访问过的网页在回答里列出来当引用。此前 OpenAI 内部实验 WebGPT（2021）已经发现带联网的 chatbot 会"挑对用户有说服力的来源引用，而不挑最可靠的"；Meta 的 BlenderBot 3.0 联网后迅速翻车。OpenAI 明知风险还是上了，配套的是红队测试过的安全措施——效果如何"让时间证明"（报道原话大意）。

**第二，code interpreter**：给 ChatGPT 一个带防火墙的沙箱 Python 解释器加磁盘空间，能上传文件、下载结果。数学、数据分析、可视化、格式转换是官方点名的用处。这个插件后来成了 ChatGPT 最成功的功能之一（2023 年 7 月向 Plus 全量开放后改名 Advanced Data Analysis），是 plugins 全家桶里唯一活下来的。

**第三，第三方插件生态**：首批 11 家——Expedia、FiscalNote、Instacart、Kayak、Klarna、Milo、OpenTable、Shopify、Slack、Speak、Wolfram、Zapier，外加 OpenAI 开源的 retrieval 插件（让 ChatGPT 用自然语言问自己的文档/邮件/笔记）。OpenTable 查餐厅、Instacart 下单、Wolfram 算数、Zapier 接 Gmail/Trello/Google Sheets——**"语言当 API 网关"的产品形态在这里第一次完整亮相**。

技术上插件的机制是：开发者提供一份 manifest（`ai-plugin.json`，描述插件是谁、能干什么）+ 一份 OpenAPI 规范（函数签名），ChatGPT 的模型在对话中决定何时调、调哪个、填什么参数。这个"模型看描述、选函数、填参数"的三段式，就是三个月后 Function Calling 的 API 化版本。

**它为什么死**：2023 年 12 月 OpenAI 发邮件让插件开发者转做 GPTs（"RIP Plugins"在开发者社区刷屏），2024 年 3 月 19 日正式宣布废弃插件 beta，4 月 9 日后插件对话逐步不可用。直接原因是 GPTs + GPT Store（2023 年 11 月 DevDay 发布）的采用率远超插件；深层原因是插件的"模型看 OpenAPI 猜参数"在多步任务里不可靠，而 GPTs 的 Actions + 更强的模型把同一件事做得更稳。讽刺的尾声：2026 年 9 月 OpenAI 又宣布 2026 年 12 月起用**新一代 plugins**替代 custom GPTs——历史转了一圈，名字回来了，内涵换成了 skills + MCP server。（此为 2026 年 10 月多家媒体报道，见本次检索。）

## 2023-06-13：Function Calling——agent 的地基

如果说 Plugins 是产品实验，Function Calling 就是**协议**。dev.to 的文章转述了 OpenAI 当天的更新，核心是一个三步循环：

**第 1 步**，开发者在调 `chat/completions` 时多传一个 `functions` 参数，每个函数用 JSON Schema 描述（名字、描述、参数类型、必填项）。用户问"Boston 天气怎么样"，模型不再用自然语言瞎猜，而是返回一个结构化的 `function_call: {name: "get_current_weather", arguments: "{\"location\": \"Boston, MA\"}"}`，`finish_reason` 为 `function_call`。

**第 2 步**，开发者**自己**去调真正的天气 API（模型不碰网、不执行代码，它只负责"决定调哪个、填什么参"）。

**第 3 步**，把 API 结果以 `role: "function"` 的消息塞回对话，模型再总结成自然语言回答用户。

三点值得展开：

1. **"模型不执行，只决策"**：执行权始终在开发者手里。这是和 Plugins 最大的不同，也是安全模型的关键——AutoGPT（109）那种"模型直接调 `execute_python_file`"的裸奔，在 Function Calling 的世界里是不可能的，中间永远隔着开发者写的一层 dispatch 代码。
2. **杀死了"文本里抠 JSON"**：AutoGPT 的 `fix_and_parse_json`、HuggingGPT 的 `logit_bias: 0.2` 强推大括号，都是 2023 年上半年"让模型输出合法 JSON"的裱糊匠。Function Calling 让 0613 版模型经过微调、专门学会"何时该调函数、按 schema 填参"，结构化输出从 prompt 技巧变成了模型能力。
3. **同批的经济账**：gpt-3.5-turbo-16k（4k→16k 上下文）、embeddings 降价 75%（$0.0001/1K tokens）、turbo 输入降价 25%（$0.0015/1K input、$0.002/1K output）。agent 循环是 token 黑洞，降价和上下文翻倍直接决定了"多步 agent 跑不跑得起"。OpenAI 在 6 月 27 日把稳定版模型名自动升级到 0613，老版本 9 月 13 日退役——**agent 开发者第一次体会到"地基在脚下换"的感觉**。

后续演进一句话：2023 年 11 月 DevDay 把 `functions` 升级为 `tools`（支持并行函数调用 parallel function calling），之后 Assistants API 把"循环+记忆+工具"做成托管服务。Function Calling 是这条线的起点。

## 这两个月到底改变了什么：接口的类型化

把两个里程碑放在一起看，2023 年 3–6 月发生的本质变化是：**LLM 与外部世界的接口从"文本约定"变成了"类型化协议"**。

- 3 月之前：AutoGPT 靠 prompt 里写"COMMANDS 清单"，模型输出文本，开发者用字符串匹配 dispatch，错一点就崩。
- 3 月：Plugins 让模型看 OpenAPI 描述选插件，但还是"模型猜"，猜错就调错 API。
- 6 月：Function Calling 把函数签名变成 API 的一等参数，模型输出被约束在 schema 内，dispatch 变成查表。

每一跳都在做同一件事：**把"模型说了什么"和"系统做了什么"之间的缝隙收窄**。缝隙越小，agent 越可靠；agent 的可靠性天花板，从此由"模型有多准"变成了"协议有多严"。这也是为什么今天的 agent 框架（LangChain、AutoGen、MCP）全部建立在 function calling 之上，而不是 prompt 之上。

## 与 RAMP 的血缘/划界

**血缘是直接的**：RAMP 的"LLM 在线多规划器调度"，在实现层就是 Function Calling 范式的驾驶特化版——LLM 不直接输出方向盘转角，而是输出"调用哪个规划器、填什么参数"的结构化决策，规划器是 typed tool，调度器是 caller。没有 2023 年 6 月这个协议，就没有"LLM 调度器"这个概念的可工程化形态。HuggingGPT（107）的模型选择是"单选题"，Function Calling 把它变成了"带 schema 的函数调用"，RAMP 的调度器输出 schema 设计应当直接继承后者。

**划界有两条**。第一，Function Calling 的三步循环是**开环**的（调一次、拿结果、总结），没有"调用结果不好怎么办"的回路；RAMP 的调度是闭环的，世界模型 rollout 和风险校验每一拍都在评估上一次调度的后果，不行就换规划器——"调用"只是开始，"校验"才是 RAMP 的正文。第二，Plugins/Function Calling 的安全模型是"开发者负责执行层"，RAMP 把"安全"内化成了一个独立模块（安全盾/风险校验），而不是留给调用方自觉——这是从"协议"到"系统"的升级，也是 RAMP 在 ICAPS 上区别于"又一个 LLM agent 框架"的关键论述点。
