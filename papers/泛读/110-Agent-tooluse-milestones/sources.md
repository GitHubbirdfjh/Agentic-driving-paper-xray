# 110 原文来源（均为公开网页，2026-10-11 抓取，仅作来源备查）

## 来源 1：ChatGPT Plugins 发布报道
- 标题：OpenAI connects ChatGPT to the internet
- 作者：Kyle Wiggers，TechCrunch
- 日期：2023-03-23
- URL：https://techcrunch.com/2023/03/23/openai-connects-chatgpt-to-the-internet/
- 要点备查：plugins alpha 先上 waitlist、Plus 用户优先；官方 web-browsing 插件（Bing Search API、回答中列引用来源）；code interpreter（沙箱 Python + 磁盘空间，传文件/下结果）；开源 retrieval 插件；首批第三方：Expedia, FiscalNote, Instacart, Kayak, Klarna, Milo, OpenTable, Shopify, Slack, Speak, Wolfram, Zapier。

## 来源 2：Function Calling 发布说明（转述）
- 标题：OpenAI GPT-3.5 Turbo and GPT-4 (Lower Pricing & New Model)
- 作者：jasonchan，DEV Community
- 日期：2023-06-13（转述 OpenAI 当日官方更新）
- URL：https://dev.to/jasonchan/openai-gpt-35-turbo-and-gpt-4-lower-pricing-new-model-10f4
- 要点备查：gpt-4-0613 / gpt-3.5-turbo-0613 新增 function calling；`functions` 参数 + JSON Schema；返回 `function_call: {name, arguments}`，`finish_reason: function_call`；三步循环（调模型→开发者调真实 API→以 `role: function` 消息回填→模型总结）；gpt-3.5-turbo-16k；embeddings 降价 75%（$0.0001/1K tokens）；turbo 输入降价 25%（$0.0015/1K in、$0.002/1K out）；6 月 27 日稳定版模型名自动升级，0301/0314 老模型 9 月 13 日退役。

## 补充核实（2026-10-11 网页检索）
- ChatGPT plugins 废弃：2024-03-19 宣布废弃 plugin beta，2024-04-09 后插件对话逐步下线，由 GPTs 接替（KnowTechie 2024-03-08、Techopedia 等多家报道口径一致）。
- 2026-09 起 OpenAI 宣布以新一代 plugins（skills + MCP server）替代 custom GPTs（2026-12-11 退役），属另一轮迭代，与 2023 版 plugins 名字相同、内涵不同。
