# LangSmith 集成

LangSmith 是可观测性和评估增强层，不是主业务强依赖。

## 它解决什么问题

本地 JSON 日志适合实时排障和基础审计；当前接入把一次 Agent 请求组织成可视化 Run Tree，用于查看节点耗时、
路由与风控决策、工具/RAG 边界和错误类型。LangSmith 平台还支持 Dataset、Evaluator、Experiment 和版本比较，
但这些远程评估能力需要单独建设，不参与保险业务决策。

## 当前项目的真实状态

运行时远程 Trace 已完整接入：

- `WorkflowEngine.run()` 为每次请求创建一个 `Insurance Advisor Agent` 根 Run；
- 每次真实状态迁移创建一个动态子 Run，名称包含步骤序号和中文步骤名；
- 子 Run 根据节点职责使用 `chain`、`llm`、`retriever` 或 `tool` 类型，控制台可以直接还原 Agent 执行树；
- 节点事件、最终状态和失败类型写入对应 Run，成功与异常路径都会正确结束根 Run；
- 完整模式为根 Run 保存请求契约，为每个子 Run 保存 `state_before/state_after`，并记录真实模型请求与供应商响应；
- 每次 Chat Completion 创建嵌套 `llm` Run，标准 metadata 包含 `ls_provider=openai`、`ls_model_type=chat` 和实际模型名；
- 供应商 `prompt_tokens/completion_tokens` 被规范化为 `input_tokens/output_tokens/total_tokens`，LangSmith 用它聚合 Tokens 并按模型价格目录计算 Cost；
- 非流式 HTTP 调用在完整响应到达时写入首个 `new_token` 事件，因此 First Token 是当前协议可观测到的 TTFT 上界；真正逐 token TTFT 需要将模型客户端升级为 Streaming；
- SDK 使用异步批量发送，业务线程不等待每一条远程写入；应用关闭时在配置的超时时间内 flush；
- 远程异常只写本地 warning 并自动降级，不影响 Agent 主业务结果；
- 本地 `state_transitions`、`trace_events`、`agent_flow_step` 和 `agent_flow_summary` 始终保留。

远程 Dataset/Experiment 自动执行是另一项能力。当前 `run_langsmith_eval()` 仍明确返回 `skipped`，因此不能把
“运行时 Trace 已接入”表述成“LangSmith 的评估平台能力全部完成”。仓库中的旧 Exporter/Callback 是兼容边界，
当前主链路由 `LangSmithAdapter` 的 SDK Run Tree 实现负责，不通过旧 REST Exporter 上传。

## 环境变量

- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_WORKSPACE_ID`：可选，多 Workspace 账号用于显式选择目标 Workspace；
- `LANGSMITH_SAMPLING_RATE`：可选，`0.0` 到 `1.0`，控制运行时采样率；
- `LANGSMITH_FLUSH_TIMEOUT_SECONDS`：可选，应用关闭时等待批量上传完成的最长秒数。
- `LANGSMITH_DATA_POLICY`：`control_plane_only` 或 `full_business_content`；
- `LANGSMITH_MAX_FIELD_CHARS`：完整模式单字符串最大字符数，默认 `50000`；
- `LANGSMITH_MAX_COLLECTION_ITEMS`：完整模式单集合最大项目数，默认 `500`。
- `LANGSMITH_THREAD_GROUPING`：当前项目配置为 `true`，使用业务 Session 作为 `thread_id`，将多轮根 Run
  聚合为 Threads/Turns；每个 Turn 仍保留完整 Trace Waterfall。设为 `false` 时项目列表只按单条 Trace 展示。

## 降级策略

如果开启 tracing 但没有 API Key，系统会：

1. 写 warning 日志；
2. 自动降级到本地结构化日志；
3. 不影响主业务响应。

## 远程数据策略

`control_plane_only` 采用默认拒绝白名单，只上传：

- 当前状态、前后状态、步骤序号、中文步骤名和迁移原因；
- 意图、置信度、路由、风险、Guardrail、工具状态、计数和错误类型；
- 根 Run 的成功/失败状态、最终状态和流程节点数量；
- tenant/session 的不可逆短哈希引用。

`full_business_content` 在上述信息之外上传：

- 根 Run 的完整 `AgentRunRequest`，包括客户原文、多轮主体和公开 metadata；
- 每个状态节点的 `state_before/state_after`，覆盖 KYC 增量、业务记忆、意图候选、规划、风控和成本；
- 实际 Chat Completion messages、模型请求参数、供应商原始响应、规范化结果、Token 和延迟；
- 工具计划、参数、执行结果、RAG/双知识库正文、组装 Prompt、Grounding、合规结果和最终回答；
- 失败路径的完整状态和经过凭据清理的异常正文。

认证凭据不属于业务可观测数据。两种模式都不可关闭地清除字段名为 `api_key`、`password`、`authorization`、
`cookie`、`secret`、`access_token` 等值；同时扫描正文中的当前环境密钥、Bearer Token、`sk-`/`lsv2_pt_`
形式和带密码的数据库 URL。超出配置体积时会写明 `TRUNCATED` 或 `__truncated_items__`，不会静默伪装完整。

当前 adapter：

- `src/agent_core/observability/langsmith_client.py`

生产环境还应在 LangSmith 控制台和组织制度中设置 Workspace 权限、数据保留周期、采样率、告警以及第三方
数据处理边界；代码层的脱敏不能替代这些治理措施。

## Tokens、Cost 与 First Token

LangSmith 顶部三个指标来自真实模型子 Run，而不是普通状态 Run：

- `Tokens`：优先使用模型网关响应的 `usage.prompt_tokens/completion_tokens`，映射为 LangSmith 标准
  `usage_metadata`；网关不返回 usage 时使用中英文字符规则估算非零 Token，并在 metadata 中标记
  `token_usage_source=estimated`，避免把估算误认为供应商账单；
- `Cost`：使用 `ls_provider + ls_model_name + usage_metadata` 交给 LangSmith 的模型价格目录计算。本项目当前的
  `gpt-4.1-mini/gpt-4.1-nano` 会标记为 `openai`；如果企业网关内部结算价不同，应在 LangSmith Workspace 的
  Model Pricing 中配置企业价格，否则控制台显示的是 LangSmith 价格目录估算，不是企业账单；
- `First Token`：当前使用非流式 `/chat/completions`，只能在完整响应收到时产生首个 `new_token` 事件，因此是
  TTFT 上界。后续切换 Streaming 后，可在首个 delta 到达时写入同名事件，得到真实首 Token 延迟。

旧 Trace 不会被回填这些字段；必须重启服务并产生新请求，查看新建的 `LLM · 模型名` 子 Run。

## Waterfall 与 Threads/Turns

Agent 步骤位于单条 Trace 的 Waterfall 中；Threads/Turns 是按会话聚合多个根 Run 的聊天视图，本身不会把每个
状态子 Run 直接展开，但每个 Turn 都能打开对应 Trace 查看完整 Waterfall。当前项目同时保留多轮聚合与单轮步骤：

```text
LANGSMITH_THREAD_GROUPING=true
```

完整模式同时写 `business_session_id` 和 LangSmith 标准 `thread_id`。在 Threads 页面选择某个 Turn 后，通过该
Turn 的 Trace 链接进入 Waterfall 即可查看子步骤。根 Run 的标准 `input/output` 已映射到真实客户问题和最终回答，
不会再把 `trace_id` 渲染成 HUMAN 内容。若不需要多轮聚合，可改为 `false`。
