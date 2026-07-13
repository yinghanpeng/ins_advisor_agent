# Observability 可观测性

可观测性用于回答三个问题：

1. 请求走到了哪里？
2. 每一步为什么这么决策？
3. 出错时如何复盘？

## 本地结构化日志

`WorkflowEngine` 会在请求开始时绑定请求级 Trace Sink。每次状态迁移、节点开始/结束、工具调用、检索、风控、
记忆写入和 Trace 收尾事件都会立即输出一条 JSON 日志；即使后续节点异常，已完成步骤的日志也不会丢失。
实时日志仅保留节点名、状态、路由、风险、计数等控制面字段，不记录客户原文、KYC 值、Prompt、工具正文、
模型回答或 API Key。未处理异常另写 `agent_run_failed`，包含失败状态和异常类型，不包含异常参数原文。

为了让终端日志可以直接还原 Agent 编排，每次状态迁移还会输出 `agent_flow_step`，包含 `step_index`、中文
`step_name`、内部 `step_code` 和进入原因。请求结束或异常时输出 `agent_flow_summary`，其 `flow` 字段采用：

```text
初始化 → 输入安全拦截 → 恢复记忆 → 消息标准化 → 意图识别 → 语义风险分类 → 执行路由 → … → 完成
```

这是本轮真实执行路径，不是写死的固定流程；澄清、工具、保险 KYC、知识检索和恢复分支只在实际执行时出现。

本地日志始终可用，LangSmith 是增强层。启用并正确配置后，每次请求会创建一个远程根 Run，每次真实状态迁移
会创建一个子 Run；本地 `agent_flow_step` 的步骤序号和中文名称与远程子 Run 对齐。远程可配置为控制面或完整
业务内容模式，认证凭据始终强制清除；LangSmith 不可用时自动退回本地结构化日志。完整数据边界见
[LangSmith 集成](langsmith-integration.md)。

每次请求至少记录：

- `trace_id`
- `session_id`
- `tenant_id`
- `workflow_name`（仅兼容标签，不参与保险路由）
- `domain_skill`
- `intent`
- `intent_source`
- `intent_vector_score`
- `intent_confidence`
- `intent_dispatch_action`
- `active_intent_action`
- `current_state`
- `final_state`

## 当前实现

- `AgentState.move_to()` 记录每次状态迁移；
- `AgentState.add_trace_event()` 记录节点、检索、Guardrail 等事件；
- `WorkflowEngine` 通过 `StructuredLogger` 输出结构化日志；
- `LangSmithAdapter` 使用 SDK Run Tree 异步写入根请求、动态节点、完整状态快照和真实模型调用；
- 真实模型子 Run 写入标准 usage、模型价格匹配 metadata 和 `new_token` 时间，控制台可聚合 Tokens、Cost 与 First Token；
- 当前写入 LangSmith `thread_id` 聚合 Threads/Turns；每个 Turn 均保留可单独打开的完整 Waterfall Agent 步骤；
- 内部 `AgentRunResponse` 返回 `trace_events` 和 `state_transitions` 方便本地调试；客户 HTTP
  `PublicAgentRunResponse` 不返回这两项，排障通过 `trace_id` 关联服务端日志。
- 中置信意图写入 `medium_confidence_intent_routed`，用于离线补充意图知识库；这只是日志，不是人工审批。
- KYC Trace 只记录字段名、缺失项、轮次和评分，不记录家庭、资产等槽位原值。

## 后续扩展

- 接 OpenTelemetry；
- 写入云日志；
- 使用 LangSmith Dataset/Experiment 自动重放评估集并比较版本；
- 增加 latency、token usage、tool error 指标。
