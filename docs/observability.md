# Observability 可观测性

可观测性用于回答三个问题：

1. 请求走到了哪里？
2. 每一步为什么这么决策？
3. 出错时如何复盘？

## 本地结构化日志

本地日志始终可用，LangSmith 只是增强层。

每次请求至少记录：

- `trace_id`
- `session_id`
- `tenant_id`
- `workflow_name`
- `domain_skill`
- `intent`
- `current_state`
- `final_state`

## 当前实现

- `AgentState.move_to()` 记录每次状态迁移；
- `AgentState.add_trace_event()` 记录节点、检索、Guardrail 等事件；
- `WorkflowEngine` 通过 `StructuredLogger` 输出结构化日志；
- `AgentRunResponse` 返回 `trace_events` 和 `state_transitions` 方便本地调试。

## 后续扩展

- 接 OpenTelemetry；
- 写入云日志；
- 对齐 LangSmith run tree；
- 增加 latency、token usage、tool error 指标。

