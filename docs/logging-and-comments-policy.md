# 注释与日志规范

本项目要求代码不仅能运行，还要能解释、能追踪、能排错。

## 注释规范

- 每个 Python 文件必须有 module docstring，说明文件职责。
- 公开类和关键函数必须有 docstring。
- 复杂业务逻辑需要短中文或英文注释说明原因，而不是解释显而易见的语法。
- mock、adapter、placeholder 必须明确标注，不能伪装成真实生产实现。

## 日志规范

本地结构化日志始终可用，LangSmith 只是增强层。

每次请求至少记录：

- `trace_id`
- `session_id`
- `tenant_id`
- `workflow_name`
- `domain_skill`
- `intent`
- `current_state`
- `final_state`

每次状态迁移记录：

- `from_state`
- `to_state`
- `reason`
- `metadata`
- `ts`

工具、RAG、Guardrail、Sales Intelligence 应记录：

- 输入摘要；
- 输出摘要；
- 选择的工具或卡片；
- score / rerank score；
- risk level；
- retry count；
- error；
- latency。

## 当前实现

- `AgentState.move_to()` 写入 `state_transitions` 和 `trace_events`。
- `WorkflowEngine` 将 state transitions 和 trace events 写入 `StructuredLogger`。
- `LangSmithAdapter` 在无 API Key 或不可用时降级，不影响主业务。

## 后续扩展

- 将 `StructuredLogger` 输出接入 JSON log 文件、OpenTelemetry 或云日志。
- 给每个 tool adapter 增加 latency 和 error 统计。
- 将 `AgentGraph` 节点事件与 LangSmith run tree 对齐。

