# Streaming Events 事件骨架

第一版 streaming 只做事件骨架，不做真正 token-by-token SSE。

## 事件格式

`state.stream_events` 中每条事件结构为：

```json
{
  "event_type": "node_started",
  "trace_id": "trace_xxx",
  "node_name": "generate_response",
  "payload": {},
  "created_at": "2026-07-09T00:00:00Z"
}
```

支持的事件类型包括：

- `node_started`
- `node_finished`
- `tool_call_started`
- `tool_call_finished`
- `tool_loop_iteration`
- `model_delta`
- `final_answer`
- `error`

当前不会输出 token delta，`model_delta` 只是预留类型。

## 已接入节点

当前至少在这些节点写入 stream event：

- `initialize_context`
- `input_guardrail`
- `context_need_planning`
- `generate_clarification_response`
- `agentic_tool_loop`
- `general_tool_call`
- `generate_response`
- `grounding_verification`
- `compliance_review`
- `response_packaging`
- `trace_finalize`

## API 预留

旧接口保持不变：

- `/agent/run`

新增 adapter-ready 入口：

- `/agent/stream`
- `run_agent_stream(request)`

当前 `/agent/stream` 同步执行 workflow，然后返回 `stream_events` 和 `final_response`。未来可替换为 FastAPI `StreamingResponse` 或 SSE，不需要改 AgentState。

## PII 安全

stream payload 会经过输出 PII 脱敏工具处理。`final_answer` 事件只应出现在 `output_pii_scan` 之后的安全文本上。
