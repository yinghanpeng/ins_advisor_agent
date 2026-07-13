# Clarify 短路与中断

Clarify 分支处理两类问题：意图置信度低/活跃意图变化不明确，以及具体工具必填参数不足。
保险 KYC 的缺失业务信息由代码化 Handler 的 `missing_fields` 和 `generate_kyc_questions` 管理，
不复用通用工具参数槽位。

## 为什么必须在工具和 RAG 前中断

如果工具参数不符合选中工具的 Schema 却继续往下走，会出现几个问题：

- 工具调用参数不完整，容易查错对象；
- 工具 Provider 收到空地点、空算式或空 URL；
- 执行错误被误当成外部服务故障；
- 模型可能用假设补齐本应由用户确认的参数。

因此 `_run_universal` 先完成工具选择和 Schema 校验，再在执行器前消费：

```python
state = nodes.general_tool_routing(state)
if state.context_needs.get("clarify"):
    state = nodes.generate_clarification_response(state)
    state = nodes.response_packaging(state)
    state = nodes.trace_finalize(state)
    return state
```

外部 planner 如果在工具选择前已经判断需要澄清，也可以复用同一个短路出口。

## 节点行为

`generate_clarification_response` 会：

- 优先读取 `intent_clarification_question`；
- 读取 `state.metadata["missing_tool_arguments"]` 和 `tool_argument_validation`；
- 生成简洁澄清问题；
- 设置 `state.intent="clarify"`；
- 设置 `state.capability_route="clarify"`；
- 写入 `state.answer`；
- 写入 `state.clarification_question`；
- 关闭 `context_needs.tool`；
- 关闭 `context_needs.rag`；
- 写入 trace 和 stream event。

它不会：

- 调 RAG；
- 调工具；
- 进入大模型生成；
- 写长期记忆候选。

## API 输出

前端可以从这些位置判断本轮是澄清：

- `intent == "clarify"`；
- `response_package["clarification_question"]`；
- `context_needs["clarify"] is True`；
- `trace_events` 中存在 `generate_clarification_response`。

## 与 KYC 缺失字段的边界

- 通用工具：`ToolSpec.input_schema → ToolInputValidator → Clarify`；
- 低置信意图：`vector/LLM confidence → Clarify`；
- KYC 领域：`InsuranceKycDelta → profile_state / missing_fields → status_router → generate_kyc_questions`。

项目不再维护一套位于两者之前的全局 `extract_slots / validate_slots`。这样工具参数由工具契约
负责，客户事实由业务状态模型负责，避免同一字段被重复抽取和校验。
