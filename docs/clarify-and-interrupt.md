# Clarify 短路与中断

Clarify 分支用于处理“信息不足，但继续工具/RAG/生成会变危险或低质”的请求。

## 为什么必须在工具和 RAG 前中断

如果缺少关键槽位却继续往下走，会出现几个问题：

- 工具调用参数不完整，容易查错对象；
- RAG 检索 query 太模糊，召回噪声证据；
- 模型会用假设补齐事实；
- 保险顾问场景可能把客户画像猜成事实。

因此 `_run_universal` 在 `context_need_planning` 后立即消费：

```python
if state.context_needs.get("clarify"):
    state = nodes.generate_clarification_response(state)
    state = nodes.response_packaging(state)
    state = nodes.trace_finalize(state)
    return state
```

工具循环中如果 planner 判断需要澄清，也会设置 `context_needs["clarify"]=True`，回到同一个短路出口。

## 节点行为

`generate_clarification_response` 会：

- 读取 `state.slot_values["missing_slots"]`；
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
