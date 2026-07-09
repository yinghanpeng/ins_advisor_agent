# LangSmith 集成

LangSmith 是可观测性和评估增强层，不是主业务强依赖。

## 环境变量

- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`

## 降级策略

如果开启 tracing 但没有 API Key，系统会：

1. 写 warning 日志；
2. 自动降级到本地结构化日志；
3. 不影响主业务响应。

## 应记录的内容

- 每个 `AgentGraph` 节点的输入摘要和输出摘要；
- 工具调用；
- RAG query、chunk、score、rerank score；
- Guardrail 结果；
- Sales Intelligence 选中的卡片；
- 最终回答；
- eval feedback。

当前 adapter：

- `src/agent_core/observability/langsmith_client.py`

