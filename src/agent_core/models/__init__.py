"""模型访问层。

这里统一封装 Chat、Embedding 和 Reranker 调用。业务节点只能依赖这些客户端，
不能自行拼 URL 或读取 API Key，这样才方便审计、限流、重试和成本统计。
"""

from agent_core.models.client import (
    ChatCompletionResult,
    OpenAICompatibleChatClient,
    OpenAICompatibleEmbeddingClient,
    RerankResult,
    RerankerClient,
)

# 公开导出列表限定业务节点可直接依赖的稳定模型客户端与结果契约。
__all__ = [
    "ChatCompletionResult",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleEmbeddingClient",
    "RerankResult",
    "RerankerClient",
]
