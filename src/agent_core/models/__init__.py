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

__all__ = [
    "ChatCompletionResult",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleEmbeddingClient",
    "RerankResult",
    "RerankerClient",
]
