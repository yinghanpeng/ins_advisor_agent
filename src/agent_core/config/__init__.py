"""运行时配置模块。

生产代码通过 `load_runtime_settings()` 读取模型、数据库、检索和记忆策略配置。
配置集中在这里，是为了避免业务节点里出现硬编码模型名、Base URL 或 API Key。
"""

from agent_core.config.runtime import (
    ApiRuntimeConfig,
    DatabaseConfig,
    IntentRoutingConfig,
    InsuranceKnowledgeConfig,
    MemoryConfig,
    ModelEndpointConfig,
    RetrievalConfig,
    RuntimeSettings,
    load_runtime_settings,
)

# 明确配置包的稳定公开接口，调用方无需依赖 runtime.py 的内部 YAML 辅助函数。
__all__ = [
    "ApiRuntimeConfig",
    "DatabaseConfig",
    "IntentRoutingConfig",
    "InsuranceKnowledgeConfig",
    "MemoryConfig",
    "ModelEndpointConfig",
    "RetrievalConfig",
    "RuntimeSettings",
    "load_runtime_settings",
]
