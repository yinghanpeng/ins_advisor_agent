"""运行时配置模块。

生产代码通过 `load_runtime_settings()` 读取模型、数据库、检索和记忆策略配置。
配置集中在这里，是为了避免业务节点里出现硬编码模型名、Base URL 或 API Key。
"""

from agent_core.config.runtime import (
    DatabaseConfig,
    MemoryConfig,
    ModelEndpointConfig,
    RetrievalConfig,
    RuntimeSettings,
    load_runtime_settings,
)

__all__ = [
    "DatabaseConfig",
    "MemoryConfig",
    "ModelEndpointConfig",
    "RetrievalConfig",
    "RuntimeSettings",
    "load_runtime_settings",
]
