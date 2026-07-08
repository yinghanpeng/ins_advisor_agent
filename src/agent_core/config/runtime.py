"""集中运行配置加载。

生产级 Agent 不能把模型名、URL、数据库连接和检索权重散落在业务节点里。
本模块负责从 `configs/*.yaml` 读取配置，并把 `${ENV_NAME}` 占位符替换为环境变量。

设计意图：
1. 业务代码只依赖结构化 RuntimeSettings；
2. 缺少生产必需配置时 fail-fast，而不是回退到本地替代数据；
3. 测试可以构造 RuntimeSettings fixture，但生产路径必须来自配置和环境变量。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ModelEndpointConfig(BaseModel):
    """一个模型端点配置。"""

    provider: str = Field(default="openai_compatible", description="模型供应商类型，例如 openai_compatible。")
    model: str | None = Field(default="", description="模型名称，由环境变量或配置文件提供。")
    base_url: str | None = Field(default="", description="模型服务 Base URL，不允许硬编码在业务代码中。")
    api_key: str | None = Field(default="", description="模型服务 API Key，从环境变量加载。")
    timeout_ms: int = Field(default=15000, description="模型请求超时时间，单位毫秒。")
    max_retries: int = Field(default=2, description="模型请求最大重试次数。")
    dimensions: int | None = Field(default=None, description="Embedding 维度；非 embedding 模型为空。")


class DatabaseConfig(BaseModel):
    """数据库和 Redis 配置。"""

    database_url: str | None = Field(default="", description="PostgreSQL 连接字符串。")
    redis_url: str | None = Field(default=None, description="Redis 连接字符串，用于限流、队列或缓存。")
    pool_size: int = Field(default=5, description="数据库连接池大小。")
    echo_sql: bool = Field(default=False, description="是否输出 SQL 调试日志。")


class RetrievalConfig(BaseModel):
    """检索、RAG 和记忆召回配置。"""

    top_k: int = Field(default=8, description="默认召回 TopK。")
    score_threshold: float = Field(default=0.05, description="召回结果最低分数阈值。")
    vector_weight: float = Field(default=0.45, description="向量相似度在最终分中的权重。")
    lexical_weight: float = Field(default=0.25, description="关键词分在最终分中的权重。")
    metadata_weight: float = Field(default=0.15, description="metadata 分在最终分中的权重。")
    recency_weight: float = Field(default=0.10, description="时间新近度在最终分中的权重。")
    confidence_weight: float = Field(default=0.05, description="事实置信度在最终分中的权重。")


class MemoryConfig(BaseModel):
    """长期记忆策略配置。"""

    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = Field(default=True, description="租户是否允许使用长期记忆。")
    model_decision_enabled: bool = Field(default=True, description="规则无法确定时是否调用模型做召回决策。")
    decision_timeout_ms: int = Field(default=1200, description="长期记忆召回决策模型的延迟预算。")
    default_ttl_days: int | None = Field(default=None, description="长期记忆默认 TTL，空表示不过期。")
    max_recall_items: int = Field(default=8, description="长期记忆最大召回条数。")


class RuntimeSettings(BaseModel):
    """Agent Runtime 的集中配置。"""

    app_env: str = Field(default="local", description="运行环境，例如 local、test、staging、prod。")
    models: dict[str, ModelEndpointConfig] = Field(default_factory=dict, description="所有模型端点配置。")
    database: DatabaseConfig = Field(..., description="数据库配置。")
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig, description="检索配置。")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="长期记忆配置。")

    def require_model(self, name: str) -> ModelEndpointConfig:
        """读取必需模型配置；缺失时直接报错，避免业务节点静默降级。"""
        config = self.models.get(name)
        if config is None:
            raise RuntimeError(f"缺少模型配置：models.{name}")
        if not config.base_url or not config.api_key or not config.model:
            raise RuntimeError(f"模型配置不完整：models.{name}")
        return config


def load_runtime_settings(config_dir: str | Path = "configs") -> RuntimeSettings:
    """从配置目录加载 RuntimeSettings。"""
    base = Path(config_dir)
    models = _load_yaml(base / "models.yaml").get("models", {})
    database = _load_yaml(base / "database.yaml").get("database", {})
    retrieval = _load_yaml(base / "retrieval.yaml").get("retrieval", {})
    memory = _load_yaml(base / "memory.yaml").get("memory", {})
    return RuntimeSettings(
        app_env=os.getenv("APP_ENV", "local"),
        models=models,
        database=database,
        retrieval=retrieval,
        memory=memory,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 并做环境变量插值。"""
    if not path.exists():
        return dict()
    raw = path.read_text(encoding="utf-8")
    expanded = ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), raw)
    data = yaml.safe_load(expanded) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML mapping：{path}")
    return data
