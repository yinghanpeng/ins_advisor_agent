"""LangSmith adapter with graceful degradation."""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agent_core.observability.logger import StructuredLogger


@dataclass
class LangSmithAdapter:
    """Optional LangSmith tracing facade.

    The adapter deliberately does not make business logic depend on LangSmith.
    """

    # enabled 表示用户是否希望启用 LangSmith tracing。
    enabled: bool
    # project 是 LangSmith 项目名，用于把不同环境的 trace 归档到不同项目。
    project: str | None = None
    # endpoint 是 LangSmith API 地址，默认官方 endpoint。
    endpoint: str | None = None
    # available 表示当前运行环境真的具备可用 LangSmith 依赖和 API key。
    available: bool = False
    # warning 保存降级原因，例如缺少 API key 或 import langsmith 失败。
    warning: str | None = None

    @classmethod
    def from_env(cls, log: StructuredLogger | None = None) -> "LangSmithAdapter":
        # LANGSMITH_TRACING=true 时才尝试启用，默认保持关闭，保证本地测试不依赖外部服务。
        tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        # API key 只从环境变量读取，避免把密钥写入配置或代码。
        api_key = os.getenv("LANGSMITH_API_KEY")
        # 项目名有默认值，方便本地开启 tracing 后不必额外配置。
        project = os.getenv("LANGSMITH_PROJECT", "insurance-advisor-agent-local")
        # endpoint 支持环境变量覆盖，兼容自建或代理环境。
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        # 未启用 tracing 时返回 disabled adapter，业务逻辑照常运行。
        if not tracing:
            return cls(enabled=False, project=project, endpoint=endpoint, available=False)
        # 用户启用了 tracing 但没有 API key 时，降级并写 warning，不让 Agent 主链路失败。
        if not api_key:
            warning = "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing"
            if log:
                log.warning("langsmith_degraded", reason=warning)
            return cls(enabled=True, project=project, endpoint=endpoint, available=False, warning=warning)
        # 尝试 import langsmith，验证依赖是否安装；失败时同样降级。
        try:
            import langsmith  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
            warning = f"langsmith import failed: {exc}"
            if log:
                log.warning("langsmith_degraded", reason=warning)
            return cls(enabled=True, project=project, endpoint=endpoint, available=False, warning=warning)
        # tracing 开启、API key 存在、依赖可导入时才标记为 available。
        return cls(enabled=True, project=project, endpoint=endpoint, available=True)

    def trace_event(self, name: str, payload: dict[str, Any]) -> None:
        """Record an event when LangSmith is available.

        Current implementation is intentionally a no-op adapter. The production
        integration point is isolated here so local tests do not require network
        access or API keys.
        """
        # LangSmith 不可用时直接返回，保证 tracing 永远不是业务链路的硬依赖。
        if not self.enabled or not self.available:
            return
        # 这里预留真实 LangSmith run/event 写入逻辑；本地实现保持 no-op，避免测试访问网络。
