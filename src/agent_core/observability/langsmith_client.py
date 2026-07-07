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

    enabled: bool
    project: str | None = None
    endpoint: str | None = None
    available: bool = False
    warning: str | None = None

    @classmethod
    def from_env(cls, log: StructuredLogger | None = None) -> "LangSmithAdapter":
        tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        api_key = os.getenv("LANGSMITH_API_KEY")
        project = os.getenv("LANGSMITH_PROJECT", "insurance-advisor-agent-local")
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        if not tracing:
            return cls(enabled=False, project=project, endpoint=endpoint, available=False)
        if not api_key:
            warning = "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing"
            if log:
                log.warning("langsmith_degraded", reason=warning)
            return cls(enabled=True, project=project, endpoint=endpoint, available=False, warning=warning)
        try:
            import langsmith  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
            warning = f"langsmith import failed: {exc}"
            if log:
                log.warning("langsmith_degraded", reason=warning)
            return cls(enabled=True, project=project, endpoint=endpoint, available=False, warning=warning)
        return cls(enabled=True, project=project, endpoint=endpoint, available=True)

    def trace_event(self, name: str, payload: dict[str, Any]) -> None:
        """Record an event when LangSmith is available.

        Current implementation is intentionally a no-op adapter. The production
        integration point is isolated here so local tests do not require network
        access or API keys.
        """
        if not self.enabled or not self.available:
            return

