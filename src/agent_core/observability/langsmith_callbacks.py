"""Placeholder callbacks for LangGraph/LangChain integrations."""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

from typing import Any


def build_langsmith_callbacks(enabled: bool) -> list[Any]:
    """Return callback handlers when remote tracing is enabled.

    This stays empty until the production model provider and callback strategy
    are selected.
    """
    return [] if not enabled else []

