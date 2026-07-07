"""Trace event helpers."""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceRecorder:
    """In-memory trace recorder for tests and local debugging."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **fields: Any) -> None:
        """记录一条本地 trace 事件，供测试和调试查询。"""
        self.events.append({"event": event, **fields})

    def by_event(self, event: str) -> list[dict[str, Any]]:
        """按事件名称过滤 trace 事件。"""
        return [item for item in self.events if item.get("event") == event]
