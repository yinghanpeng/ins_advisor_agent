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

    # events 按发生顺序保存结构化事件；default_factory 避免实例之间共享列表。
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **fields: Any) -> None:
        """记录一条本地 trace 事件，供测试和调试查询。"""

        # 事件名与调用方字段合并后追加，保留完整时间顺序供断言和排障。
        self.events.append({"event": event, **fields})

    def by_event(self, event: str) -> list[dict[str, Any]]:
        """按事件名称过滤 trace 事件。"""

        # 返回新的列表而不是内部列表引用，避免调用方意外改变 recorder 状态。
        return [item for item in self.events if item.get("event") == event]
