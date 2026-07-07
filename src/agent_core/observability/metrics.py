"""Simple metrics container."""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    counters: dict[str, int] = field(default_factory=dict)

    def inc(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

