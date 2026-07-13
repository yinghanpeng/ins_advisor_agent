"""Simple metrics container."""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    """在进程内按指标名累计整数计数，供测试与轻量本地观测使用。"""

    # counters 保存指标名到累计值的映射；每个实例使用独立字典。
    counters: dict[str, int] = field(default_factory=dict)

    def inc(self, key: str, value: int = 1) -> None:
        """把指定指标增加 value，尚不存在的指标从零开始。"""

        # 读取当前值并原子地完成单线程累加；本容器不承诺跨线程同步。
        self.counters[key] = self.counters.get(key, 0) + value
