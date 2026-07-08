"""LangGraph / LangChain callback 构建入口。"""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

from typing import Any


def build_langsmith_callbacks(enabled: bool) -> list[Any]:
    """根据开关返回 callback handlers。"""
    if enabled:
        raise RuntimeError("LangSmith callback strategy 未配置，不能启用远程 callback")
    return list()
