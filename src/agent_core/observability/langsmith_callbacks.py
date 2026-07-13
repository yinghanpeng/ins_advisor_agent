"""LangSmith / LangChain callback 构建入口。"""

# 文件说明：
# - 本文件属于可观测性层，负责本地结构化日志、trace、metrics 或 LangSmith adapter。
# - LangSmith 不可用时，本地日志仍必须能支撑排查。
from __future__ import annotations

from typing import Any


def build_langsmith_callbacks(enabled: bool) -> list[Any]:
    """返回模型调用兼容列表；运行时追踪统一由 Engine 级 Run Tree 负责。"""

    # 自研模型客户端不依赖 LangChain Callback；启用远程追踪时也不能重复创建第二套 Run。
    if enabled:
        # 返回空列表，实际根 Run 和节点子 Run 由 LangSmithAdapter 在 WorkflowEngine 中创建。
        return []
    # 关闭状态同样返回新的空列表，调用方可直接拼接且不会共享可变对象。
    return list()
