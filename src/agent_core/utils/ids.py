"""Identifier helpers."""

# 文件说明：
# - 本文件属于工具函数层，提供 ID、时间、JSON 等通用辅助能力。
# - 工具函数应保持简单、可测试、无业务耦合。
from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return a readable unique id with a stable prefix."""

    # UUID4 提供随机唯一主体，稳定前缀让日志与存储中的实体类型易于识别。
    return f"{prefix}_{uuid4().hex}"


def new_trace_id() -> str:
    """Return a new trace id."""

    # 统一复用 new_id，确保所有 trace 标识遵循相同命名与唯一性规则。
    return new_id("trace")
