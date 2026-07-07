"""Identifier helpers."""

# 文件说明：
# - 本文件属于工具函数层，提供 ID、时间、JSON 等通用辅助能力。
# - 工具函数应保持简单、可测试、无业务耦合。
from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return a readable unique id with a stable prefix."""
    return f"{prefix}_{uuid4().hex}"


def new_trace_id() -> str:
    """Return a new trace id."""
    return new_id("trace")

