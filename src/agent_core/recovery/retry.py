"""Retry helpers."""

# 文件说明：
# - 本文件属于 Retry / Recovery 层，负责重试、降级、JSON repair 或恢复计划。
# - 失败时应清楚记录原因，不能无依据编造答案。
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(fn: Callable[[], T], attempts: int = 2) -> T:
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error

