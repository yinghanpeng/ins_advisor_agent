"""Retry helpers."""

# 文件说明：
# - 本文件属于 Retry / Recovery 层，负责重试、降级、JSON repair 或恢复计划。
# - 失败时应清楚记录原因，不能无依据编造答案。
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

# 泛型 T 保留被重试函数的原始返回类型，避免适配器把结果退化为 Any。
T = TypeVar("T")


def retry(fn: Callable[[], T], attempts: int = 2) -> T:
    """在限定次数内执行无参函数，并在全部失败后重抛最后一个异常。"""
    # 初始化最后异常容器，以便循环耗尽后保留最接近根因的错误。
    last_error: Exception | None = None
    # 至少执行一次；负数或零不会导致函数静默跳过。
    for _ in range(max(1, attempts)):
        # 每次尝试都独立捕获异常，成功则立即返回结果。
        try:
            # 调用业务函数并保持其泛型返回类型不变。
            return fn()
        # 捕获普通运行时异常，供下一轮尝试或最终重抛。
        except Exception as exc:
            # 记录本轮异常，后续失败会用更新、更接近最终状态的异常覆盖。
            last_error = exc
    # 循环至少执行一次且只有异常才会走到这里，因此最后异常理论上必然存在。
    assert last_error is not None
    # 重抛原始异常对象，保留异常类型和调用上下文供恢复层判断。
    raise last_error
