"""Time helpers."""

# 文件说明：
# - 本文件属于工具函数层，提供 ID、时间、JSON 等通用辅助能力。
# - 工具函数应保持简单、可测试、无业务耦合。
from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""

    # 每次调用实时读取带时区 UTC 时间并转为 ISO-8601，便于跨服务排序和解析。
    return datetime.now(UTC).isoformat()
