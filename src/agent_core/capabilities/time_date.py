"""Time/date capability."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations

from agent_core.utils.time import utc_now_iso


def run(_: dict | None = None) -> dict[str, str]:
    """返回当前 UTC 时间，作为无外部依赖的时间工具。"""
    # 在返回时实时读取 UTC 时间，避免模块加载时间被错误地当作请求时间。
    return {"utc_time": utc_now_iso()}
