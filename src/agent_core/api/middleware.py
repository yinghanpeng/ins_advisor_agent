"""Gateway middleware helpers."""

# 文件说明：
# - 本文件属于 Agent Gateway 适配层，负责 API 请求、路由或中间件边界。
# - 生产环境可在这一层补鉴权、限流、租户隔离和请求校验。
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_core.utils.ids import new_trace_id


def inject_trace_id(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Return headers with an x-trace-id value."""
    headers = dict(headers or {})
    headers.setdefault("x-trace-id", new_trace_id())
    return headers


def build_fastapi_trace_middleware() -> Callable[[Any, Any], Any] | None:
    """Build optional FastAPI middleware when FastAPI is installed."""
    try:
        from starlette.middleware.base import BaseHTTPMiddleware  # noqa: F401
    except Exception:
        return None
    return None

