"""FastAPI application adapter."""

# 文件说明：
# - 本文件属于 Agent Gateway 适配层，负责 API 请求、路由或中间件边界。
# - 生产环境可在这一层补鉴权、限流、租户隔离和请求校验。
from __future__ import annotations

from agent_core.api.routes import build_router


def create_app():
    """Create the FastAPI app when FastAPI is installed."""
    try:
        from fastapi import FastAPI
    except Exception as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install with `pip install -e \".[api]\"`."
        ) from exc
    app = FastAPI(title="Production Agent Framework", version="0.1.0")
    router = build_router()
    if router:
        app.include_router(router)
    return app


try:
    app = create_app()
except RuntimeError:
    app = None

