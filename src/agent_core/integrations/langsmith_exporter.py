"""LangSmith trace exporter."""

# 文件说明：
# - 本文件属于外部集成层，负责 Dify、LangSmith 等系统的 adapter。
# - 外部服务不可用时应 graceful degradation。
from __future__ import annotations

import os

import httpx


def export_trace(trace: dict) -> dict:
    """把 trace 发送到配置的 LangSmith endpoint。"""
    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("LANGSMITH_ENDPOINT 或 LANGSMITH_API_KEY 未配置，无法导出 trace")
    response = httpx.post(
        f"{endpoint.rstrip('/')}/runs",
        headers={"Authorization": f"Bearer {api_key}"},
        json=trace,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("LangSmith 返回的 JSON 顶层不是对象")
    return data
