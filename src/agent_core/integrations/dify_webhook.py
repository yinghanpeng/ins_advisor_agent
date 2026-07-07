"""Dify webhook schema helpers."""

# 文件说明：
# - 本文件属于外部集成层，负责 Dify、LangSmith 等系统的 adapter。
# - 外部服务不可用时应 graceful degradation。
from __future__ import annotations


def normalize_dify_payload(payload: dict) -> dict:
    return {
        "input": payload.get("query") or payload.get("input") or "",
        "session_id": payload.get("conversation_id") or "dify_session",
        "metadata": {"source": "dify", **payload.get("metadata", {})},
    }

