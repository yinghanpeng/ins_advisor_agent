"""LangSmith export adapter placeholder."""

# 文件说明：
# - 本文件属于外部集成层，负责 Dify、LangSmith 等系统的 adapter。
# - 外部服务不可用时应 graceful degradation。
from __future__ import annotations


def export_trace(trace: dict) -> dict:
    return {"exported": False, "reason": "remote export adapter not configured", "trace": trace}

