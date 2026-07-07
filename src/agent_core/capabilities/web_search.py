"""Web search adapter placeholder."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations


def run(arguments: dict) -> dict:
    """返回联网搜索的占位结果；生产环境应接入受控搜索 provider。"""
    return {"query": arguments.get("query", ""), "results": [], "mode": "provider_not_configured"}
