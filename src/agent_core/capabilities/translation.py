"""Translation adapter placeholder."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations


def run(arguments: dict) -> dict:
    """返回本地翻译占位结果；生产环境可替换为模型或翻译服务。"""
    return {
        "translated_text": arguments.get("text", ""),
        "source_language": arguments.get("source_language", "auto"),
        "target_language": arguments.get("target_language", "zh"),
        "mode": "mock",
    }
