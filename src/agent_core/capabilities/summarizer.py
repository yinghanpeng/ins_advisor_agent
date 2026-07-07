"""Local summarizer adapter."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from __future__ import annotations


def run(arguments: dict) -> dict:
    """用本地截断方式生成摘要，保证无模型环境下也能测试工具链路。"""
    text = str(arguments.get("text", ""))
    max_chars = int(arguments.get("max_chars", 300))
    summary = text[:max_chars]
    return {"summary": summary, "truncated": len(text) > max_chars}
