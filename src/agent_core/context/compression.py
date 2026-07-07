"""Context compression helpers."""

# 文件说明：
# - 本文件属于 Context Engineering 层，负责上下文压缩、证据边界和生成输入。
# - 外部网页、文件、RAG、销售访谈都只能作为 evidence。
from __future__ import annotations


def truncate_context(text: str, max_chars: int = 2000) -> str:
    """按字符数截断上下文，作为本地 token budget 控制的轻量实现。"""
    # 本地 demo 不依赖 tokenizer，因此用字符数做近似；生产可替换为模型 tokenizer 精确裁剪。
    return text[:max_chars]
