"""Text chunking helpers."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """按固定窗口和重叠长度切分文本，供后续向量化与引用使用。"""
    # 空文本没有可入库内容，返回新的空列表避免制造空 chunk。
    if not text:
        # 显式构造列表，维持函数声明的 list[str] 契约。
        return list()
    # 按原文顺序累计切片，确保检索引用仍能还原文档顺序。
    chunks: list[str] = []
    # 首个窗口从原文起点开始。
    start = 0
    # 只要起点尚未到达原文末尾，就继续生成一个窗口。
    while start < len(text):
        # 窗口终点不越过文本长度，最后一个 chunk 可以短于 chunk_size。
        end = min(len(text), start + chunk_size)
        # 保存当前窗口原文，不在分块阶段改变字符内容。
        chunks.append(text[start:end])
        # 当前窗口已覆盖文末时结束，避免重叠逻辑重复生成尾块。
        if end == len(text):
            # 跳出循环并返回已经按序生成的全部 chunk。
            break
        # 下一窗口回退 overlap 个字符，保留跨边界语义；同时保证起点不为负数。
        start = max(0, end - overlap)
    # 返回可直接用于 embedding 与持久化的有序分块列表。
    return chunks
