"""Text chunking helpers."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if not text:
        return list()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks
