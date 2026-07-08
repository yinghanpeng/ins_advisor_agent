"""轻量词法相关性评分。"""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def score(query: str, document: str) -> float:
    query_terms = set(query.lower().split())
    doc_terms = set(document.lower().split())
    if not query_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)
