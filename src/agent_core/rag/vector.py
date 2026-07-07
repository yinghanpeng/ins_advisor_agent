"""Deterministic local vector retriever adapter.

This is not a production embedding model. It provides a stable local interface
for hybrid search tests and can be replaced by a real vector database adapter.
"""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations

import math
import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def token_jaccard(query: str, document: str) -> float:
    """Return a small deterministic semantic-ish score based on token overlap."""
    left = _tokens(query)
    right = _tokens(document)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


class VectorRetriever:
    """Replaceable vector-search adapter."""

    def search(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """用 token_jaccard 生成稳定向量近似分数，并返回排序结果。"""
        scored = []
        for doc in documents:
            text = str(doc.get("text", ""))
            scored.append({**doc, "vector_score": token_jaccard(query, text)})
        return sorted(scored, key=lambda item: item.get("vector_score", 0), reverse=True)[:top_k]
