"""Generic reranker."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def rerank(items: list[dict], top_k: int = 5) -> list[dict]:
    return sorted(items, key=lambda item: item.get("score", 0), reverse=True)[:top_k]


def combine_scores(
    lexical_score: float,
    vector_score: float,
    metadata_score: float,
    lexical_weight: float = 0.45,
    vector_weight: float = 0.35,
    metadata_weight: float = 0.20,
) -> float:
    """Weighted score used by local hybrid retrieval."""
    return (
        lexical_score * lexical_weight
        + vector_score * vector_weight
        + metadata_score * metadata_weight
    )
