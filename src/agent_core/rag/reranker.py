"""Generic reranker."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def rerank(items: list[dict], top_k: int = 5) -> list[dict]:
    """按已有综合分数降序截取前 top_k 条候选。"""
    # 缺少 score 的候选按零分处理，排序后只返回调用方要求的上限。
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
    # 对词法、向量与 metadata 三路分数做线性加权，得到统一排序分数。
    return (
        lexical_score * lexical_weight
        + vector_score * vector_weight
        + metadata_score * metadata_weight
    )
