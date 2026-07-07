"""Query rewriting for retrieval."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def rewrite_sales_queries(user_input: str, sales_pain: str | None = None, scene: str | None = None) -> list[str]:
    queries = [user_input.strip()]
    if sales_pain:
        queries.append(f"销售痛点 {sales_pain}")
    if scene:
        queries.append(f"销售场景 {scene}")
    queries.append(f"话术策略 {user_input.strip()}")
    return [q for q in queries if q]

