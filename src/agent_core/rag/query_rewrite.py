"""Query rewriting for retrieval."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def rewrite_sales_queries(user_input: str, sales_pain: str | None = None, scene: str | None = None) -> list[str]:
    """依据用户原话、销售痛点与场景生成多路检索查询。"""
    # 首路查询保留去除首尾空白后的用户原话，避免改写丢失核心诉求。
    queries = [user_input.strip()]
    # 存在结构化销售痛点时增加一条显式痛点查询，提高策略召回率。
    if sales_pain:
        # 添加领域前缀，帮助词法与向量检索聚焦销售问题。
        queries.append(f"销售痛点 {sales_pain}")
    # 存在沟通场景时增加场景查询，缩小话术适用范围。
    if scene:
        # 添加场景前缀，便于知识库 metadata 与正文共同匹配。
        queries.append(f"销售场景 {scene}")
    # 始终补充话术策略视角，覆盖用户未明确使用领域词汇的表达。
    queries.append(f"话术策略 {user_input.strip()}")
    # 过滤空字符串并保持原始顺序，返回可直接并行检索的查询列表。
    return [q for q in queries if q]
