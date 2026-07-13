"""轻量词法相关性评分。"""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations


def score(query: str, document: str) -> float:
    """计算查询词元在文档词元中的轻量覆盖率。"""
    # 对查询统一小写并去重，避免同一词重复出现放大权重。
    query_terms = set(query.lower().split())
    # 文档采用相同的词元化规则，保证集合交集可比较。
    doc_terms = set(document.lower().split())
    # 空查询没有可计算的相关性，直接返回零并避免除零。
    if not query_terms:
        # 用浮点零保持评分接口的返回类型稳定。
        return 0.0
    # 以查询词元数为分母，返回文档覆盖查询的比例。
    return len(query_terms & doc_terms) / len(query_terms)
