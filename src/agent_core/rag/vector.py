"""词元重叠向量近似工具。

生产在线召回应使用 `agent_core.rag.production.ProductionRagRetriever` 和 pgvector；
这里保留给规则评测和轻量本地检索工具使用。
"""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations

import math
import re


def _tokens(text: str) -> set[str]:
    """将中英文文本规范化为去重词元集合。"""
    # 统一小写后抽取字母、数字、下划线和中文字符，供轻量相似度计算。
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def token_jaccard(query: str, document: str) -> float:
    """基于词元交集计算轻量相关性分数。"""
    # 抽取查询词元作为相似度计算左侧集合。
    left = _tokens(query)
    # 用同一规则抽取文档词元，避免预处理不一致造成偏差。
    right = _tokens(document)
    # 任一侧为空都无法形成有效交集，直接返回零并避免除零。
    if not left or not right:
        # 返回浮点零，保持评分接口类型稳定。
        return 0.0
    # 用集合交集除以两侧规模的几何平均，兼顾查询与文档长度差异。
    return len(left & right) / math.sqrt(len(left) * len(right))


class VectorRetriever:
    """Replaceable vector-search adapter."""

    def search(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """用 token_jaccard 生成稳定向量近似分数，并返回排序结果。"""
        # 为每个输入文档累计附加分数后的副本，不直接修改调用方对象。
        scored = []
        # 逐文档计算相同查询下的轻量向量近似分数。
        for doc in documents:
            # 将缺失正文视为空字符串，并统一转成文本以兼容非字符串值。
            text = str(doc.get("text", ""))
            # 合并原字段与 vector_score，供后续混合排序使用。
            scored.append({**doc, "vector_score": token_jaccard(query, text)})
        # 按向量分数降序截断，返回最多 top_k 条候选。
        return sorted(scored, key=lambda item: item.get("vector_score", 0), reverse=True)[:top_k]
