"""Generic retrieval facade with metadata-aware hybrid search."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations

from agent_core.rag.bm25 import score
from agent_core.rag.reranker import combine_scores, rerank
from agent_core.rag.schemas import (
    DocumentMetadata,
    MetadataFilter,
    RetrievalDocument,
    RetrievalQuery,
    RetrievalResult,
)
from agent_core.rag.vector import token_jaccard


RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


class InMemoryRetriever:
    def __init__(self, documents: list[dict] | None = None) -> None:
        """初始化本地文档列表，用于轻量单元测试。"""
        # documents 是测试注入的原始 dict 文档；为空时使用空集合，保证检索器可安全实例化。
        self.documents = documents or []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """使用本地词法分数检索文档，并返回排序后的 dict 结果。"""
        # scored 保存每个文档及其词法相关性分数。
        scored = []
        # 遍历本地文档集合，为每个文档计算 query 和 text 的匹配分。
        for doc in self.documents:
            # 本地简单检索只用文本分数，适合单元测试，不代表生产检索质量。
            text = str(doc.get("text", ""))
            # 把 score 写回 dict，便于 rerank 函数排序。
            scored.append({**doc, "score": score(query, text)})
        # 返回排序后的 TopK 结果。
        return rerank(scored, top_k=top_k)


class HybridRetriever:
    """本地混合检索器，融合词法、向量近似和 metadata 分数。"""

    def __init__(self, documents: list[RetrievalDocument] | None = None) -> None:
        """初始化可检索文档集合；生产环境可替换为向量库或搜索服务。"""
        # documents 保存结构化 RetrievalDocument，包含 text 和 metadata。
        self.documents = documents or []

    @classmethod
    def from_dicts(cls, documents: list[dict]) -> "HybridRetriever":
        """Build a retriever from dictionaries used in tests and adapters."""
        # parsed 保存转换后的 RetrievalDocument 列表。
        parsed = []
        # 将外部 dict 逐条归一化，避免检索阶段到处处理缺省字段。
        for index, doc in enumerate(documents):
            # 外部传入的 dict 必须归一化为 RetrievalDocument，避免下游字段缺失。
            metadata = doc.get("metadata") or {}
            # 为缺失 source_id/chunk_id/library/tenant_id 的文档补默认值，保证 trace 可溯源。
            parsed.append(
                RetrievalDocument(
                    text=str(doc.get("text", "")),
                    metadata=DocumentMetadata(
                        source_id=str(metadata.get("source_id", f"source_{index}")),
                        chunk_id=str(metadata.get("chunk_id", f"chunk_{index}")),
                        library=str(metadata.get("library", "generic")),
                        tenant_id=str(metadata.get("tenant_id", "local")),
                        tags=list(metadata.get("tags", [])),
                        risk_level=metadata.get("risk_level", "low"),
                        approved_for_generation=bool(metadata.get("approved_for_generation", True)),
                        extra=dict(metadata.get("extra", {})),
                    ),
                )
            )
        # 用结构化文档创建 HybridRetriever。
        return cls(parsed)

    def _metadata_allowed(self, document: RetrievalDocument, filters: MetadataFilter | None) -> bool:
        """根据租户、知识库、审批状态、风险和标签判断文档是否允许进入候选集。"""
        # 没有 filters 时默认允许，方便本地测试不配置完整检索约束。
        if filters is None:
            return True
        # 取出文档 metadata，后续所有过滤都基于它。
        metadata = document.metadata
        # 租户过滤放在最前面，生产环境不能跨租户检索资料。
        if filters.tenant_id and metadata.tenant_id != filters.tenant_id:
            return False
        # library 过滤保证销售智能检索不会混入其他知识库。
        if filters.libraries and metadata.library not in filters.libraries:
            return False
        # 默认只检索 approved_for_generation=true 的资料。
        if filters.approved_only and not metadata.approved_for_generation:
            return False
        # 高风险资料即使文本相关，也不能进入生成链路。
        if RISK_ORDER[metadata.risk_level] > RISK_ORDER[filters.max_risk_level]:
            return False
        # required_tags 全部命中才允许返回，避免场景不匹配的 chunk 混入结果。
        if filters.required_tags and not set(filters.required_tags).issubset(set(metadata.tags)):
            return False
        # 所有 metadata 约束都通过后，文档才进入打分阶段。
        return True

    def _metadata_score(self, document: RetrievalDocument, filters: MetadataFilter | None) -> float:
        """计算业务 metadata 加权分，用于和词法/向量分融合。"""
        # 没有 filters 时给中性分，避免 metadata 分完全为 0。
        if filters is None:
            return 0.5
        # score_value 表示业务约束匹配程度，不直接代表文本语义相关性。
        score_value = 0.0
        # tenant_id 命中时加分，优先返回当前租户资料。
        if filters.tenant_id and document.metadata.tenant_id == filters.tenant_id:
            score_value += 0.25
        # library 命中时加分，优先返回目标知识库资料。
        if filters.libraries and document.metadata.library in filters.libraries:
            score_value += 0.25
        # 标签有重叠时加分，越贴近当前场景分越高。
        if filters.required_tags:
            overlap = len(set(filters.required_tags) & set(document.metadata.tags))
            score_value += min(0.3, overlap * 0.15)
        # 低风险资料优先级更高，鼓励生成链路使用更安全的证据。
        if document.metadata.risk_level == "low":
            score_value += 0.2
        # metadata 分限制在 1.0 以内，便于和其他分数融合。
        return min(score_value, 1.0)

    def search(
        self,
        queries: list[RetrievalQuery] | str,
        filters: MetadataFilter | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """执行本地 hybrid search，并保留各类得分方便 trace。"""
        # 如果调用方只传字符串，包装成 RetrievalQuery，统一后续多 query 处理。
        if isinstance(queries, str):
            queries = [RetrievalQuery(text=queries)]
        # scored 用 source_id+chunk_id 去重，同一 chunk 被多个 query 命中时只保留最高分。
        scored: dict[tuple[str, str], RetrievalResult] = {}
        # 遍历每个 rewritten query，实现多 query 召回。
        for query in queries:
            # 遍历所有候选文档；真实生产环境会由搜索后端完成召回。
            for document in self.documents:
                # metadata 过滤先于打分，保证不合规/跨租户资料不会进入候选集。
                if not self._metadata_allowed(document, filters):
                    continue
                # lexical 代表 BM25/关键词匹配分，本地用轻量 score 模拟。
                lexical = score(query.text, document.text) * query.weight
                # vector 代表语义相似度，本地用 token_jaccard 近似，生产可替换向量数据库分数。
                vector = token_jaccard(query.text, document.text) * query.weight
                # metadata_score 表示租户、library、tag、risk 等业务约束匹配程度。
                metadata_score = self._metadata_score(document, filters)
                # combined 是融合分数，后续排序按它进行。
                combined = combine_scores(lexical, vector, metadata_score)
                # 用 source_id+chunk_id 作为唯一键，避免同一片段重复返回。
                key = (document.metadata.source_id, document.metadata.chunk_id)
                # previous 是该 chunk 之前被其他 rewritten query 命中的最好结果。
                previous = scored.get(key)
                # 如果当前 query 命中分更高，就用当前结果覆盖旧结果。
                if previous is None or combined > previous.score:
                    # 同一个 chunk 可能被多个 rewritten query 命中，只保留最高分版本。
                    scored[key] = RetrievalResult(
                        document=document,
                        lexical_score=lexical,
                        vector_score=vector,
                        metadata_score=metadata_score,
                        rerank_score=combined,
                        score=combined,
                    )
        # 按融合分从高到低排序，并截断 TopK 作为最终检索结果。
        return sorted(scored.values(), key=lambda item: item.score, reverse=True)[:top_k]
