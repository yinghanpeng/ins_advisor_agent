"""Sales Intelligence retrieval.

This retriever deliberately searches reviewed insight cards, not raw interview
transcripts. It rewrites the user query, applies metadata filters, runs local
hybrid retrieval, then maps selected chunks back to approved cards.
"""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from pathlib import Path

from agent_core.rag.query_rewrite import rewrite_sales_queries
from agent_core.rag.retriever import HybridRetriever
from agent_core.rag.schemas import DocumentMetadata, MetadataFilter, RetrievalDocument, RetrievalQuery
from agent_core.sales_intelligence.indexer import SalesInsightIndexer
from agent_core.sales_intelligence.schemas import SalesInsightCard, sample_card


class SalesIntelligenceRetriever:
    """销售智能检索器，只检索已审核销售卡片，绝不直接返回原始访谈。"""

    def __init__(self, cards_dir: str | Path = "data/sales_insight_cards") -> None:
        """初始化销售洞察卡片目录；没有真实数据时会使用安全样例卡片。"""
        # cards_dir 是结构化销售洞察卡片目录，生产可挂载对象存储或数据库导出目录。
        self.cards_dir = Path(cards_dir)

    def _load_cards(self) -> list[SalesInsightCard]:
        """加载所有销售洞察卡片；本地空目录时返回 sample_card 保证 demo 可运行。"""
        # SalesInsightIndexer 负责从 cards_dir 读取 JSON 卡片并转成 SalesInsightCard。
        cards = SalesInsightIndexer(self.cards_dir).load_all()
        # 没有真实卡片时返回 sample_card，保证 main.py 和测试开箱可运行。
        return cards or [sample_card()]

    def retrieve(self, query: str, top_k: int = 5) -> list[SalesInsightCard]:
        """检索与用户问题相关的销售洞察，并过滤不适合生成的卡片。"""
        # Sales Intelligence 只允许检索 suitable_for_rag、非 high risk、已批准生成的卡片。
        candidates = [
            card
            # 从所有加载卡片中逐条筛选安全候选。
            for card in self._load_cards()
            # suitable_for_rag 控制卡片是否适合作为检索资料；high risk 和未审批卡片不能进入生成。
            if card.suitable_for_rag and card.risk_level != "high" and card.approved_for_generation
        ]
        # 建立 source_id/chunk_id 到卡片对象的映射，检索返回 chunk 后可映射回原始卡片。
        card_by_key = {(card.source_id, card.chunk_id): card for card in candidates}
        # 把业务卡片转换成通用 RetrievalDocument，复用统一 RAG 检索协议。
        documents = [
            RetrievalDocument(
                # text 拼接场景、客户类型、痛点、策略、话术和下一问，作为检索正文。
                text=" ".join(
                    [
                        card.scene,
                        card.customer_type,
                        card.sales_pain_solved,
                        card.effective_strategy,
                        card.usable_script,
                        card.next_question,
                    ]
                ),
                # metadata 保存销售智能库、标签、风险、审批和额外业务字段。
                metadata=DocumentMetadata(
                    source_id=card.source_id,
                    chunk_id=card.chunk_id,
                    library="sales_intelligence",
                    tags=card.tags,
                    risk_level=card.risk_level,
                    approved_for_generation=card.approved_for_generation,
                    extra={"scene": card.scene, "customer_type": card.customer_type},
                ),
            )
            # 每张安全候选卡片都会转换成一个可检索文档。
            for card in candidates
        ]
        # 对用户 query 做销售场景 query rewrite，第一条是原始 query，后续是策略/场景增强 query。
        rewritten = [
            RetrievalQuery(text=text, purpose="original" if index == 0 else "strategy")
            # enumerate 用于区分原始 query 和改写 query 的 purpose。
            for index, text in enumerate(rewrite_sales_queries(query))
        ]
        # metadata filter 再次限制 library/risk/approval，双重防止高风险卡片进入生成。
        results = HybridRetriever(documents).search(
            rewritten,
            filters=MetadataFilter(
                # 只允许 sales_intelligence 知识库，防止混入其他知识库 chunk。
                libraries=["sales_intelligence"],
                # 最高允许 medium 风险，高风险销售内容不能进入候选。
                max_risk_level="medium",
                # 只返回 approved_for_generation=True 的卡片。
                approved_only=True,
            ),
            # 控制最终返回卡片数量，避免上下文过长。
            top_k=top_k,
        )
        # 将 RetrievalResult 映射回 SalesInsightCard，让下游保留完整销售洞察字段。
        selected = [
            card_by_key[(result.document.metadata.source_id, result.document.metadata.chunk_id)]
            # 逐条处理检索命中的 chunk。
            for result in results
        ]
        # 返回已审核、已过滤、按相关性排序后的销售洞察卡片。
        return selected
