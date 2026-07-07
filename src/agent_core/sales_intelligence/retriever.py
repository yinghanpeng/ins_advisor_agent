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
        self.cards_dir = Path(cards_dir)

    def _load_cards(self) -> list[SalesInsightCard]:
        """加载所有销售洞察卡片；本地空目录时返回 sample_card 保证 demo 可运行。"""
        cards = SalesInsightIndexer(self.cards_dir).load_all()
        # 重点逻辑：没有真实卡片时返回 sample_card，保证 main.py 和测试开箱可运行。
        return cards or [sample_card()]

    def retrieve(self, query: str, top_k: int = 5) -> list[SalesInsightCard]:
        """检索与用户问题相关的销售洞察，并过滤不适合生成的卡片。"""
        # 重点逻辑：Sales Intelligence 只允许检索 suitable_for_rag、非 high risk、已批准生成的卡片。
        candidates = [
            card
            for card in self._load_cards()
            if card.suitable_for_rag and card.risk_level != "high" and card.approved_for_generation
        ]
        card_by_key = {(card.source_id, card.chunk_id): card for card in candidates}
        # 重点逻辑：把业务卡片转换成通用 RetrievalDocument，复用统一 RAG 检索协议。
        documents = [
            RetrievalDocument(
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
            for card in candidates
        ]
        rewritten = [
            RetrievalQuery(text=text, purpose="original" if index == 0 else "strategy")
            for index, text in enumerate(rewrite_sales_queries(query))
        ]
        # 重点逻辑：metadata filter 再次限制 library/risk/approval，双重防止高风险卡片进入生成。
        results = HybridRetriever(documents).search(
            rewritten,
            filters=MetadataFilter(
                libraries=["sales_intelligence"],
                max_risk_level="medium",
                approved_only=True,
            ),
            top_k=top_k,
        )
        selected = [
            card_by_key[(result.document.metadata.source_id, result.document.metadata.chunk_id)]
            for result in results
        ]
        return selected
