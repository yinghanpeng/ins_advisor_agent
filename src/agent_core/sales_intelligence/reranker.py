"""Sales intelligence reranking."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from agent_core.sales_intelligence.schemas import SalesInsightCard


def rerank_sales_cards(cards: list[SalesInsightCard], query: str, top_k: int = 5) -> list[SalesInsightCard]:
    """按审批状态、风险等级和词法命中度对销售洞察卡片排序。"""

    def score(card: SalesInsightCard) -> tuple[int, int, int]:
        """给单张卡片生成稳定排序分数，优先保留已审批、低风险、相关内容。"""
        # 拼接结构化场景、客群、痛点和标签，作为轻量词法匹配语料。
        hay = " ".join([card.scene, card.customer_type, card.sales_pain_solved, *card.tags])
        # 逐字符统计查询命中数，作为稳定但轻量的本地相关性近似。
        lexical = sum(1 for token in query if token in hay)
        # 已通过静态生成准入的卡片获得第一优先级，不代表在线人工审批。
        approved = 1 if card.approved_for_generation else 0
        # 低风险卡片获得第二优先级，使安全材料在同类候选中靠前。
        low_risk = 1 if card.risk_level == "low" else 0
        # 返回元组供 Python 按准入、风险、词法相关性依次比较。
        return (approved, low_risk, lexical)

    # 按复合评分降序截断，返回最多 top_k 张结构化洞察卡片。
    return sorted(cards, key=score, reverse=True)[:top_k]
