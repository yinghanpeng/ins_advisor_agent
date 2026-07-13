"""Sales insight evidence compression."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from agent_core.sales_intelligence.schemas import SalesInsightCard, SalesInsightDigest


def build_sales_insight_digest(cards: list[SalesInsightCard]) -> SalesInsightDigest:
    """把已筛选销售卡片压缩为生成所需的策略、话术、动作和来源摘要。"""
    # 仅投影结构化卡片字段并限制策略摘要长度，原始访谈全文不会进入 Prompt。
    return SalesInsightDigest(
        applicable_scene=", ".join(sorted({card.scene for card in cards})) or "unknown",
        insight_summary="；".join(card.effective_strategy for card in cards)[:1200],
        usable_scripts=[card.usable_script for card in cards if card.usable_script],
        forbidden_expressions=[card.wrong_way for card in cards if card.wrong_way],
        next_actions=[card.follow_up_action or card.next_question for card in cards],
        sources=[
            {
                "source_id": card.source_id,
                "chunk_id": card.chunk_id,
                "risk_level": card.risk_level,
            }
            for card in cards
        ],
        compliance_notes=[card.compliance_notes for card in cards if card.compliance_notes],
    )
