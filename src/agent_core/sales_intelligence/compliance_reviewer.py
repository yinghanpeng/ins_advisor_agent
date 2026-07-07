"""Compliance review for sales insight cards."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from agent_core.sales_intelligence.schemas import SalesInsightCard


HIGH_RISK_TERMS = ["保证收益", "避债避税", "绝对安全", "谁都动不了", "一定成交", "制造焦虑"]
MEDIUM_RISK_TERMS = ["资产隔离", "税务安排", "收益率", "港保更安全"]


def review_card(card: SalesInsightCard) -> SalesInsightCard:
    # 重点逻辑：把策略、话术、错误方式、合规说明拼在一起统一扫描风险词。
    text = " ".join(
        [
            card.effective_strategy,
            card.usable_script,
            card.wrong_way,
            card.compliance_notes,
        ]
    )
    high_hits = [term for term in HIGH_RISK_TERMS if term in text]
    medium_hits = [term for term in MEDIUM_RISK_TERMS if term in text]
    if high_hits:
        # 重点逻辑：高风险卡片必须禁止生成，不能靠下游 Prompt 自觉避开。
        card.risk_level = "high"
        card.approved_for_generation = False
        card.compliance_notes = f"High risk terms: {','.join(high_hits)}"
    elif medium_hits:
        # 重点逻辑：中风险卡片进入人工复核，不直接用于最终话术生成。
        card.risk_level = "medium"
        card.approved_for_generation = False
        card.compliance_notes = f"Needs human review: {','.join(medium_hits)}"
    else:
        card.risk_level = "low"
    return card
