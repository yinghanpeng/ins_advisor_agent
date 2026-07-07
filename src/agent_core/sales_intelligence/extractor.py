"""Structured insight extraction.

Production extraction should call an LLM with JSON schema validation. The local
implementation is deterministic so tests and demos do not require model access.
"""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from agent_core.sales_intelligence.compliance_reviewer import review_card
from agent_core.sales_intelligence.schemas import CustomerKYC, SalesInsightCard
from agent_core.sales_intelligence.segmenter import InterviewSegment


def extract_structured_insight(segment: InterviewSegment, metadata: dict | None = None) -> SalesInsightCard:
    metadata = metadata or {}
    # 重点逻辑：当前是确定性抽取，字段结构与未来 LLM JSON 输出保持一致。
    card = SalesInsightCard(
        source_id=segment.source_id,
        chunk_id=segment.chunk_id,
        interviewee_role=metadata.get("interviewee_role", "unknown"),
        sales_experience_years=metadata.get("sales_experience_years"),
        channel=metadata.get("channel"),
        business_stage=metadata.get("business_stage", "unknown"),
        scene=segment.scene,
        customer_type=metadata.get("customer_type", "unknown"),
        customer_kyc=CustomerKYC(
            age=metadata.get("age"),
            family=metadata.get("family"),
            occupation=metadata.get("occupation"),
            asset_preference=metadata.get("asset_preference"),
            decision_style=metadata.get("decision_style"),
        ),
        sales_pain_solved="从访谈片段中抽取的销售痛点，需要人工复核",
        root_cause="从业者缺少场景化提问和低压推进结构",
        effective_strategy=segment.text[:220],
        usable_script="先认可客户处境，再追问资金用途和决策边界。",
        wrong_way="直接承诺收益或强推具体产品。",
        why_it_works="先建立共情，再把话题落到客户自己的资金安排。",
        next_question="这笔钱未来三到五年更可能承担什么责任？",
        tags=[segment.scene, "auto_extracted"],
        compliance_notes="auto extracted, requires review before production generation",
        approved_for_generation=False,
    )
    # 重点逻辑：抽取完成后立刻进入合规审查，不让未审查卡片进入索引。
    return review_card(card)
