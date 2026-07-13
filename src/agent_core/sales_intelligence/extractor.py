"""结构化销售洞察抽取。

本模块把脱敏后的访谈分段转换成 SalesInsightCard，并立即进入合规审查。
生产接入 LLM 时应保持同一个 Pydantic schema 和审查边界。
"""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from agent_core.sales_intelligence.compliance_reviewer import review_card
from agent_core.sales_intelligence.schemas import CustomerKYC, SalesInsightCard
from agent_core.sales_intelligence.segmenter import InterviewSegment


def extract_structured_insight(segment: InterviewSegment, metadata: dict | None = None) -> SalesInsightCard:
    """把一个访谈分段抽取成 SalesInsightCard，并立即做合规审查。"""
    # metadata 承载采访对象、渠道、客户画像等外部信息；为空时用空字典兜底。
    metadata = metadata or {}
    # 字段结构与未来 LLM JSON 输出保持一致，方便统一审计和评测。
    card = SalesInsightCard(
        # source_id/chunk_id 保证这张洞察卡片可以回溯到原始访谈分段。
        source_id=segment.source_id,
        chunk_id=segment.chunk_id,
        # 受访者角色、年限、渠道、业务阶段来自外部 metadata。
        interviewee_role=metadata.get("interviewee_role", "unknown"),
        sales_experience_years=metadata.get("sales_experience_years"),
        channel=metadata.get("channel"),
        business_stage=metadata.get("business_stage", "unknown"),
        # scene 来自分段器的场景识别结果。
        scene=segment.scene,
        # customer_type 和 customer_kyc 影响后续检索和个性化回答。
        customer_type=metadata.get("customer_type", "unknown"),
        customer_kyc=CustomerKYC(
            age=metadata.get("age"),
            family=metadata.get("family"),
            occupation=metadata.get("occupation"),
            asset_preference=metadata.get("asset_preference"),
            decision_style=metadata.get("decision_style"),
        ),
        sales_pain_solved="从访谈片段中抽取的销售痛点，需要通过自动生成准入校验",
        root_cause="从业者缺少场景化提问和低压推进结构",
        # effective_strategy 暂取分段前 220 字，避免过长原文直接进入卡片。
        effective_strategy=segment.text[:220],
        # usable_script 必须经过合规审查后才允许进入生成链路。
        usable_script="先认可客户处境，再追问资金用途和决策边界。",
        # wrong_way 明确禁用直接承诺收益或强推产品。
        wrong_way="直接承诺收益或强推具体产品。",
        # why_it_works 说明策略背后的沟通机制。
        why_it_works="先建立共情，再把话题落到客户自己的资金安排。",
        # next_question 给出低压追问，便于从破冰进入 KYC。
        next_question="这笔钱未来三到五年更可能承担什么责任？",
        # tags 标记这张卡是自动抽取，后续只能由自动 Schema/风险/合规策略决定是否发布。
        tags=[segment.scene, "auto_extracted"],
        # 自动抽取结果默认不准入，禁止在客户请求中创建任何人工审批或等待任务。
        compliance_notes="auto extracted; generation gate defaults to deny",
        approved_for_generation=False,
    )
    # 抽取完成后立刻进入自动合规准入，不让未通过策略的卡片进入索引。
    return review_card(card)
