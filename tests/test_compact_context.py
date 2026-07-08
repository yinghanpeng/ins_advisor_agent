from agent_core.memory.business_schemas import (
    AdvisorProfileFact,
    CustomerProfileFact,
    OpportunityCase,
)
from agent_core.memory.compact_context import build_compact_context
from agent_core.sales_intelligence.schemas import DialoguePattern


def _customer_fact(
    fact_key: str,
    fact_value: str,
    *,
    certainty: str = "confirmed",
    sensitivity_level: str = "internal",
) -> CustomerProfileFact:
    return CustomerProfileFact(
        tenant_id="tenant_a",
        customer_id="customer_a",
        fact_key=fact_key,
        fact_value=fact_value,
        certainty=certainty,  # type: ignore[arg-type]
        sensitivity_level=sensitivity_level,  # type: ignore[arg-type]
        source_type="user_message",
        evidence_text=f"用户明确提供 {fact_key}",
    )


def test_compact_context_filters_pii_and_separates_certainty() -> None:
    """compact_context 不包含 PII，且 confirmed/uncertain 分区清晰。"""
    confirmed = [
        _customer_fact("occupation", "企业主"),
        _customer_fact("phone", "13800000000", sensitivity_level="pii"),
    ]
    uncertain = [_customer_fact("concerns", "可能担心长期锁定", certainty="uncertain")]
    advisor_facts = [
        AdvisorProfileFact(
            tenant_id="tenant_a",
            advisor_id="advisor_a",
            fact_key="career_stage",
            fact_value="newbie",
            source_type="user_message",
            evidence_text="用户说自己刚做保险",
        )
    ]

    context = build_compact_context(
        confirmed_customer_facts=confirmed,
        uncertain_customer_facts=uncertain,
        advisor_facts=advisor_facts,
        opportunity_case=OpportunityCase(tenant_id="tenant_a", advisor_id="advisor_a", customer_id="customer_a"),
        kyc_completeness_score=42,
        opportunity_score=58,
        external_grade="C",
        asked_focuses=["financial_preference"],
        missing_fields=["available_long_term_funds"],
        support_note="先补一个关键点。",
    )

    assert context["customer_profile"]["confirmed"] == {"occupation": "企业主"}
    assert context["customer_profile"]["uncertain"] == {"concerns": "可能担心长期锁定"}
    assert "phone" not in str(context)
    assert context["advisor_profile"]["career_stage"] == "newbie"


def test_compact_context_only_allows_approved_non_high_risk_patterns() -> None:
    """未审批或高风险 DialoguePattern 不进入 compact_context。"""
    safe_pattern = DialoguePattern(
        tenant_id="tenant_a",
        pattern_type="kyc_question",
        situation_summary="先确认资金用途。",
        recommended_move="只补问一个最关键字段。",
        approved_for_generation=True,
        risk_level="low",
    )
    blocked_pattern = DialoguePattern(
        tenant_id="tenant_a",
        pattern_type="product_bridge",
        situation_summary="未审模式。",
        recommended_move="直接讲收益。",
        approved_for_generation=False,
        risk_level="low",
    )
    high_pattern = DialoguePattern(
        tenant_id="tenant_a",
        pattern_type="product_bridge",
        situation_summary="高风险模式。",
        recommended_move="保证收益。",
        approved_for_generation=True,
        risk_level="high",
    )

    context = build_compact_context(
        confirmed_customer_facts=[],
        uncertain_customer_facts=[],
        advisor_facts=[],
        opportunity_case=None,
        kyc_completeness_score=0,
        opportunity_score=0,
        external_grade="D",
        asked_focuses=[],
        missing_fields=[],
        support_note="",
        retrieved_dialogue_patterns=[safe_pattern, blocked_pattern, high_pattern],
    )

    assert [item["id"] for item in context["retrieved_patterns"]] == [safe_pattern.id]
    assert "保证收益" not in str(context)
