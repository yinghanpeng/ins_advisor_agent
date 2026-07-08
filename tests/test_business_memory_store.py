from agent_core.memory.business_schemas import (
    CustomerProfileFact,
    GeneratedOutput,
    KYCQuestion,
)
from agent_core.memory.business_store import InMemoryBusinessMemoryStore


def _customer_fact(tenant_id: str, value: str) -> CustomerProfileFact:
    return CustomerProfileFact(
        tenant_id=tenant_id,
        customer_id="same_customer",
        fact_key="occupation",
        fact_value=value,
        source_type="user_message",
        evidence_text=f"用户明确说客户是{value}",
    )


def test_customer_facts_are_isolated_by_tenant() -> None:
    """相同 customer_id 在不同 tenant 下不能串事实。"""
    store = InMemoryBusinessMemoryStore()
    store.upsert_customer_fact(_customer_fact("tenant_a", "企业主"))
    store.upsert_customer_fact(_customer_fact("tenant_b", "高管"))

    tenant_a_facts = store.get_current_customer_facts("tenant_a", "same_customer")
    tenant_b_facts = store.get_current_customer_facts("tenant_b", "same_customer")

    assert [fact.fact_value for fact in tenant_a_facts] == ["企业主"]
    assert [fact.fact_value for fact in tenant_b_facts] == ["高管"]


def test_long_term_fact_must_have_evidence() -> None:
    """没有 evidence_text 的事实不能写入长期事实表。"""
    store = InMemoryBusinessMemoryStore()
    fact = CustomerProfileFact(
        tenant_id="tenant_a",
        customer_id="customer_a",
        fact_key="occupation",
        fact_value="企业主",
        source_type="user_message",
        evidence_text=" ",
    )

    try:
        store.upsert_customer_fact(fact)
    except ValueError as exc:
        assert "evidence_text" in str(exc)
    else:
        raise AssertionError("缺少 evidence_text 的事实不应写入成功")


def test_conflicting_fact_closes_old_version() -> None:
    """同 key 新旧事实冲突时关闭旧版本，而不是覆盖删除。"""
    store = InMemoryBusinessMemoryStore()
    old_fact = store.upsert_customer_fact(_customer_fact("tenant_a", "企业主"))
    new_fact = store.upsert_customer_fact(_customer_fact("tenant_a", "职业经理人"))

    assert old_fact.is_current is False
    assert old_fact.valid_to is not None
    assert new_fact.is_current is True
    current = store.get_current_customer_facts("tenant_a", "same_customer")
    assert [fact.fact_value for fact in current] == ["职业经理人"]


def test_asked_focuses_are_read_from_kyc_questions_without_duplicates() -> None:
    """已问焦点来自 KYCQuestion 记录，重复 focus 不会被重复追问。"""
    store = InMemoryBusinessMemoryStore()
    question = KYCQuestion(
        tenant_id="tenant_a",
        opportunity_case_id="case_a",
        conversation_id="conv_a",
        round_no=1,
        focus_key="financial_preference",
        question_text="他平时更偏好哪类资金安排？",
    )
    store.insert_kyc_question(question)
    store.insert_kyc_question(question.model_copy(update={"id": "another_id"}))

    assert store.get_asked_focuses("tenant_a", "case_a") == ["financial_preference"]


def test_generated_output_records_context_and_pattern_ids() -> None:
    """GeneratedOutput 记录输出类型、compact_context 和使用过的模式 ID。"""
    store = InMemoryBusinessMemoryStore()
    output = GeneratedOutput(
        tenant_id="tenant_a",
        conversation_id="conv_a",
        opportunity_case_id="case_a",
        output_type="strategy",
        input_context={"customer_profile": {"confirmed": {"occupation": "企业主"}}},
        output_text="先低压确认资金用途。",
        used_case_pattern_ids=["pattern_1"],
    )

    store.insert_generated_output(output)

    assert store.generated_outputs[0].output_type == "strategy"
    assert store.generated_outputs[0].used_case_pattern_ids == ["pattern_1"]
    assert store.generated_outputs[0].input_context["customer_profile"]["confirmed"]["occupation"] == "企业主"
