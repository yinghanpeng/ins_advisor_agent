from agent_core.memory.business_schemas import CustomerProfileFact
from agent_core.memory.write_policy import MemoryWriteProposal, validate_memory_write_proposal


def _fact(
    fact_key: str,
    fact_value: str,
    *,
    evidence_text: str = "用户明确提供该事实",
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
        evidence_text=evidence_text,
    )


def test_memory_write_policy_blocks_fact_without_evidence() -> None:
    """长期事实必须有证据，不能靠模型猜测写入。"""
    proposal = MemoryWriteProposal(facts_to_upsert=[_fact("occupation", "企业主", evidence_text="")])

    result = validate_memory_write_proposal(proposal)

    assert result.is_valid is False
    assert "evidence_text" in result.errors[0]


def test_memory_write_policy_keeps_uncertain_fact_as_warning_not_confirmed() -> None:
    """不确定线索可以写入，但必须保持 uncertain 语义。"""
    fact = _fact("concerns", "可能担心长期锁定", certainty="uncertain")
    proposal = MemoryWriteProposal(facts_to_upsert=[fact])

    result = validate_memory_write_proposal(proposal)

    assert result.is_valid is True
    assert result.allowed_fact_ids == [fact.id]
    assert "uncertain" in result.warnings[0]


def test_memory_write_policy_blocks_pii_and_generated_advice_as_fact() -> None:
    """PII 和模型建议不能作为客户长期事实进入 prompt 记忆。"""
    pii_fact = _fact("phone", "13800000000", sensitivity_level="pii")
    advice_fact = _fact("next_best_action", "直接给客户发方案")
    proposal = MemoryWriteProposal(facts_to_upsert=[pii_fact, advice_fact])

    result = validate_memory_write_proposal(proposal)

    assert result.is_valid is False
    assert set(result.blocked_fact_ids) == {pii_fact.id, advice_fact.id}
    assert "PII" in " ".join(result.errors)
    assert "生成的建议" in " ".join(result.errors)
