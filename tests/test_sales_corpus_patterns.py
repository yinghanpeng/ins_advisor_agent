from agent_core.sales_intelligence.retriever import build_dialogue_pattern_digest
from agent_core.sales_intelligence.schemas import CorpusMessage, DialoguePattern


def test_unapproved_dialogue_pattern_does_not_enter_generation_digest() -> None:
    """未通过生成准入的模式不能进入最终生成摘要。"""
    approved = DialoguePattern(
        tenant_id="tenant_a",
        pattern_type="objection_handling",
        situation_summary="客户担心长期锁定。",
        recommended_move="先确认资金用途，再讨论流动性边界。",
        approved_for_generation=True,
        risk_level="low",
    )
    unapproved = DialoguePattern(
        tenant_id="tenant_a",
        pattern_type="objection_handling",
        situation_summary="未审模式。",
        recommended_move="直接推动成交。",
        approved_for_generation=False,
        risk_level="low",
    )

    digest = build_dialogue_pattern_digest([approved, unapproved])

    assert [item["id"] for item in digest] == [approved.id]
    assert "直接推动成交" not in str(digest)


def test_raw_corpus_message_does_not_enter_dialogue_pattern_digest() -> None:
    """原始脱敏消息只能做抽取/评测来源，不进入最终模式摘要。"""
    message = CorpusMessage(
        tenant_id="tenant_a",
        corpus_case_id="case_a",
        seq_no=1,
        speaker_role="customer",
        content_redacted="我以前有个真实客户的完整对话内容。",
    )
    pattern = DialoguePattern(
        tenant_id="tenant_a",
        pattern_type="kyc_question",
        situation_summary="抽象后的客户有长期资金疑问。",
        recommended_move="只问一个资金用途问题。",
        approved_for_generation=True,
        risk_level="low",
        source_corpus_case_ids=[message.corpus_case_id],
    )

    digest = build_dialogue_pattern_digest([pattern])

    assert message.content_redacted not in str(digest)
    assert digest[0]["situation_summary"] == "抽象后的客户有长期资金疑问。"
