# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.guardrails.output import OutputGuardrail
from agent_core.graph import nodes
from agent_core.graph.state import AgentNode, AgentState
from agent_core.sales_intelligence.retriever import SalesIntelligenceRetriever


def test_sales_retriever_returns_approved_cards():
    cards = SalesIntelligenceRetriever().retrieve("企业主破冰")
    assert cards
    assert all(card.approved_for_generation for card in cards)


def test_output_guardrail_blocks_high_risk_claim():
    result = OutputGuardrail().review("这个产品保证收益，而且绝对安全。")
    assert result["action"] == "block"
    assert result["triggered"] is True


def test_compliance_review_replaces_unsafe_output_without_pending_state():
    state = AgentState(input_text="介绍产品", answer="这个产品保证收益，而且绝对安全。")

    result = nodes.compliance_review(state)

    assert result.current_state == AgentNode.RESPONSE_PACKAGING
    assert result.final_state is None
    assert "无法提供保证收益" in (result.answer or "")
    assert result.metadata["output_policy_fallback"] is True
