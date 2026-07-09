from __future__ import annotations

from agent_core.graph import nodes
from agent_core.graph.state import AgentState
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


def test_evaluator_optimizer_regenerates_at_most_once() -> None:
    """质量不合格时最多重生成 1 次。"""
    state = AgentState(input_text="需要证据的问题", answer="短")
    state.grounding_result = {"grounded": False}

    nodes.evaluate_response_quality(state)
    nodes.regenerate_response_if_needed(state)
    first_answer = state.answer
    nodes.regenerate_response_if_needed(state)

    assert state.regeneration_attempts == 1
    assert state.answer == first_answer
    assert "证据不足/已降级" in state.metadata.get("response_warnings", [])


def test_regeneration_is_followed_by_pii_grounding_and_compliance_again() -> None:
    """重生成后会再次跑 PII scan、grounding 和 compliance。"""
    response = WorkflowEngine().run(AgentRunRequest(input="今天上海天气"))

    if response.evaluation_result.get("regenerated"):
        assert sum(1 for item in response.guardrails if item.get("guardrail_name") == "output_pii_scan") >= 2
        assert sum(1 for item in response.guardrails if item.get("guardrail_name") == "insurance_output_compliance") >= 2
        assert any(event["event"] == "response_regenerated" for event in response.trace_events)
        assert response.grounding_result
