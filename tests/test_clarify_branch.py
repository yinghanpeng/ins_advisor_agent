from __future__ import annotations

from agent_core.graph import nodes
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


def test_clarify_context_need_short_circuits_before_tool_rag_and_generation(monkeypatch) -> None:
    """context_needs.clarify=True 时直接返回澄清问题，不调用工具、RAG 或生成节点。"""
    original_context_need_planning = nodes.context_need_planning

    def force_clarify(state):
        state = original_context_need_planning(state)
        state.slot_values["missing_slots"] = ["customer_profile"]
        state.context_needs["clarify"] = True
        state.context_needs["tool"] = True
        state.context_needs["rag"] = True
        return state

    monkeypatch.setattr(nodes, "context_need_planning", force_clarify)
    monkeypatch.setattr(nodes, "agentic_tool_loop", lambda _state: (_ for _ in ()).throw(AssertionError("tool called")))
    monkeypatch.setattr(nodes, "retrieve_sales_intelligence", lambda _state: (_ for _ in ()).throw(AssertionError("rag called")))
    monkeypatch.setattr(nodes, "generate_response", lambda _state: (_ for _ in ()).throw(AssertionError("model called")))

    response = WorkflowEngine().run(AgentRunRequest(input="帮我处理一下这个客户"))

    assert response.final_state == "FINAL"
    assert response.intent == "clarify"
    assert response.response_package["clarification_question"]
    assert "客户背景" in response.answer
    assert response.tool_calls == []
    assert response.retrieved_context == []
