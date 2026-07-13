from __future__ import annotations

from agent_core.graph import nodes
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


def test_clarify_context_need_short_circuits_before_tool_rag_and_generation(monkeypatch) -> None:
    """context_needs.clarify=True 时直接返回澄清问题，不调用工具、RAG 或生成节点。"""
    original_context_need_planning = nodes.context_need_planning

    def force_clarify(state):
        state = original_context_need_planning(state)
        state.metadata["missing_tool_arguments"] = ["query"]
        state.metadata["tool_argument_validation"] = {"tool_name": "web_search"}
        state.context_needs["clarify"] = True
        state.context_needs["tool"] = True
        state.context_needs["rag"] = True
        return state

    monkeypatch.setattr(nodes, "context_need_planning", force_clarify)
    monkeypatch.setattr(
        nodes,
        "agentic_tool_loop",
        lambda _state: (_ for _ in ()).throw(AssertionError("tool called")),
    )
    # 补充说明：当前主链路使用单轮工具节点，Clarify 必须同样在 routing/call 前短路。
    monkeypatch.setattr(
        nodes,
        "general_tool_routing",
        lambda _state: (_ for _ in ()).throw(AssertionError("tool routing called")),
    )
    monkeypatch.setattr(
        nodes,
        "general_tool_call",
        lambda _state: (_ for _ in ()).throw(AssertionError("tool call called")),
    )
    monkeypatch.setattr(
        nodes,
        "retrieve_sales_intelligence",
        lambda _state: (_ for _ in ()).throw(AssertionError("rag called")),
    )
    monkeypatch.setattr(
        nodes,
        "generate_response",
        lambda _state: (_ for _ in ()).throw(AssertionError("model called")),
    )

    response = WorkflowEngine().run(AgentRunRequest(input="请帮我简单介绍一下人工智能"))

    assert response.final_state == "FINAL"
    assert response.intent == "clarify"
    assert response.response_package["clarification_question"]
    assert "主题或关键词" in response.answer
    assert response.tool_calls == []
    assert response.retrieved_context == []


def test_tool_schema_missing_argument_clarifies_before_execution(monkeypatch) -> None:
    """工具选定后由自身 Schema 判缺参，不能调用执行器或依赖全局槽位。"""
    monkeypatch.setattr(
        nodes,
        "general_tool_call",
        lambda _state: (_ for _ in ()).throw(
            AssertionError("tool executed before schema clarification")
        ),
    )

    response = WorkflowEngine().run(AgentRunRequest(input="今天天气怎么样"))

    assert response.final_state == "FINAL"
    assert response.intent == "clarify"
    assert response.context_needs["clarify"] is True
    assert "城市或地区" in response.answer
    assert response.tool_calls == []
    path = [item["to_state"] for item in response.state_transitions]
    assert "GENERAL_TOOL_ROUTING" in path
    assert "GENERAL_TOOL_CALL" not in path
    assert "EXTRACT_SLOTS" not in path
    assert "VALIDATE_SLOTS" not in path
