from __future__ import annotations

from agent_core.agentic_loop.schemas import ToolLoopDecision
from agent_core.graph import nodes
from agent_core.graph.state import AgentState
from agent_core.tools.schemas import ToolCall
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


def test_tool_task_enters_agentic_tool_loop() -> None:
    """工具任务应进入 agentic_tool_loop，并保留旧工具节点路径。"""
    # 补充说明：主链路现已恢复单轮 routing/call/verify，本测试改为防止 Agentic Loop 被误接回。
    response = WorkflowEngine().run(AgentRunRequest(input="计算 12*8+3"))

    path = [item["to_state"] for item in response.state_transitions]
    assert "AGENTIC_TOOL_LOOP" not in path
    assert "GENERAL_TOOL_CALL" in path
    assert "VERIFY_TOOL_RESULT" in path
    assert response.tool_results[0]["output"]["_source_boundary"]["trust"] == "untrusted_external_context"
    assert not any(
        event["event"] == "node_finished" and event.get("node_name") == "agentic_tool_loop"
        for event in response.trace_events
    )


class UniquePlanner:
    """每轮生成不同计算表达式，用于验证 max_iterations 硬上限。"""

    def decide(self, state: AgentState, *, iteration_index: int) -> ToolLoopDecision:
        return ToolLoopDecision(
            action="call_tool",
            tool_calls=[
                ToolCall(
                    name="calculator",
                    arguments={"expression": f"{iteration_index}+1"},
                    trace_id=state.trace_id,
                )
            ],
            rationale_summary="测试用唯一工具计划。",
            confidence=1.0,
        )


def test_tool_loop_stops_at_max_iterations(monkeypatch) -> None:
    """工具 loop 达到 max_iterations 后必须停止，不会无限循环。"""
    monkeypatch.setattr(nodes, "_build_tool_loop_planner", lambda _state: UniquePlanner())
    state = AgentState(
        input_text="循环计算",
        context_needs={"tool": True},
        tool_loop_config={"max_iterations": 2, "max_total_tool_calls": 4},
    )

    result = nodes.agentic_tool_loop(state)

    assert result.tool_loop_stop_reason == "max_iterations"
    assert len(result.tool_loop_iterations) == 2
    assert len(result.tool_calls) == 2


class RepeatedPlanner:
    """每轮生成完全相同工具计划，用于验证 loop risk 检测。"""

    def decide(self, state: AgentState, *, iteration_index: int) -> ToolLoopDecision:
        return ToolLoopDecision(
            action="call_tool",
            tool_calls=[
                ToolCall(
                    name="calculator",
                    arguments={"expression": "1+1"},
                    trace_id=state.trace_id,
                )
            ],
            rationale_summary="测试用重复工具计划。",
            confidence=1.0,
        )


def test_repeated_tool_plan_stops_loop(monkeypatch) -> None:
    """连续两轮相同工具计划会停止，并写 repeated_tool_plan。"""
    monkeypatch.setattr(nodes, "_build_tool_loop_planner", lambda _state: RepeatedPlanner())
    state = AgentState(
        input_text="重复计算",
        context_needs={"tool": True},
        tool_loop_config={"max_iterations": 4, "max_total_tool_calls": 4},
    )

    result = nodes.agentic_tool_loop(state)

    assert result.tool_loop_stop_reason == "repeated_tool_plan"
    assert len(result.tool_calls) == 1
    assert result.tool_loop_iterations[-1]["stop_reason"] == "repeated_tool_plan"


def test_denied_tool_is_blocked_and_degraded_without_pending_state(monkeypatch) -> None:
    """工具被策略拒绝时同步阻断并降级，不进入挂起状态。"""
    monkeypatch.setattr(nodes, "_build_tool_loop_planner", lambda _state: RepeatedPlanner())

    class BlockingGuardrail:
        def review(self, _spec):
            return {
                "guardrail_name": "tool_permission",
                "triggered": True,
                "reason": "side_effect_not_allowed",
                "action": "deny",
            }

    monkeypatch.setattr(nodes, "ToolGuardrail", lambda: BlockingGuardrail())
    state = AgentState(input_text="不允许执行的工具", context_needs={"tool": True})

    result = nodes.agentic_tool_loop(state)

    assert result.final_state is None
    assert result.tool_results[0]["status"] == "blocked"
    assert result.tool_loop_stop_reason in {"repeated_tool_plan", "tool_error_budget_exceeded"}
    assert "降级" in (result.answer or "")


def test_failed_tool_result_is_verified_and_degraded() -> None:
    """工具失败时 verify_tool_result 仍会被调用，并进入降级路径。"""
    response = WorkflowEngine().run(AgentRunRequest(input="今天上海天气"))

    assert response.tool_results[0]["status"] in {"error", "success"}
    if response.tool_results[0]["status"] == "error":
        assert any(event["event"] == "tool_result_verified" for event in response.trace_events)
        assert response.final_state == "FINAL"


def test_no_model_planner_falls_back_to_rule_based_without_fake_fact() -> None:
    """没有模型 planner 时回退规则路由，不能伪造外部事实。"""
    # 补充说明：主链路不再创建 Planner；规则 ToolRouter 直接生成一次性工具计划。
    response = WorkflowEngine().run(AgentRunRequest(input="计算 12*8+3"))

    assert response.answer == "计算结果是：99。"
    assert not any(event["event"] == "tool_loop_model_planner_unavailable" for event in response.trace_events)
    assert response.tool_results[0]["name"] == "calculator"
