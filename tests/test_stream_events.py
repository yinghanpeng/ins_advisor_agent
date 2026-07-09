from __future__ import annotations

from agent_core.api.routes import run_agent_stream
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


def test_stream_events_include_node_tool_and_final_answer_events() -> None:
    """stream_events 至少包含节点、工具和最终答案事件。"""
    response = WorkflowEngine().run(AgentRunRequest(input="计算 12*8+3"))
    event_types = {event["event_type"] for event in response.stream_events}

    assert "node_started" in event_types
    assert "tool_call_started" in event_types
    assert "tool_call_finished" in event_types
    assert "final_answer" in event_types
    assert all(event["trace_id"] == response.trace_id for event in response.stream_events)


def test_run_agent_stream_returns_adapter_ready_event_package() -> None:
    """API 层 run_stream 骨架返回 stream_events，不破坏旧 run 接口。"""
    result = run_agent_stream(AgentRunRequest(input="计算 1+1"))

    assert result["trace_id"]
    assert result["stream_events"]
    assert result["final_response"]["answer"] == "计算结果是：2。"
