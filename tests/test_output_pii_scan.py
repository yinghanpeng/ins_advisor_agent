from __future__ import annotations

from agent_core.graph import nodes
from agent_core.graph.state import AgentState


def test_output_pii_scan_redacts_phone_email_and_id_card() -> None:
    """输出侧 PII 扫描能脱敏手机号、邮箱和身份证。"""
    state = AgentState()
    state.answer = "客户电话 13800138000，邮箱 a@example.com，身份证 11010519491231002X。"

    result = nodes.output_pii_scan(state)

    assert "13800138000" not in result.answer
    assert "a@example.com" not in result.answer
    assert "11010519491231002X" not in result.answer
    assert set(result.output_pii_scan_result["pii_types"]) >= {"phone", "email", "id_card"}
    assert result.output_pii_scan_result["high_sensitivity"] is True
    assert result.risk_level == "high"


def test_output_pii_scan_does_not_write_raw_pii_to_public_trace() -> None:
    """高敏 PII 不会以原文留在 trace_events 或 stream_events。"""
    state = AgentState()
    state.answer = "银行卡 6222020202020202020，手机号 13800138000。"
    state.add_trace_event("unsafe_debug", answer=state.answer)
    nodes.emit_stream_event(state, "final_answer", {"node_name": "test", "answer": state.answer})

    result = nodes.output_pii_scan(state)

    public_dump = str(result.trace_events) + str(result.stream_events)
    assert "6222020202020202020" not in public_dump
    assert "13800138000" not in public_dump
    assert "bank_card" in str(result.output_pii_scan_result)
