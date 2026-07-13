"""代码化保险处理器的生产边界契约测试。"""

from agent_core.graph.state import AgentNode, AgentState
from agent_core.workflow.contracts import AgentRunResponse


def test_agent_state_declares_code_native_insurance_nodes() -> None:
    """保险逻辑必须表现为代码节点，而不是外部 Workflow Contract。"""
    assert AgentNode.EXTRACT_INSURANCE_KYC.value == "EXTRACT_INSURANCE_KYC"
    assert AgentNode.RETRIEVE_INSURANCE_KNOWLEDGE.value == "RETRIEVE_INSURANCE_KNOWLEDGE"
    state = AgentState(intent="insurance_break_ice", domain_skill="insurance_advisor")
    assert state.workflow_name == "universal_agent_workflow"


def test_public_response_exposes_safe_intent_and_kyc_summaries() -> None:
    """公开契约只返回路由/缺失字段摘要，不强制暴露客户槽位值。"""
    fields = AgentRunResponse.model_fields
    assert "intent_routing_result" in fields
    assert "active_intent" in fields
    assert "insurance_kyc_status" in fields
