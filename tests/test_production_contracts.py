# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.graph.state import AgentNode
from agent_core.workflow.steps import BREAK_ICE_ASSISTANT_CONTRACT


def test_break_ice_workflow_has_step_contracts():
    contract = BREAK_ICE_ASSISTANT_CONTRACT
    assert contract.entry_state == AgentNode.CLASSIFY_INTENT
    assert contract.final_states
    assert {step.name for step in contract.steps} >= {
        "classify_intent",
        "retrieve_sales_intelligence",
        "build_context",
        "generate_response",
        "compliance_review",
    }
    assert all(step.required_inputs or step.name == "classify_intent" for step in contract.steps)


def test_workflow_contract_declares_trace_and_guardrails():
    review_step = next(step for step in BREAK_ICE_ASSISTANT_CONTRACT.steps if step.name == "compliance_review")
    assert "insurance_output_compliance" in review_step.guardrails
    assert "trace_id" in review_step.trace_fields

