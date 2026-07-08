from agent_core.graph.nodes import analyze_kyc_and_route, load_business_memory, status_router
from agent_core.graph.state import AgentNode, AgentState
from agent_core.memory.business_schemas import KYCQuestion, OpportunityCase
from agent_core.memory.business_store import InMemoryBusinessMemoryStore
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine
from agent_core.workflow.steps import INSURANCE_KYC_COACH_CONTRACT, INSURANCE_KYC_COACH_STEPS


def test_insurance_kyc_coach_contract_contains_required_steps() -> None:
    """KYC 教练 workflow contract 显式包含 16 个业务步骤。"""
    step_names = [step.name for step in INSURANCE_KYC_COACH_CONTRACT.steps]

    assert step_names == INSURANCE_KYC_COACH_STEPS
    assert INSURANCE_KYC_COACH_CONTRACT.entry_state == AgentNode.INIT_CONTEXT
    assert AgentNode.FINAL in INSURANCE_KYC_COACH_CONTRACT.final_states


def test_load_business_memory_reads_asked_focuses_from_store() -> None:
    """asked_focuses 来自 KYCQuestion 表，节点不再依赖字符串拼接。"""
    store = InMemoryBusinessMemoryStore()
    case = OpportunityCase(tenant_id="tenant_a", advisor_id="advisor_a", customer_id="customer_a")
    store.upsert_opportunity_case(case)
    store.insert_kyc_question(
        KYCQuestion(
            tenant_id="tenant_a",
            opportunity_case_id=case.id,
            conversation_id="conv_a",
            round_no=1,
            focus_key="financial_preference",
            question_text="他平时更偏好哪类资金安排？",
        )
    )
    state = AgentState(
        tenant_id="tenant_a",
        session_id="conv_a",
        metadata={"advisor_id": "advisor_a", "customer_id": "customer_a", "conversation_id": "conv_a"},
    )

    load_business_memory(state, store)

    assert state.asked_focuses == ["financial_preference"]
    assert state.metadata["opportunity_case_id"] == case.id


def test_kyc_round_limit_forces_strategy_path_after_four_rounds() -> None:
    """第 5 轮后不能继续卡在 insufficient，而要基于现有信息输出策略。"""
    state = AgentState(
        tenant_id="tenant_a",
        input_text="目前就这些信息，客户只知道是企业主",
        kyc_question_round_count=4,
        profile_state={"occupation": "企业主"},
    )

    analyze_kyc_and_route(state)
    assert state.information_status == "matched"

    state.information_status = "insufficient"
    status_router(state)

    assert state.information_status == "matched"
    assert state.current_state == AgentNode.RETRIEVE_DIALOGUE_PATTERNS


def test_workflow_engine_runs_insurance_kyc_coach_branch() -> None:
    """显式 workflow_name 可以直接运行 KYC 教练业务记忆链路。"""
    store = InMemoryBusinessMemoryStore()
    engine = WorkflowEngine(business_store=store)

    response = engine.run(
        AgentRunRequest(
            input="我有个45岁企业主客户，两个孩子，喜欢银行理财，先给我初版策略",
            workflow_name="insurance_kyc_coach_workflow",
            tenant_id="tenant_a",
            session_id="conv_a",
            metadata={"advisor_id": "advisor_a", "customer_id": "customer_a", "conversation_id": "conv_a"},
        )
    )

    assert response.final_state == "FINAL"
    assert response.domain_skill == "insurance_advisor"
    assert "合规边界" in response.answer
    assert store.generated_outputs
    assert store.generated_outputs[0].input_context["customer_profile"]["confirmed"]
