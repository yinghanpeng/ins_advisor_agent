from agent_core.graph.nodes import analyze_kyc_and_route, load_business_memory, status_router
from agent_core.graph.state import AgentNode, AgentState
from agent_core.guardrails.metadata import TRUSTED_BUSINESS_IDENTITY_FLAG
from agent_core.memory.business_schemas import CustomerProfileFact, KYCQuestion, OpportunityCase
from agent_core.memory.business_store import InMemoryBusinessMemoryStore
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


class ConsentDeniedBusinessStore(InMemoryBusinessMemoryStore):
    """模拟生产 Store 在客户未授权 memory_processing 时拒绝创建 Case。"""

    def upsert_opportunity_case(self, case):
        del case
        raise PermissionError("memory_processing consent is required")


def test_load_business_memory_reads_asked_focuses_for_continued_active_task() -> None:
    """同一 active task 续接时从 KYCQuestion 表恢复已问焦点。"""
    store = InMemoryBusinessMemoryStore()
    case = OpportunityCase(tenant_id="tenant_a", advisor_id="advisor_a", customer_id="customer_a")
    store.upsert_opportunity_case(case)
    store.insert_kyc_question(
        KYCQuestion(
            tenant_id="tenant_a",
            opportunity_case_id=case.id,
            conversation_id="conv_a",
            round_no=1,
            focus_key="insurance_experience",
            question_text="这位客户以前接触或配置过保险吗？",
        )
    )
    state = AgentState(
        tenant_id="tenant_a",
        session_id="conv_a",
        intent="insurance_break_ice",
        intent_routing_result={"active_intent_action": "continued"},
        metadata={
            TRUSTED_BUSINESS_IDENTITY_FLAG: True,
            "advisor_id": "advisor_a",
            "customer_id": "customer_a",
            "conversation_id": "conv_a",
        },
    )

    load_business_memory(state, store)

    assert state.asked_focuses == ["insurance_experience"]
    assert state.metadata["opportunity_case_id"] == case.id


def test_new_insurance_intent_resets_old_task_questions_but_keeps_customer_facts() -> None:
    """created/replaced 意图不能继承旧三轮计数；客户明确事实仍跨任务保留。"""
    store = InMemoryBusinessMemoryStore()
    case = OpportunityCase(tenant_id="tenant_a", advisor_id="advisor_a", customer_id="customer_a")
    store.upsert_opportunity_case(case)
    store.insert_kyc_question(
        KYCQuestion(
            tenant_id="tenant_a",
            opportunity_case_id=case.id,
            conversation_id="conv_a",
            round_no=1,
            focus_key="insurance_experience",
            question_text="这位客户以前接触或配置过保险吗？",
        )
    )
    # 直接写入一条有用户证据的历史事实；公开请求不允许通过 metadata 选择 customer_id。
    store.upsert_customer_fact(
        CustomerProfileFact(
            tenant_id="tenant_a",
            customer_id="customer_a",
            fact_key="customer_role",
            fact_value="企业主",
            source_type="user_message",
            evidence_text="用户明确说他是企业主",
        )
    )
    state = AgentState(
        tenant_id="tenant_a",
        session_id="new_task_session",
        intent="insurance_objection_handling",
        intent_routing_result={"active_intent_action": "replaced"},
        metadata={
            TRUSTED_BUSINESS_IDENTITY_FLAG: True,
            "advisor_id": "advisor_a",
            "customer_id": "customer_a",
        },
    )

    load_business_memory(state, store)

    assert state.asked_focuses == []
    assert state.kyc_question_round_count == 0
    assert state.profile_state["customer_role"] == "企业主"
    assert state.metadata["opportunity_case_id"] != case.id


def test_kyc_round_limit_forces_strategy_path_after_configured_rounds() -> None:
    """达到配置的三轮上限后不能继续卡在 insufficient。"""
    state = AgentState(
        tenant_id="tenant_a",
        input_text="目前就这些信息，客户只知道是企业主",
        intent="insurance_strategy",
        kyc_question_round_count=3,
        profile_state={"customer_role": "企业主"},
        metadata={"max_kyc_question_rounds": 3},
    )

    analyze_kyc_and_route(state)
    assert state.information_status == "matched"

    state.information_status = "insufficient"
    status_router(state)

    assert state.information_status == "matched"
    assert state.current_state == AgentNode.RETRIEVE_DIALOGUE_PATTERNS


def test_engine_routes_insurance_to_code_handler_without_workflow_name() -> None:
    """普通请求自动进入保险代码处理器，不需要外部 workflow_name。"""
    store = InMemoryBusinessMemoryStore()
    engine = WorkflowEngine(business_store=store)

    response = engine.run(
        AgentRunRequest(
            input="我有个45岁企业主客户，两个孩子，喜欢银行理财，先给我初版策略",
            tenant_id="tenant_a",
            session_id="conv_a",
            user_id="advisor_a",
        )
    )

    assert response.final_state == "FINAL"
    assert response.domain_skill == "insurance_advisor"
    assert response.intent == "insurance_strategy"
    assert "合规边界" in response.answer
    assert store.generated_outputs
    assert store.generated_outputs[0].input_context["customer_profile"]["confirmed"]


def test_presented_kyc_question_is_persisted_after_generation() -> None:
    """问题只有真正生成到回答后才写入 asked 记录，轮次与 active pending focus 一致。"""
    store = InMemoryBusinessMemoryStore()
    response = WorkflowEngine(business_store=store).run(
        AgentRunRequest(
            input="我想给客户设计保险破冰开场",
            tenant_id="tenant_a",
            session_id="question_persist_session",
        )
    )

    assert response.insurance_kyc_status["information_status"] == "insufficient"
    assert len(store.kyc_questions) == 1
    question = store.kyc_questions[0]
    assert question.focus_key == response.active_intent["pending_focus"]
    assert question.round_no == response.insurance_kyc_status["kyc_question_round_count"] == 1
    path = [item["to_state"] for item in response.state_transitions]
    assert path.index("GENERATE_KYC_QUESTIONS") < path.index("PERSIST_MEMORY_SNAPSHOT")


def test_missing_business_memory_consent_uses_no_persistence_mode_instead_of_500() -> None:
    """未授权客户仍能获得本轮安全回答，但不会写业务事实、问题或输出。"""
    store = ConsentDeniedBusinessStore()

    response = WorkflowEngine(business_store=store).run(
        AgentRunRequest(
            input="帮我设计保险破冰开场",
            tenant_id="tenant_a",
            session_id="consent_missing_session",
        )
    )

    assert response.final_state == "FINAL"
    assert response.answer
    skipped = [event for event in response.trace_events if event.get("event") == "business_memory_skipped"]
    assert skipped[-1]["reason"] == "memory_processing_consent_missing_or_revoked"
    assert store.customer_facts == []
    assert store.kyc_questions == []
    assert store.generated_outputs == []
