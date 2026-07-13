from agent_core.graph.nodes import load_business_memory, restore_memory
from agent_core.graph.state import AgentState
from agent_core.guardrails.metadata import TRUSTED_BUSINESS_IDENTITY_FLAG
from agent_core.memory.business_schemas import CustomerProfileFact, OpportunityCase
from agent_core.memory.business_store import InMemoryBusinessMemoryStore
from agent_core.memory.manager import MemoryLayer, MemoryManager


def test_restore_memory_skips_long_term_preference_for_calculator_request() -> None:
    """计算类请求不应召回长期偏好，避免无关记忆污染工具结果。"""
    manager = MemoryManager()
    manager.write(
        MemoryLayer.PREFERENCE,
        "tenant_a",
        "user_a",
        {"preferred_style": "喜欢非常长的销售话术"},
    )
    manager.audit_log.clear()
    state = AgentState(tenant_id="tenant_a", user_id="user_a", session_id="session_a", input_text="计算 12*8+3")

    restore_memory(state, manager)

    assert state.memory_recall_decision["should_recall"] is False
    assert state.memory_context["preference"] == {}
    assert not [entry for entry in manager.audit_log if entry.get("layer") == "preference" and entry.get("action") == "read"]


def test_restore_memory_recalls_preference_with_hybrid_search_when_needed() -> None:
    """命中偏好信号时，长期偏好才通过 hybrid search + rerank 进入上下文。"""
    manager = MemoryManager()
    manager.write(
        MemoryLayer.PREFERENCE,
        "tenant_a",
        "user_a",
        {
            "memory_candidates": [
                {"type": "preferred_style", "value": "喜欢结构化中文，先结论后步骤"},
                {"type": "irrelevant", "value": "临时问过天气"},
            ]
        },
    )
    manager.audit_log.clear()
    state = AgentState(
        tenant_id="tenant_a",
        user_id="user_a",
        session_id="session_a",
        input_text="按我喜欢的风格写客户沟通策略",
    )

    restore_memory(state, manager)

    assert state.memory_recall_decision["should_recall"] is True
    assert state.memory_recall_results
    assert "结构化中文" in str(state.memory_context["preference"])
    assert any(entry.get("layer") == "preference" and entry.get("action") == "read" for entry in manager.audit_log)


def test_business_memory_recall_uses_hybrid_rerank_for_relevant_customer_fact() -> None:
    """业务长期事实按需召回，并通过 rerank 优先返回与当前问题相关的事实。"""
    store = InMemoryBusinessMemoryStore()
    case = OpportunityCase(tenant_id="tenant_a", advisor_id="advisor_a", customer_id="customer_a")
    store.upsert_opportunity_case(case)
    store.upsert_customer_fact(
        CustomerProfileFact(
            tenant_id="tenant_a",
            customer_id="customer_a",
            fact_key="financial_preference",
            fact_value="偏好银行理财和稳健现金流",
            source_type="user_message",
            evidence_text="用户明确说客户喜欢银行理财",
        )
    )
    store.upsert_customer_fact(
        CustomerProfileFact(
            tenant_id="tenant_a",
            customer_id="customer_a",
            fact_key="overseas_experience",
            fact_value="曾经去过欧洲旅行",
            source_type="user_message",
            evidence_text="用户提到客户去过欧洲旅行",
        )
    )
    state = AgentState(
        tenant_id="tenant_a",
        session_id="session_a",
        input_text="这个客户喜欢银行理财，我怎么低压切入",
        intent="insurance_break_ice",
        domain_skill="insurance_advisor",
        metadata={
            TRUSTED_BUSINESS_IDENTITY_FLAG: True,
            "advisor_id": "advisor_a",
            "customer_id": "customer_a",
            "conversation_id": "session_a",
        },
    )

    load_business_memory(state, store)

    assert state.memory_recall_decision["should_recall"] is True
    assert state.memory_recall_results[0]["metadata"]["fact_key"] == "financial_preference"
    assert state.profile_state["financial_preference"] == "偏好银行理财和稳健现金流"
