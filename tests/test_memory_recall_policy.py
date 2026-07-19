from agent_core.graph.nodes import load_business_memory, restore_memory
from agent_core.graph.state import AgentState
from agent_core.guardrails.metadata import TRUSTED_BUSINESS_IDENTITY_FLAG
from agent_core.memory.business_schemas import CustomerProfileFact, OpportunityCase
from agent_core.memory.business_store import InMemoryBusinessMemoryStore
from agent_core.memory.manager import MemoryLayer, MemoryManager
from agent_core.memory.recall import MemoryRecallItem, MemoryRecallResult


class _ProductionLikeMemoryBackend:
    """仅提供 Redis Session/Task 读取；若节点误读 Preference，测试立即暴露。"""

    def __init__(self) -> None:
        self.read_layers: list[MemoryLayer] = []

    def read(self, layer, tenant_id, subject_id):
        # 身份参数必须存在，但此测试只记录访问层级。
        assert tenant_id and subject_id
        self.read_layers.append(layer)
        if layer == MemoryLayer.PREFERENCE:
            raise AssertionError("生产召回应走 PostgreSQL Retriever，不能整包读取 Preference")
        return {"_version": 0}


class _RecallAuditRepository:
    """记录生产长期记忆决策与结果审计，不连接真实 PostgreSQL。"""

    def __init__(self) -> None:
        self.decisions: list[dict] = []
        self.results: list[dict] = []

    def insert_memory_recall_decision(self, **kwargs) -> None:
        self.decisions.append(kwargs)

    def insert_memory_recall_result(self, **kwargs) -> None:
        self.results.append(kwargs)


class _ProductionLikeRetriever:
    """模拟 PostgreSQL pgvector Retriever，并记录调用身份与决策。"""

    def __init__(self) -> None:
        self.repository = _RecallAuditRepository()
        self.calls: list[dict] = []

    def recall(self, *, decision, tenant_id, user_id):
        self.calls.append(
            {"decision": decision, "tenant_id": tenant_id, "user_id": user_id}
        )
        return MemoryRecallResult(
            decision=decision,
            items=[
                MemoryRecallItem(
                    layer="preference",
                    source_id="preference:user_a",
                    chunk_id="response_language",
                    content="response_language=中文",
                    rerank_score=0.95,
                )
            ],
            compact_summary={"preference": {"response_language": "中文"}},
        )


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


def test_production_restore_uses_postgresql_retriever_and_audits_decision() -> None:
    """生产长期偏好只在决策允许时走 Retriever，并把决策与命中摘要分别写审计表。"""
    backend = _ProductionLikeMemoryBackend()
    retriever = _ProductionLikeRetriever()
    state = AgentState(
        tenant_id="tenant_a",
        user_id="user_a",
        session_id="session_a",
        input_text="按我喜欢的风格回答",
    )

    restore_memory(
        state,
        backend,  # type: ignore[arg-type]
        memory_retriever=retriever,  # type: ignore[arg-type]
    )

    assert backend.read_layers == [MemoryLayer.SESSION, MemoryLayer.TASK]
    assert len(retriever.calls) == 1
    assert retriever.calls[0]["tenant_id"] == "tenant_a"
    assert retriever.calls[0]["user_id"] == "user_a"
    assert state.memory_context["preference"] == {"response_language": "中文"}
    assert state.memory_recall_decisions["preference"]["should_recall"] is True
    assert len(retriever.repository.decisions) == 1
    assert len(retriever.repository.results) == 1


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
