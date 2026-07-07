# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.guardrails.human_approval import (
    ApprovalDecision,
    ApprovalRequest,
    InMemoryApprovalStore,
)
from agent_core.memory.manager import MemoryLayer, MemoryManager
from agent_core.rag.retriever import HybridRetriever
from agent_core.rag.schemas import MetadataFilter, RetrievalQuery


def test_hybrid_retriever_uses_metadata_filter_and_scores():
    # 构造一个安全 chunk 和一个高风险未审批 chunk，验证 metadata filter 会拦截后者。
    retriever = HybridRetriever.from_dicts(
        [
            {
                "text": "企业主 破冰 资金分层 长期稳定",
                "metadata": {
                    "source_id": "s1",
                    "chunk_id": "c1",
                    "library": "sales_intelligence",
                    "tags": ["破冰"],
                    "risk_level": "low",
                    "approved_for_generation": True,
                },
            },
            {
                "text": "保证收益 高风险话术",
                "metadata": {
                    "source_id": "s2",
                    "chunk_id": "c2",
                    "library": "sales_intelligence",
                    "tags": ["高风险"],
                    "risk_level": "high",
                    "approved_for_generation": False,
                },
            },
        ]
    )
    # 检索时限定 sales_intelligence、最高 medium 风险、只允许 approved 内容。
    results = retriever.search(
        [RetrievalQuery(text="企业主破冰资金分层")],
        MetadataFilter(libraries=["sales_intelligence"], max_risk_level="medium", approved_only=True),
    )
    # 只应返回安全且已审批的 s1。
    assert len(results) == 1
    assert results[0].document.metadata.source_id == "s1"
    # score 大于 0 证明 hybrid scoring 生效。
    assert results[0].score > 0


def test_memory_manager_separates_layers_and_audits_access():
    # MemoryManager 应同时管理 session/task/preference 三层记忆。
    manager = MemoryManager()
    # session memory 保存当前会话任务阶段。
    manager.write(MemoryLayer.SESSION, "tenant_a", "session_1", {"stage": "collect_kyc"})
    # preference memory 保存跨会话偏好。
    manager.write(MemoryLayer.PREFERENCE, "tenant_a", "user_1", {"tone": "low_pressure"})
    # 读取 session 层时不应混到 preference 层。
    assert manager.read(MemoryLayer.SESSION, "tenant_a", "session_1")["stage"] == "collect_kyc"
    # 读取 preference 层时不应混到 session 层。
    assert manager.read(MemoryLayer.PREFERENCE, "tenant_a", "user_1")["tone"] == "low_pressure"
    # read/write 都应写入 audit_log，方便生产审计。
    assert len(manager.audit_log) >= 4


def test_human_approval_store_tracks_pending_and_decisions():
    # InMemoryApprovalStore 模拟高风险工具/输出的人审队列。
    store = InMemoryApprovalStore()
    # submit 后请求应进入 pending。
    request = store.submit(
        ApprovalRequest(trace_id="trace_1", reason="high risk output", payload_summary="blocked")
    )
    assert store.pending()[0].approval_id == request.approval_id
    # 写入审批结论后，该请求不应继续出现在 pending 中。
    store.decide(
        ApprovalDecision(
            approval_id=request.approval_id,
            decision="rejected",
            reviewer="compliance",
            comment="unsafe",
        )
    )
    assert store.pending() == []
