# 这个测试文件专门约束“文档和注释质量”，防止项目再次退化成无描述字段或模板化注释。
from pathlib import Path

import pytest

from agent_core.evals.feedback import HumanFeedback
from agent_core.graph.state import AgentNode, AgentState
from agent_core.guardrails.human_approval import ApprovalDecision, ApprovalRequest
from agent_core.memory.business_schemas import (
    Advisor,
    AdvisorProfileFact,
    AgentSessionState,
    AnalysisRun,
    CaseOutcome,
    Conversation,
    ConversationMessage,
    Customer,
    CustomerProfileFact,
    DifyKYCAnalysisOutput,
    GeneratedOutput,
    KYCQuestion,
    MemoryEvent,
    OpportunityCase,
    Tenant,
)
from agent_core.memory.write_policy import MemoryWriteProposal, MemoryWriteValidationResult
from agent_core.memory.recall import MemoryRecallDecision, MemoryRecallItem, MemoryRecallResult
from agent_core.rag.schemas import (
    DocumentMetadata,
    MetadataFilter,
    RetrievalDocument,
    RetrievalQuery,
    RetrievalResult,
)
from agent_core.recovery.fallback import RecoveryPlan
from agent_core.sales_intelligence.ingestion import RawInterview
from agent_core.sales_intelligence.schemas import (
    CorpusBatch,
    CorpusCase,
    CorpusMessage,
    CustomerKYC,
    DialoguePattern,
    SalesInsightCard,
    SalesInsightDigest,
)
from agent_core.sales_intelligence.segmenter import InterviewSegment
from agent_core.tools.schemas import ToolCall, ToolPermissionSpec, ToolResult, ToolSpec
from agent_core.workflow.contracts import (
    AgentRunRequest,
    AgentRunResponse,
    EvalCase,
    StepRetryPolicy,
    WorkflowContract,
    WorkflowStepContract,
)


# PYTHON_PATHS 覆盖 main、evals、src、tests，确保注释质量约束扫描整个项目 Python 文件。
PYTHON_PATHS = [Path("main.py"), Path("evals/run_evals.py"), *Path("src").rglob("*.py"), *Path("tests").rglob("*.py")]

# PYDANTIC_MODELS 是所有必须具备 Field(description=...) 的结构化契约模型清单。
PYDANTIC_MODELS = [
    # AgentState 是主链路状态黑匣子，字段描述必须完整。
    AgentState,
    # Workflow contract 模型约束 step 输入、输出、风控、重试和 trace。
    StepRetryPolicy,
    WorkflowStepContract,
    WorkflowContract,
    AgentRunRequest,
    AgentRunResponse,
    EvalCase,
    # Tool schema 模型约束工具权限、风险、调用和结果。
    ToolPermissionSpec,
    ToolSpec,
    ToolCall,
    ToolResult,
    # RAG schema 模型约束 query、metadata、document、result 和 filters。
    RetrievalQuery,
    DocumentMetadata,
    RetrievalDocument,
    RetrievalResult,
    MetadataFilter,
    # Sales Intelligence schema 模型约束 KYC、洞察卡片和摘要。
    CustomerKYC,
    SalesInsightCard,
    SalesInsightDigest,
    CorpusBatch,
    CorpusCase,
    CorpusMessage,
    DialoguePattern,
    # 业务记忆 schema 模型约束租户、顾问、客户、事实、case、会话、分析、输出和结果闭环。
    Tenant,
    Advisor,
    Customer,
    AdvisorProfileFact,
    CustomerProfileFact,
    OpportunityCase,
    Conversation,
    ConversationMessage,
    AgentSessionState,
    KYCQuestion,
    DifyKYCAnalysisOutput,
    AnalysisRun,
    GeneratedOutput,
    MemoryEvent,
    CaseOutcome,
    MemoryWriteProposal,
    MemoryWriteValidationResult,
    MemoryRecallDecision,
    MemoryRecallItem,
    MemoryRecallResult,
    # 访谈导入和分段模型约束销售语料资产化入口。
    RawInterview,
    InterviewSegment,
    # 人工审批模型约束 human-in-the-loop 请求与决策。
    ApprovalRequest,
    ApprovalDecision,
    # 反馈和恢复模型约束评估与 recovery 策略。
    HumanFeedback,
    RecoveryPlan,
]


def test_all_pydantic_fields_have_business_descriptions() -> None:
    """所有 Pydantic 字段都必须写业务含义，避免 API schema 变成只有字段名的空壳。"""
    # missing 收集缺少 description 的字段名，最后统一报错方便一次性修复。
    missing: list[str] = []
    # 遍历所有核心 Pydantic 模型。
    for model in PYDANTIC_MODELS:
        # model_fields 是 Pydantic v2 暴露的字段定义集合。
        for field_name, field in model.model_fields.items():
            # description 为空说明该字段没有业务解释。
            if not field.description or not field.description.strip():
                missing.append(f"{model.__name__}.{field_name}")
    # 任何字段缺描述都直接失败，防止契约文档空心化。
    assert not missing, "缺少 Field(description=...) 的字段：" + ", ".join(missing)


def test_no_template_style_generated_comments_remain() -> None:
    """禁止重新引入模板化坏注释。"""
    # forbidden 故意拆字符串，避免测试文件自身被搜索命中。
    forbidden = ["方法" + "说明：", "类" + "说明：", "职责说明；" + "具体输入输出"]
    # offenders 收集包含模板化注释的文件路径。
    offenders: list[str] = []
    # 扫描项目 Python 文件，防止机械生成注释重新进入代码库。
    for path in PYTHON_PATHS:
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden):
            offenders.append(str(path))
    # 发现模板化注释时直接失败，并列出文件。
    assert not offenders, "存在模板化注释：" + ", ".join(offenders)


def test_agent_state_move_to_uses_allowed_transitions_guard() -> None:
    """move_to 是状态切换入口，并已预留 allowed_transitions 合法跳转校验。"""
    # 构造一个只允许 IDLE -> CLASSIFY_INTENT -> ROUTE_CAPABILITY 的状态白名单。
    state = AgentState(
        allowed_transitions={
            AgentNode.IDLE.value: [AgentNode.CLASSIFY_INTENT.value],
            AgentNode.CLASSIFY_INTENT.value: [AgentNode.ROUTE_CAPABILITY.value],
        }
    )
    # 合法跳转应成功，并写入 state_transitions。
    state.move_to(AgentNode.CLASSIFY_INTENT, reason="allowed_by_contract")
    assert state.current_state == AgentNode.CLASSIFY_INTENT
    assert state.state_transitions[-1]["to_state"] == AgentNode.CLASSIFY_INTENT.value

    # 非法跳转应被 move_to 阻断，证明状态切换唯一入口具备校验预留。
    with pytest.raises(ValueError):
        state.move_to(AgentNode.FINAL, reason="not_allowed_by_contract")
