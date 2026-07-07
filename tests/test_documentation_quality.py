from pathlib import Path

import pytest

from agent_core.evals.feedback import HumanFeedback
from agent_core.graph.state import AgentNode, AgentState
from agent_core.guardrails.human_approval import ApprovalDecision, ApprovalRequest
from agent_core.rag.schemas import (
    DocumentMetadata,
    MetadataFilter,
    RetrievalDocument,
    RetrievalQuery,
    RetrievalResult,
)
from agent_core.recovery.fallback import RecoveryPlan
from agent_core.sales_intelligence.ingestion import RawInterview
from agent_core.sales_intelligence.schemas import CustomerKYC, SalesInsightCard, SalesInsightDigest
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


PYTHON_PATHS = [Path("main.py"), Path("evals/run_evals.py"), *Path("src").rglob("*.py"), *Path("tests").rglob("*.py")]

PYDANTIC_MODELS = [
    AgentState,
    StepRetryPolicy,
    WorkflowStepContract,
    WorkflowContract,
    AgentRunRequest,
    AgentRunResponse,
    EvalCase,
    ToolPermissionSpec,
    ToolSpec,
    ToolCall,
    ToolResult,
    RetrievalQuery,
    DocumentMetadata,
    RetrievalDocument,
    RetrievalResult,
    MetadataFilter,
    CustomerKYC,
    SalesInsightCard,
    SalesInsightDigest,
    RawInterview,
    InterviewSegment,
    ApprovalRequest,
    ApprovalDecision,
    HumanFeedback,
    RecoveryPlan,
]


def test_all_pydantic_fields_have_business_descriptions() -> None:
    """所有 Pydantic 字段都必须写业务含义，避免 API schema 变成只有字段名的空壳。"""
    missing: list[str] = []
    for model in PYDANTIC_MODELS:
        for field_name, field in model.model_fields.items():
            if not field.description or not field.description.strip():
                missing.append(f"{model.__name__}.{field_name}")
    assert not missing, "缺少 Field(description=...) 的字段：" + ", ".join(missing)


def test_no_template_style_generated_comments_remain() -> None:
    """禁止重新引入“方法说明/类说明”这类模板化注释。"""
    forbidden = ["方法" + "说明：", "类" + "说明：", "职责说明；" + "具体输入输出"]
    offenders: list[str] = []
    for path in PYTHON_PATHS:
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden):
            offenders.append(str(path))
    assert not offenders, "存在模板化注释：" + ", ".join(offenders)


def test_agent_state_move_to_uses_allowed_transitions_guard() -> None:
    """move_to 是状态切换入口，并已预留 allowed_transitions 合法跳转校验。"""
    state = AgentState(
        allowed_transitions={
            AgentNode.IDLE.value: [AgentNode.CLASSIFY_INTENT.value],
            AgentNode.CLASSIFY_INTENT.value: [AgentNode.ROUTE_CAPABILITY.value],
        }
    )
    state.move_to(AgentNode.CLASSIFY_INTENT, reason="allowed_by_contract")
    assert state.current_state == AgentNode.CLASSIFY_INTENT
    assert state.state_transitions[-1]["to_state"] == AgentNode.CLASSIFY_INTENT.value

    with pytest.raises(ValueError):
        state.move_to(AgentNode.FINAL, reason="not_allowed_by_contract")
