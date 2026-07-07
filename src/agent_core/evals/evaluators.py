"""Evaluation helpers."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations

from agent_core.workflow.contracts import AgentRunResponse, EvalCase


def rule_based_evaluate(case: EvalCase, response: AgentRunResponse) -> dict:
    missing = [term for term in case.must_include if term not in response.answer]
    forbidden = [term for term in case.must_not_include if term in response.answer]
    passed = not missing and not forbidden and bool(response.trace_id)
    return {"passed": passed, "missing": missing, "forbidden": forbidden}


def schema_evaluate(response: AgentRunResponse) -> dict:
    response.model_validate(response.model_dump())
    return {"passed": True}


def llm_as_judge_placeholder(case: EvalCase, response: AgentRunResponse) -> dict:
    return {
        "passed": bool(response.answer),
        "mode": "mock",
        "note": "Replace with configured LLM judge for production experiments.",
    }

