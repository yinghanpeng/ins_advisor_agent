"""Evaluation helpers."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations

from agent_core.models.client import OpenAICompatibleChatClient
from agent_core.workflow.contracts import AgentRunResponse, EvalCase
from pydantic import BaseModel, Field


def rule_based_evaluate(case: EvalCase, response: AgentRunResponse) -> dict:
    missing = [term for term in case.must_include if term not in response.answer]
    forbidden = [term for term in case.must_not_include if term in response.answer]
    passed = not missing and not forbidden and bool(response.trace_id)
    return {"passed": passed, "missing": missing, "forbidden": forbidden}


def schema_evaluate(response: AgentRunResponse) -> dict:
    response.model_validate(response.model_dump())
    return {"passed": True}


def llm_as_judge_evaluate(
    case: EvalCase,
    response: AgentRunResponse,
    judge_client: OpenAICompatibleChatClient | None = None,
) -> dict:
    """使用配置化 judge 模型评估输出质量。"""
    if judge_client is None:
        raise RuntimeError("LLM judge client 未配置，不能执行模型评测")
    parsed, model_result = judge_client.complete_json(
        messages=[
            {"role": "system", "content": "你是 Agent 回归评测裁判，只输出 JSON。"},
            {
                "role": "user",
                "content": f"Eval case: {case.model_dump()}\nResponse: {response.model_dump()}",
            },
        ],
        schema_model=JudgeResult,
    )
    result = parsed.model_dump()
    result["model_name"] = model_result.model_name
    return result
class JudgeResult(BaseModel):
    """LLM-as-judge 结构化输出。"""

    passed: bool = Field(..., description="该样本是否通过评测。")
    reason: str = Field(default="", description="裁判模型给出的通过或失败原因。")
    score: float = Field(default=0.0, ge=0, le=1, description="0 到 1 的质量分。")
