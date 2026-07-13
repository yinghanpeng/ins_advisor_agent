"""Evaluation helpers."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations

from agent_core.models.client import OpenAICompatibleChatClient
from agent_core.workflow.contracts import AgentRunResponse, EvalCase
from pydantic import BaseModel, Field


def rule_based_evaluate(case: EvalCase, response: AgentRunResponse) -> dict:
    """按必含词、禁含词和 trace 完整性执行确定性回归评测。"""

    # 收集所有未命中的必含词，而不是遇到首个失败就停止，便于一次定位完整差异。
    missing = [term for term in case.must_include if term not in response.answer]
    # 收集回答中实际出现的禁含词，供失败报告直接展示违规项。
    forbidden = [term for term in case.must_not_include if term in response.answer]
    # 三项条件全部满足才通过：无遗漏、无禁词且运行返回了可追踪的 trace_id。
    passed = not missing and not forbidden and bool(response.trace_id)
    # 返回结构化诊断信息，使 CI 能展示具体失败原因而不只看到布尔值。
    return {"passed": passed, "missing": missing, "forbidden": forbidden}


def schema_evaluate(response: AgentRunResponse) -> dict:
    """通过序列化后重新校验，确认 AgentRunResponse 满足公开响应契约。"""

    # 先导出再用同一模型校验，覆盖嵌套字段序列化后的真实 API 形态。
    response.model_validate(response.model_dump())
    # 校验调用未抛异常即表示响应 schema 合法。
    return {"passed": True}


def llm_as_judge_evaluate(
    case: EvalCase,
    response: AgentRunResponse,
    judge_client: OpenAICompatibleChatClient | None = None,
    ) -> dict:
    """使用配置化 judge 模型评估输出质量。"""
    # LLM 评测必须使用显式配置的真实 judge，禁止用规则结果冒充模型裁判。
    if judge_client is None:
        # 缺少客户端时明确失败，避免 CI 将未执行评测误判为通过。
        raise RuntimeError("LLM judge client 未配置，不能执行模型评测")
    # 把评测样本和实际响应交给 judge，并要求输出 JudgeResult 结构化 JSON。
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
    # 将通过 schema 校验的裁判结果转换为可序列化字典。
    result = parsed.model_dump()
    # 追加实际 judge 模型名，便于比较模型版本变更导致的评分漂移。
    result["model_name"] = model_result.model_name
    # 返回包含裁判结论、理由、分数与模型标识的完整结果。
    return result


class JudgeResult(BaseModel):
    """LLM-as-judge 结构化输出。"""

    # passed 是最终通过结论，供 CI 门禁直接消费。
    passed: bool = Field(..., description="该样本是否通过评测。")
    # reason 保存裁判的简要依据，便于人工复核失败样本。
    reason: str = Field(default="", description="裁判模型给出的通过或失败原因。")
    # score 用零到一的连续值表达质量，方便版本间趋势比较。
    score: float = Field(default=0.0, ge=0, le=1, description="0 到 1 的质量分。")
