"""Fallback responses and recovery plan contracts."""

# 文件说明：
# - 本文件属于 Retry / Recovery 层，负责重试、降级、JSON repair 或恢复计划。
# - 失败时应清楚记录原因，不能无依据编造答案。
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def fallback_answer(reason: str) -> str:
    """生成保守降级回答，避免在缺少证据或工具失败时编造确定性结论。"""
    return f"当前能力进入降级模式：{reason}。我会先基于已有信息给出保守建议。"


class RecoveryPlan(BaseModel):
    """Structured recovery decision used by workflow nodes and evals."""

    error_type: str = Field(
        ...,
        description="错误类型，例如 tool_timeout、json_parse_failed、rag_no_result、high_risk_output。",
    )
    action: Literal["retry", "fallback", "human_approval", "fail"] = Field(
        ...,
        description="恢复动作：retry 重试，fallback 降级回答，human_approval 人工审批，fail 终止。",
    )
    reason: str = Field(..., description="选择该恢复动作的原因，要求可写入 trace 并对用户解释。")
    retryable: bool = Field(
        default=False,
        description="当前错误是否允许自动重试。高风险输出和不可恢复错误应为 False。",
    )
    max_attempts: int = Field(
        default=0,
        description="允许的最大尝试次数。retryable=True 时通常大于 0。",
    )
    fallback_message: str = Field(
        default="",
        description="降级时返回给用户的保守说明。非 fallback 动作可以为空。",
    )
    trace_fields: dict = Field(
        default_factory=dict,
        description="恢复决策写入 trace 的扩展字段，例如 failed_step、tool_name、retry_count。",
    )


def plan_recovery(error_type: str, reason: str, retryable: bool = False) -> RecoveryPlan:
    """根据错误类型生成恢复计划，供 workflow engine 决定重试、降级或人工审批。"""
    if retryable:
        return RecoveryPlan(
            error_type=error_type,
            action="retry",
            reason=reason,
            retryable=True,
            max_attempts=2,
            fallback_message=fallback_answer(reason),
        )
    if error_type in {"high_risk_output", "high_risk_tool"}:
        return RecoveryPlan(error_type=error_type, action="human_approval", reason=reason)
    return RecoveryPlan(
        error_type=error_type,
        action="fallback",
        reason=reason,
        fallback_message=fallback_answer(reason),
    )
