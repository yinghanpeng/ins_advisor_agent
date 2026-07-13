"""Fallback responses and recovery plan contracts."""

# 文件说明：
# - 本文件属于 Retry / Recovery 层，负责重试、降级、JSON repair 或恢复计划。
# - 失败时应清楚记录原因，不能无依据编造答案。
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def fallback_answer(reason: str) -> str:
    """生成保守降级回答，避免在缺少证据或工具失败时编造确定性结论。"""
    # 降级回答必须说明原因，并承诺只基于已有信息保守处理，避免伪装成工具/RAG 成功。
    return f"当前能力进入降级模式：{reason}。我会先基于已有信息给出保守建议。"


class RecoveryPlan(BaseModel):
    """Structured recovery decision used by workflow nodes and evals."""

    # error_type 用于把工具超时、JSON 解析失败、RAG 无结果等错误分类。
    error_type: str = Field(
        ...,
        description="错误类型，例如 tool_timeout、json_parse_failed、rag_no_result、high_risk_output。",
    )
    # action 决定 workflow 接下来是重试、降级还是失败终止。
    action: Literal["retry", "fallback", "fail"] = Field(
        ...,
        description="恢复动作：retry 重试，fallback 同步降级回答，fail 终止。",
    )
    # reason 是面向 trace 和用户解释的恢复原因。
    reason: str = Field(..., description="选择该恢复动作的原因，要求可写入 trace 并对用户解释。")
    # retryable 为 True 时才允许自动重试，高风险错误必须保持 False。
    retryable: bool = Field(
        default=False,
        description="当前错误是否允许自动重试。高风险输出和不可恢复错误应为 False。",
    )
    # max_attempts 限制自动重试次数，防止无限循环。
    max_attempts: int = Field(
        default=0,
        description="允许的最大尝试次数。retryable=True 时通常大于 0。",
    )
    # fallback_message 是降级时可直接返回给用户的保守说明。
    fallback_message: str = Field(
        default="",
        description="降级时返回给用户的保守说明。非 fallback 动作可以为空。",
    )
    # trace_fields 保存失败 step、工具名、重试次数等排障字段。
    trace_fields: dict = Field(
        default_factory=dict,
        description="恢复决策写入 trace 的扩展字段，例如 failed_step、tool_name、retry_count。",
    )


def plan_recovery(error_type: str, reason: str, retryable: bool = False) -> RecoveryPlan:
    """根据错误类型生成恢复计划，供 workflow engine 决定重试、降级或终止。"""
    # retryable=True 时优先生成 retry 计划，并附带最终失败后的 fallback_message。
    if retryable:
        # 返回最多两次尝试的计划；耗尽次数后使用同一原因生成保守降级说明。
        return RecoveryPlan(
            error_type=error_type,
            action="retry",
            reason=reason,
            retryable=True,
            max_attempts=2,
            fallback_message=fallback_answer(reason),
        )
    # 高风险输出或工具不执行原动作，同步返回安全降级说明。
    if error_type in {"high_risk_output", "high_risk_tool"}:
        # 高风险错误不可自动重试原动作，直接返回无副作用的信息型降级计划。
        return RecoveryPlan(
            error_type=error_type,
            action="fallback",
            reason=reason,
            fallback_message="该高风险操作或表达已被阻断，我可以提供无副作用的信息说明。",
        )
    # 其他不可重试错误默认降级回答，保证用户能得到清楚说明。
    return RecoveryPlan(
        error_type=error_type,
        action="fallback",
        reason=reason,
        fallback_message=fallback_answer(reason),
    )
