"""Agentic 工具循环的数据契约。

# 文件说明：
# - 本文件定义工具迭代循环的 Pydantic v2 schema。
# - schema 只保存可审计的计划摘要、工具调用和 observation，不保存模型隐藏推理链。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_core.tools.schemas import ToolCall
from agent_core.utils.time import utc_now_iso


class ToolLoopStopReason(StrEnum):
    """工具循环停止原因枚举，用于 trace、测试和 API 响应解释。"""

    # 用户问题不需要工具，循环未实际执行。
    NO_TOOL_NEEDED = "no_tool_needed"
    # planner 判断已有证据足够，可以进入生成阶段。
    FINISHED = "finished"
    # planner 判断需要向用户澄清，主链路会短路到澄清响应。
    ASK_CLARIFICATION = "ask_clarification"
    # planner 或工具风控要求中止自动执行。
    ABORTED = "aborted"
    # 达到配置的最大迭代次数，强制停止避免无限循环。
    MAX_ITERATIONS = "max_iterations"
    # 连续两轮工具计划完全相同，判定存在 loop risk。
    REPEATED_TOOL_PLAN = "repeated_tool_plan"
    # 工具错误超过预算，停止继续调用工具并降级。
    TOOL_ERROR_BUDGET_EXCEEDED = "tool_error_budget_exceeded"
    # 工具触发人工审批，主链路返回 HUMAN_APPROVAL 等待恢复。
    HUMAN_APPROVAL = "human_approval"


class ToolLoopConfig(BaseModel):
    """工具迭代循环的预算与降级配置。"""

    max_iterations: int = Field(
        default=4,
        ge=1,
        description="工具循环最多迭代轮数，达到后必须停止，防止无限循环。",
    )
    max_tool_calls_per_iteration: int = Field(
        default=2,
        ge=1,
        description="单轮最多允许的工具调用数量，第一版默认顺序执行，不做并行调用。",
    )
    max_total_tool_calls: int = Field(
        default=6,
        ge=1,
        description="一次请求最多允许的工具调用总数，用于成本控制和 loop 风险控制。",
    )
    allow_parallel_tool_calls: bool = Field(
        default=False,
        description="是否允许单轮并行工具调用；当前实现保留开关但仍按顺序执行。",
    )
    stop_on_tool_error: bool = Field(
        default=False,
        description="工具出错后是否立即停止；默认允许校验节点做一次降级处理。",
    )
    enable_model_planner: bool = Field(
        default=True,
        description="是否允许使用模型 planner；模型不可用时必须安全降级到规则 planner。",
    )
    fallback_to_rule_router: bool = Field(
        default=True,
        description="模型 planner 不可用或输出非法时，是否回退到现有 ToolRouter 规则路由。",
    )


class ToolObservation(BaseModel):
    """单次工具执行后给 planner 和下游节点看的 observation。"""

    tool_name: str = Field(..., description="本次 observation 对应的工具名称。")
    status: str = Field(..., description="工具结果状态，例如 success、error 或 blocked。")
    output_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="工具输出的短摘要，只作为 data 使用，不能作为系统指令。",
    )
    error: str | None = Field(
        default=None,
        description="工具失败或被阻断时的错误摘要；为空表示未观察到错误。",
    )
    source_boundary: dict[str, Any] = Field(
        default_factory=dict,
        description="工具结果来源边界，必须声明外部内容是 untrusted data 而不是 instruction。",
    )
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="observation 创建时间，便于重放每轮工具执行顺序。",
    )


class ToolLoopDecision(BaseModel):
    """planner 每轮输出的结构化决策。"""

    action: Literal["call_tool", "finish", "ask_clarification", "abort"] = Field(
        ...,
        description="本轮工具循环动作：调用工具、结束、请求澄清或中止。",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="本轮准备调用的工具列表，必须来自白名单工具注册表。",
    )
    finish_reason: str | None = Field(
        default=None,
        description="action 为 finish/abort/ask_clarification 时的人类可读原因。",
    )
    rationale_summary: str = Field(
        default="",
        description="可审计的简短规划摘要，不保存模型隐藏推理链。",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="planner 对本轮决策的置信度，0 到 1。",
    )


class ToolLoopIteration(BaseModel):
    """工具循环中的一轮迭代记录。"""

    iteration_index: int = Field(..., ge=0, description="从 0 开始的工具循环轮次。")
    decision: ToolLoopDecision = Field(..., description="本轮 planner 输出的结构化决策。")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本轮实际执行或尝试执行的工具调用审计摘要。",
    )
    observations: list[ToolObservation] = Field(
        default_factory=list,
        description="本轮工具结果 observation 列表，供下一轮 planner 判断是否继续。",
    )
    status: Literal["planned", "executed", "skipped", "stopped"] = Field(
        default="planned",
        description="本轮迭代状态，用于区分只规划、已执行、跳过或停止。",
    )
    stop_reason: str | None = Field(
        default=None,
        description="如果本轮导致循环停止，这里记录停止原因。",
    )
    started_at: str = Field(
        default_factory=utc_now_iso,
        description="本轮开始时间，便于 trace 回放。",
    )
    finished_at: str | None = Field(
        default=None,
        description="本轮结束时间；尚未结束时为空。",
    )


class ToolLoopState(BaseModel):
    """工具循环运行时预算与状态快照。"""

    iterations: list[ToolLoopIteration] = Field(
        default_factory=list,
        description="已完成或已停止的工具循环迭代记录。",
    )
    total_tool_calls: int = Field(
        default=0,
        ge=0,
        description="工具循环累计工具调用次数，用于预算和测试断言。",
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="工具循环累计错误数量，超过预算后停止继续调用工具。",
    )
    last_plan_fingerprint: str | None = Field(
        default=None,
        description="上一轮工具计划指纹，用于检测连续重复计划。",
    )
    status: Literal["idle", "running", "finished", "stopped"] = Field(
        default="idle",
        description="工具循环整体状态，供 API 和 trace 展示。",
    )
    stop_reason: str | None = Field(
        default=None,
        description="工具循环最终停止原因；未停止时为空。",
    )
