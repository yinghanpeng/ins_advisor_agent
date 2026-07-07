"""Human-in-the-loop approval contracts and in-memory queue."""

# 文件说明：
# - 本文件属于 Guardrails 层，负责输入安全、工具权限、输出合规或人工审批。
# - 保险金融场景必须拦截收益承诺、避税避债、恐吓营销和编造案例。
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_core.utils.ids import new_id
from agent_core.utils.time import utc_now_iso


class ApprovalRequest(BaseModel):
    """人工审批请求。高风险工具调用或高风险输出应先进入该队列。"""

    # approval_id 是审批单唯一 ID，用于把请求和后续 decision 关联起来。
    approval_id: str = Field(
        default_factory=lambda: new_id("approval"),
        description="审批请求 ID。用于把待审批事项和最终审批决定关联起来。",
    )
    # trace_id 指向触发审批的 Agent 运行，审批人可以回放完整状态链路。
    trace_id: str = Field(..., description="触发审批的 Agent 运行 trace_id，便于回看完整链路。")
    # reason 说明为什么需要人工审批，例如工具有副作用或输出命中高风险表达。
    reason: str = Field(..., description="进入人工审批的原因，例如 high_risk_tool、收益承诺、外部写操作。")
    # risk_level 限定为 medium/high，低风险事项不应进入审批队列。
    risk_level: Literal["medium", "high"] = Field(
        default="high",
        description="审批事项风险等级。high 通常表示必须人工确认后才能继续。",
    )
    # payload_summary 只保存摘要，避免审批界面泄露完整敏感 payload。
    payload_summary: str = Field(
        ...,
        description="待审批内容摘要。只放必要信息，避免把完整敏感 payload 直接暴露给审批界面。",
    )
    # created_at 用于审计和计算待审批时长。
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="审批请求创建时间，ISO 字符串，用于超时提醒和审计。",
    )


class ApprovalDecision(BaseModel):
    """人工审批结果。workflow engine 根据该对象决定继续、拒绝或要求修改。"""

    # approval_id 必须对应已有审批请求，避免写入孤立审批结果。
    approval_id: str = Field(..., description="被处理的审批请求 ID，必须已经存在于审批队列中。")
    # decision 是审批结论，workflow 可以据此放行、拒绝或要求修改。
    decision: Literal["approved", "rejected", "needs_edit"] = Field(
        ...,
        description="审批结论：approved 放行，rejected 拒绝，needs_edit 表示需要修改后重提。",
    )
    # reviewer 记录审批人身份，满足审计追责要求。
    reviewer: str = Field(..., description="审批人标识，例如用户名、员工号或本地调试 reviewer。")
    # comment 保存审批备注，特别是 rejected/needs_edit 时的修改建议。
    comment: str = Field(
        default="",
        description="审批备注，说明拒绝原因、修改建议或放行条件。",
    )
    # decided_at 记录审批完成时间，用于 SLA 和审计。
    decided_at: str = Field(
        default_factory=utc_now_iso,
        description="审批完成时间，ISO 字符串，用于审计和 SLA 统计。",
    )


class InMemoryApprovalStore:
    """高风险工具调用和输出内容的本地人工审批队列。"""

    def __init__(self) -> None:
        """初始化内存审批队列；生产环境可替换为数据库或工单系统。"""
        # requests 保存所有待审批请求，key 是 approval_id。
        self.requests: dict[str, ApprovalRequest] = {}
        # decisions 保存已经处理过的审批结果，key 也是 approval_id。
        self.decisions: dict[str, ApprovalDecision] = {}

    def submit(self, request: ApprovalRequest) -> ApprovalRequest:
        """提交一个待审批请求，并按 approval_id 暂存。"""
        # 将审批请求写入队列；同 ID 写入会覆盖旧请求，生产可增加幂等校验。
        self.requests[request.approval_id] = request
        # 返回 request 便于调用方拿到 approval_id。
        return request

    def decide(self, decision: ApprovalDecision) -> ApprovalDecision:
        """写入审批结论；如果请求不存在则抛错避免误审批。"""
        # 审批结果必须对应已有请求，否则说明审批链路缺少上下文。
        if decision.approval_id not in self.requests:
            raise KeyError(f"approval request not found: {decision.approval_id}")
        # 写入审批结论；workflow 后续可根据 decision 决定继续或终止。
        self.decisions[decision.approval_id] = decision
        # 返回 decision 方便 API 层直接响应。
        return decision

    def pending(self) -> list[ApprovalRequest]:
        """返回尚未被审批人处理的请求列表。"""
        # pending 只返回没有对应 decision 的 request。
        return [
            request
            # 遍历所有审批请求并检查它是否还没有被处理。
            for approval_id, request in self.requests.items()
            # decisions 中不存在该 approval_id，说明仍待审批。
            if approval_id not in self.decisions
        ]
