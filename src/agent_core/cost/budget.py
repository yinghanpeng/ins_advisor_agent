"""Cost budget tracking with structured decisions."""

# 文件说明：
# - 本文件属于成本控制层，负责 token budget、预算决策或模型路由。
# - 预算压力下应压缩上下文、减少工具调用或降级输出。
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CostDecision:
    """Structured budget decision for logs and traces."""

    # allowed 表示本次 token 申请是否可以按原计划执行。
    allowed: bool
    # used_tokens 保存决策前已经累计使用的 token 数。
    used_tokens: int
    # requested_tokens 是当前节点计划新增消耗的 token 数。
    requested_tokens: int
    # budget 是单次请求允许使用的 token 总上限。
    budget: int
    # action 给工作流一个可直接执行的预算动作，而不是仅返回布尔值。
    action: Literal["allow", "reduce_context", "skip_optional_tool", "fallback"]


@dataclass
class CostBudget:
    """单次请求的 token 预算记录器，帮助节点决定继续、压缩或降级。"""

    # request_token_budget 控制单次 Agent 请求的 token 消耗上限。
    request_token_budget: int = 12000
    # used_tokens 累计当前请求中已经确认扣除的 token 数。
    used_tokens: int = 0

    def can_spend(self, tokens: int) -> bool:
        """试算新增 token 是否仍处于请求预算内，但不实际扣减额度。"""

        # 重点逻辑：只做预算判断，不修改 used_tokens，方便上游先试算。
        return self.used_tokens + tokens <= self.request_token_budget

    def decide(self, tokens: int) -> CostDecision:
        """把预算判断转换为结构化动作，供 workflow 节点执行降级策略。"""
        # 重点逻辑：把预算判断转成结构化 action，后续节点可以据此降级。
        allowed = self.can_spend(tokens)
        # 预算足够时允许节点按原计划执行。
        if allowed:
            # allow 表示无需压缩上下文或跳过工具。
            action = "allow"
        # 单个申请本身就超过总预算时，优先要求压缩输入上下文。
        elif tokens > self.request_token_budget:
            # reduce_context 提示节点缩短 prompt 后重新试算。
            action = "reduce_context"
        # 累计消耗导致剩余额度不足时，跳过非必需工具以保护主回答。
        else:
            # skip_optional_tool 保留核心生成额度，牺牲可选增强能力。
            action = "skip_optional_tool"
        # 把判断时的预算快照完整返回，便于日志、trace 和节点执行一致动作。
        return CostDecision(
            allowed=allowed,
            used_tokens=self.used_tokens,
            requested_tokens=tokens,
            budget=self.request_token_budget,
            action=action,
        )

    def spend(self, tokens: int) -> None:
        """在预算允许时累计实际 token 花费，超额申请则拒绝。"""

        # 重点逻辑：真正扣减预算前再次检查，防止并发或上游误判导致超支。
        if not self.can_spend(tokens):
            # 拒绝超预算扣减，避免记录器进入无法恢复的非法状态。
            raise ValueError("request token budget exceeded")
        # 仅在检查通过后累计实际花费，保持 used_tokens 不超过预算上限。
        self.used_tokens += tokens
