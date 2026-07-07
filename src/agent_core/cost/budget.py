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

    allowed: bool
    used_tokens: int
    requested_tokens: int
    budget: int
    action: Literal["allow", "reduce_context", "skip_optional_tool", "fallback"]


@dataclass
class CostBudget:
    """单次请求的 token 预算记录器，帮助节点决定继续、压缩或降级。"""

    request_token_budget: int = 12000
    used_tokens: int = 0

    def can_spend(self, tokens: int) -> bool:
        # 重点逻辑：只做预算判断，不修改 used_tokens，方便上游先试算。
        return self.used_tokens + tokens <= self.request_token_budget

    def decide(self, tokens: int) -> CostDecision:
        """把预算判断转换为结构化动作，供 workflow 节点执行降级策略。"""
        # 重点逻辑：把预算判断转成结构化 action，后续节点可以据此降级。
        allowed = self.can_spend(tokens)
        if allowed:
            action = "allow"
        elif tokens > self.request_token_budget:
            action = "reduce_context"
        else:
            action = "skip_optional_tool"
        return CostDecision(
            allowed=allowed,
            used_tokens=self.used_tokens,
            requested_tokens=tokens,
            budget=self.request_token_budget,
            action=action,
        )

    def spend(self, tokens: int) -> None:
        # 重点逻辑：真正扣减预算前再次检查，防止并发或上游误判导致超支。
        if not self.can_spend(tokens):
            raise ValueError("request token budget exceeded")
        self.used_tokens += tokens
