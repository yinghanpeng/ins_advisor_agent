"""Tool guardrails."""

# 文件说明：
# - 本文件属于 Guardrails 层，负责输入安全、工具权限、输出合规或人工审批。
# - 保险金融场景必须拦截收益承诺、避税避债、恐吓营销和编造案例。
from __future__ import annotations

from agent_core.tools.permissions import ToolPermissionPolicy
from agent_core.tools.schemas import ToolSpec


class ToolGuardrail:
    """工具调用前的权限和审批检查器。"""

    def __init__(self, policy: ToolPermissionPolicy | None = None) -> None:
        """初始化工具权限策略；未传入时使用本地默认策略。"""
        self.policy = policy or ToolPermissionPolicy()

    def review(self, spec: ToolSpec) -> dict:
        """根据 ToolSpec 判断工具是否允许自动调用。"""
        allowed = self.policy.can_call(spec)
        return {
            "guardrail_name": "tool_permission",
            "triggered": not allowed,
            "reason": "" if allowed else f"scope not allowed: {spec.permission_scope}",
            "action": "pass" if allowed else "human_approval",
        }
