"""Tool guardrails."""

# 文件说明：
# - 本文件属于 Guardrails 层，负责工具权限和副作用的同步准入判断。
# - 保险金融场景必须拦截收益承诺、避税避债、恐吓营销和编造案例。
from __future__ import annotations

from agent_core.tools.permissions import ToolPermissionPolicy
from agent_core.tools.schemas import ToolSpec


class ToolGuardrail:
    """工具调用前的权限和副作用检查器。"""

    def __init__(self, policy: ToolPermissionPolicy | None = None) -> None:
        """初始化工具权限策略；未传入时使用本地默认策略。"""
        # 权限策略由应用注入；缺省实例提供一致的本地最小权限规则。
        self.policy = policy or ToolPermissionPolicy()

    def review(self, spec: ToolSpec) -> dict:
        """根据 ToolSpec 判断工具是否允许自动调用。"""
        # explain 同时返回布尔结果和拒绝原因，便于 Trace 解释权限决策。
        decision = self.policy.explain(spec)
        # 显式转为 bool，避免后续把非布尔真值直接暴露为契约字段。
        allowed = bool(decision["allowed"])
        # 返回稳定 Guardrail 字典；拒绝工具时不创建任何人工等待状态。
        return {
            "guardrail_name": "tool_permission",
            "triggered": not allowed,
            "reason": "" if allowed else str(decision["denial_reason"]),
            "action": "pass" if allowed else "deny",
            "decision": decision,
        }
