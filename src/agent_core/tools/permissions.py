"""Tool permission policy with scope and permission-level checks."""

# 文件说明：
# - 本文件属于工具系统，负责工具 schema、权限、注册或路由。
# - 工具调用必须有风险等级、权限 scope、超时、重试和错误结构。
from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.tools.schemas import ToolSpec


@dataclass
class ToolPermissionPolicy:
    """定义客户渠道允许的只读工具作用域与权限等级白名单。"""

    # allowed_scopes 是当前客户渠道允许的权限范围；不在集合里的工具直接阻断。
    allowed_scopes: set[str] = field(
        default_factory=lambda: {
            # local 表示纯本地能力。
            "local",
            # local.time 只读取本地时间。
            "local.time",
            # local.compute 只做本地计算或换算。
            "local.compute",
            # weather.read 表示天气读取能力。
            "weather.read",
            # internet.read 表示只读互联网能力，例如搜索或新闻。
            "internet.read",
            # knowledge.read 表示内部知识库只读检索。
            "knowledge.read",
            # llm.transform 表示文本变换类能力，例如摘要和翻译。
            "llm.transform",
            # files.read 表示读取已授权上传文件。
            "files.read",
        }
    )
    # allowed_levels 限制客户渠道只允许 public/tenant；privileged/admin 直接阻断。
    allowed_levels: set[str] = field(default_factory=lambda: {"public", "tenant"})

    def can_call(self, spec: ToolSpec) -> bool:
        """判断客户渠道是否同时满足副作用、scope 与权限级别约束。"""
        # 客户系统不允许任何写入、对外动作或金融副作用工具。
        if spec.side_effect or spec.side_effect_level not in {"none", "read_only"}:
            # 副作用工具同步拒绝，不创建人工审批或异步等待状态。
            return False
        # scope 和权限级别必须同时命中客户渠道白名单。
        scope_allowed = spec.permission.scope in self.allowed_scopes or spec.permission_scope in self.allowed_scopes
        # 同时满足 scope 与 public/tenant 等级才允许执行，任一失败都返回 False。
        return scope_allowed and spec.permission.level in self.allowed_levels

    def explain(self, spec: ToolSpec) -> dict:
        """Return a structured decision for logs and tool guardrails."""
        # 先计算最终布尔值，下面再给出稳定的首个拒绝原因。
        allowed = self.can_call(spec)
        # 副作用拒绝优先级最高，避免把金融动作误报为普通 scope 问题。
        if spec.side_effect or spec.side_effect_level not in {"none", "read_only"}:
            # 记录稳定机器码，日志和前端可据此展示“客户渠道不支持该动作”。
            denial_reason = "side_effect_not_allowed"
        # 权限级别不在 public/tenant 时返回 level 拒绝。
        elif spec.permission.level not in self.allowed_levels:
            # privileged/admin 统一归类为权限等级不允许。
            denial_reason = "permission_level_not_allowed"
        # level 允许但新旧任一 scope 都未命中时返回 scope 拒绝。
        elif spec.permission.scope not in self.allowed_scopes and spec.permission_scope not in self.allowed_scopes:
            # 新旧 scope 字段都未命中白名单时记录作用域拒绝。
            denial_reason = "permission_scope_not_allowed"
        # 所有约束通过时使用空拒绝原因。
        else:
            # 空字符串表示无拒绝原因，与 allowed=True 保持一致。
            denial_reason = ""
        # 返回完整权限决策，让 ToolGuardrail 可以写入 guardrail_results 和 trace。
        return {
            "tool_name": spec.name,
            "allowed": allowed,
            "permission_level": spec.permission.level,
            "permission_scope": spec.permission.scope,
            "risk_level": spec.risk_level,
            "side_effect_level": spec.side_effect_level,
            "denial_reason": denial_reason,
        }
