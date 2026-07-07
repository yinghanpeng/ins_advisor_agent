"""Tool permission policy with scope and permission-level checks."""

# 文件说明：
# - 本文件属于工具系统，负责工具 schema、权限、注册或路由。
# - 工具调用必须有风险等级、权限 scope、超时、重试和错误结构。
from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.tools.schemas import ToolSpec


@dataclass
class ToolPermissionPolicy:
    # allowed_scopes 是当前自动执行允许的权限范围；不在集合里的工具必须被阻断或转人审。
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
    # allowed_levels 限制自动执行只允许 public/tenant；privileged/admin 需要更严格审批。
    allowed_levels: set[str] = field(default_factory=lambda: {"public", "tenant"})

    def can_call(self, spec: ToolSpec) -> bool:
        # 任何显式需要审批的工具都不能自动调用，必须进入 HUMAN_APPROVAL。
        if spec.requires_approval or spec.permission.requires_human_approval:
            return False
        # scope 必须命中允许集合；兼容检查 permission.scope 和旧字段 permission_scope。
        return spec.permission.scope in self.allowed_scopes or spec.permission_scope in self.allowed_scopes

    def explain(self, spec: ToolSpec) -> dict:
        """Return a structured decision for logs and tool guardrails."""
        # allowed 同时要求 can_call 通过，并且权限等级属于自动执行白名单。
        allowed = self.can_call(spec) and spec.permission.level in self.allowed_levels
        # 返回完整权限决策，让 ToolGuardrail 可以写入 guardrail_results 和 trace。
        return {
            "tool_name": spec.name,
            "allowed": allowed,
            "permission_level": spec.permission.level,
            "permission_scope": spec.permission.scope,
            "risk_level": spec.risk_level,
            "requires_approval": spec.requires_approval or spec.permission.requires_human_approval,
        }
