"""Tool contracts with permission and execution metadata."""

# 文件说明：
# - 本文件属于工具系统，负责工具 schema、权限、注册或路由。
# - 工具调用必须有风险等级、权限 scope、超时、重试和错误结构。
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# 风险等级限定为三档，供工具 Guardrail 进行稳定枚举判断。
RiskLevel = Literal["low", "medium", "high"]
# 权限等级区分公开、租户、特权和管理员，客户渠道只开放前两档。
PermissionLevel = Literal["public", "tenant", "privileged", "admin"]
# 副作用等级明确区分只读、写入、外部动作和金融动作，后三类会被同步阻断。
SideEffectLevel = Literal["none", "read_only", "write", "external_action", "financial"]
# 审计等级控制是否及以何种粒度记录工具调用元数据。
AuditLevel = Literal["none", "basic", "full"]


class ToolPermissionSpec(BaseModel):
    """Permission metadata used by tool guardrails and audit logs."""

    # level 表示工具权限等级；ToolPermissionPolicy 会用它判断是否允许客户渠道执行。
    level: PermissionLevel = Field(
        default="public",
        description="工具权限等级。public 可本地调用，tenant 需要租户隔离，privileged/admin 在客户渠道直接禁止。",
    )
    # scope 表示具体权限作用域，例如 internet.read 或 local.compute，用于白名单校验。
    scope: str = Field(
        default="local",
        description="权限作用域，例如 local、tenant:abc、crm:write，用于审计和权限判断。",
    )
    # requires_tenant_boundary 表示工具执行前是否必须确保 tenant_id 隔离。
    requires_tenant_boundary: bool = Field(
        default=True,
        description="是否要求执行前校验租户边界，防止读取或写入其它租户的数据。",
    )


class ToolSpec(BaseModel):
    """描述一个工具的 schema、权限、风险和执行策略。"""

    # name 是工具稳定 ID；ToolRouter、ToolExecutor、trace 和 eval 都靠它串联。
    name: str = Field(..., description="工具稳定名称。路由、权限校验、trace 和 eval 都通过该名称引用工具。")
    # version 用于工具 schema 演进；同名工具升级参数或输出结构时必须更新版本。
    version: str = Field(default="1.0.0", description="工具版本。用于审计、灰度和工具 schema 兼容判断。")
    # description 给路由器和人类读者说明工具适用场景。
    description: str = Field(..., description="工具业务用途说明，帮助路由器判断何时允许调用该工具。")
    # input_schema 约束工具入参，生产执行前应按该 JSON Schema 校验。
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="工具输入 JSON Schema。用于参数校验、Dify 工具节点映射和调用前审计。",
    )
    # output_schema 约束工具返回，便于 verify_tool_result 判断结果是否可用。
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="工具输出 JSON Schema。用于校验工具返回是否符合下游工作流预期。",
    )
    # risk_level 决定工具是否需要更严格审计或同步阻断。
    risk_level: RiskLevel = Field(
        default="low",
        description="工具风险等级。medium/high 工具应触发更严格的权限、同步阻断或日志策略。",
    )
    # permission 是结构化权限对象，统一承载权限等级、scope 和租户边界要求。
    permission: ToolPermissionSpec = Field(
        default_factory=ToolPermissionSpec,
        description="工具权限配置，声明调用该工具需要的权限等级、作用域和租户边界。",
    )
    # side_effect 标记工具是否会改变外部世界，例如写数据库或提交表单。
    side_effect: bool = Field(
        default=False,
        description="工具是否会产生外部副作用，例如发消息、写数据库、扣费或提交表单。",
    )
    # side_effect_level 比 bool 更细，便于区分只读、写入、外部动作和金融动作。
    side_effect_level: SideEffectLevel = Field(
        default="read_only",
        description="工具副作用等级。none/read_only 可执行；write/external_action/financial 在客户系统中直接禁止。",
    )
    # retryable 决定失败后能否自动重试，非幂等操作应关闭。
    retryable: bool = Field(
        default=True,
        description="工具失败后是否允许自动重试。非幂等写操作通常应设为 False。",
    )
    # timeout_seconds 避免外部 provider 卡住整个 workflow。
    timeout_seconds: int = Field(
        default=10,
        description="工具单次调用的超时时间，单位秒。用于防止 workflow 被外部调用阻塞。",
    )
    # timeout_ms 是生产配置主字段；timeout_seconds 保留兼容旧配置。
    timeout_ms: int = Field(
        default=10000,
        description="工具单次调用超时时间，单位毫秒。生产执行器优先使用该字段。",
    )
    # retry_policy 描述最大重试次数和退避策略，避免所有工具共用同一个重试行为。
    retry_policy: dict[str, Any] = Field(
        default_factory=lambda: {"max_attempts": 1, "backoff_ms": 200},
        description="工具重试策略，例如 max_attempts、backoff_ms。非幂等工具应只允许 1 次尝试。",
    )
    # rate_limit 为后续 Redis 限流预留，防止单租户或单用户打爆外部工具。
    rate_limit: dict[str, Any] = Field(
        default_factory=dict,
        description="工具限流配置，例如每分钟最大调用数、租户维度或用户维度限制。",
    )
    # owner 记录工具负责人，生产告警或审计时可以快速定位责任团队。
    owner: str = Field(default="agent_platform", description="工具负责人或团队，用于审计、告警和问题追踪。")
    # audit_level 决定工具调用和结果记录的详细程度。
    audit_level: AuditLevel = Field(
        default="basic",
        description="工具审计级别。full 表示需要记录更完整的参数摘要、结果摘要和策略决策。",
    )
    # idempotency_required 用于写操作，要求调用方提供幂等键防止重复提交。
    idempotency_required: bool = Field(
        default=False,
        description="工具是否要求调用方提供幂等键。涉及写入、提交、付款等场景应启用。",
    )
    # permission_scope 是兼容旧配置的字符串 scope，后续可逐步迁移到 permission.scope。
    permission_scope: str = Field(
        default="local",
        description="兼容字段：工具权限作用域。后续可统一收敛到 permission.scope。",
    )
    # error_schema 为 recovery 节点预留标准错误结构。
    error_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="工具错误返回结构。用于 recovery 判断错误类型、是否可重试和如何降级。",
    )


class ToolCall(BaseModel):
    """模型或路由节点提交给执行器的结构化工具调用。"""

    # name 必须出现在 ToolRegistry 中，executor 不允许执行未注册工具。
    name: str = Field(..., description="本次要调用的工具名称，必须能在 ToolRegistry 中找到对应 ToolSpec。")
    # arguments 是工具实际入参，由 general_tool_routing 根据 Query Understanding 构造。
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="传给工具的结构化参数。进入工具前应按 ToolSpec.input_schema 校验。",
    )
    # trace_id 把工具调用挂回同一次 Agent 运行，便于日志和 LangSmith 追踪。
    trace_id: str | None = Field(
        default=None,
        description="本次工具调用所属的 trace_id，用于把工具日志挂回同一次 Agent 运行。",
    )


class ToolResult(BaseModel):
    """工具执行、阻断或失败后返回给工作流的统一结果契约。"""

    # name 标明该结果来自哪个工具，response_package 会用它生成工具卡片。
    name: str = Field(..., description="返回结果对应的工具名称。")
    # status 把成功、失败和风控阻断明确区分，便于 recovery 采取不同策略。
    status: Literal["success", "error", "blocked"] = Field(
        ...,
        description="工具执行状态：success 表示成功，error 表示失败，blocked 表示被权限或风控拦截。",
    )
    # output 保存成功结果；失败时保持空字典，避免下游误读脏数据。
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="工具成功时返回的结构化结果。失败或被拦截时可为空。",
    )
    # error 保存失败或阻断原因，用于用户解释、日志和 recovery 判断。
    error: str | None = Field(
        default=None,
        description="工具失败或被拦截时的错误原因，供 recovery、trace 和用户解释使用。",
    )
    # latency_ms 用于工具性能分析和成本预算。
    latency_ms: int | None = Field(
        default=None,
        description="工具调用耗时，单位毫秒。用于性能分析和成本预算。",
    )
    # retry_count 记录该工具已经重试多少次，防止无限重试。
    retry_count: int = Field(
        default=0,
        description="该工具调用已经重试的次数。用于判断是否进入 recovery 或降级回答。",
    )
