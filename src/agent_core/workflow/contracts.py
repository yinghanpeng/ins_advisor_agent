"""Workflow input/output and step-level contracts.

The production rule is simple: anything that crosses a workflow step boundary
must have a contract. These Pydantic models make each node's required inputs,
outputs, retries, guardrails, and trace fields explicit.
"""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from agent_core.graph.state import AgentNode
from agent_core.guardrails.metadata import (
    rejected_public_metadata_keys,
    unknown_public_metadata_keys,
)


class StepRetryPolicy(BaseModel):
    """Retry policy attached to a workflow step."""

    # 控制这个 step 失败后是否允许自动重试，避免高风险副作用操作被重复执行。
    retryable: bool = Field(
        default=True,
        description="该 workflow step 失败后是否允许自动重试。高风险写操作通常应设为 False。",
    )
    # 限制最多尝试次数，防止工具超时、模型解析失败等问题造成无限循环。
    max_attempts: int = Field(
        default=2,
        description="该 step 最多尝试次数，包含首次执行。用于防止工具或模型调用无限循环。",
    )
    # 定义重试间隔；本地实现轻量，生产环境可以换成指数退避。
    backoff_seconds: float = Field(
        default=0.2,
        description="两次重试之间的基础等待秒数。本地实现保持轻量，生产可替换为指数退避。",
    )
    # 定义失败后进入哪个恢复状态，通常是 RECOVERY 或 ERROR。
    recovery_state: AgentNode = Field(
        default=AgentNode.RECOVERY,
        description="该 step 失败且可恢复时进入的状态节点，通常用于降级、重试或安全终止。",
    )


class WorkflowStepContract(BaseModel):
    """Contract for one workflow step/node."""

    # step 名称必须稳定，因为日志、Dify 节点、测试断言都会引用它。
    name: str = Field(
        ...,
        description="step 的稳定名称，用于日志、trace、Dify 节点映射和测试断言。",
    )
    # state 绑定显式状态机节点，避免 workflow 只靠 prompt 隐式流转。
    state: AgentNode = Field(
        ...,
        description="该 step 对应的显式状态机节点。执行引擎根据此字段记录状态路径。",
    )
    # description 用自然语言说明该 step 的业务职责，方便面试讲解和 Dify 文档映射。
    description: str = Field(
        ...,
        description="该 step 的业务职责说明，例如意图识别、工具调用、上下文组装或合规审查。",
    )
    # required_inputs 声明 step 执行前必须存在的 AgentState 字段，便于做 contract 校验。
    required_inputs: list[str] = Field(
        default_factory=list,
        description="进入该 step 前必须已经存在的状态字段，例如 input_text、intent、retrieved_context。",
    )
    # produced_outputs 声明 step 执行后应该产出的字段，防止节点职责不清。
    produced_outputs: list[str] = Field(
        default_factory=list,
        description="该 step 成功执行后应该写入或更新的状态字段，用于检查 step 边界是否清晰。",
    )
    # allowed_next_states 为状态合法性校验预留，后续可接到 AgentState.move_to。
    allowed_next_states: list[AgentNode] = Field(
        default_factory=list,
        description="该 step 完成后允许进入的后续状态。后续可同步到 AgentState.allowed_transitions 做运行时校验。",
    )
    # guardrails 列出这个 step 必须执行的风控规则，避免安全逻辑散落在 prompt 中。
    guardrails: list[str] = Field(
        default_factory=list,
        description="该 step 需要执行的安全、权限或合规规则名称，例如 prompt_injection、output_compliance。",
    )
    # tools_allowed 是工具白名单；为空代表该 step 不应该直接调用工具。
    tools_allowed: list[str] = Field(
        default_factory=list,
        description="该 step 允许调用的工具白名单。为空表示该 step 不应直接调用工具。",
    )
    # retry_policy 把失败恢复策略显式放进 contract，而不是写在节点内部隐式判断。
    retry_policy: StepRetryPolicy = Field(
        default_factory=StepRetryPolicy,
        description="该 step 的重试和恢复策略，供 workflow engine 在异常时选择 retry、recovery 或 fail。",
    )
    # trace_fields 约束日志必须携带哪些字段，保证本地日志和 LangSmith 能对齐。
    trace_fields: list[str] = Field(
        default_factory=lambda: ["trace_id", "state", "latency_ms"],
        description="该 step 写入结构化 trace 时必须携带的字段，便于本地日志和 LangSmith 对齐。",
    )


class WorkflowContract(BaseModel):
    """Named workflow contract used by Agent Core and Dify documentation."""

    # name 标识一组可复用节点契约；保险在线路由不再通过该名称选择执行图。
    name: str = Field(
        ...,
        description="节点契约集合名称。在线保险请求不使用它选择执行路径。",
    )
    # entry_state 声明这条工作流从哪个状态机节点开始。
    entry_state: AgentNode = Field(
        ...,
        description="工作流入口状态。一次请求进入 workflow engine 后应从该节点开始。",
    )
    # final_states 声明允许的终止状态，用于测试和执行引擎判断是否正常结束。
    final_states: list[AgentNode] = Field(
        ...,
        description="该工作流允许的终止状态集合，通常包含 FINAL 和 ERROR。",
    )
    # steps 是这条工作流的 step contract 清单，每个节点都必须有输入、输出、风控和重试边界。
    steps: list[WorkflowStepContract] = Field(
        ...,
        description="工作流包含的 step contract 列表。每个 step 都显式声明输入、输出、风控、工具和重试策略。",
    )


class AgentRunRequest(BaseModel):
    """公开 Agent 请求契约，限制主体字段和 default-deny metadata。"""
    # input 是外部调用进入 Agent 的原始用户问题。
    input: str = Field(
        ...,
        min_length=1,
        description="用户本轮输入的原始文本。API、CLI、Dify webhook 都会统一映射到该字段。",
    )
    # session_id 用于多轮会话和短期记忆索引。
    session_id: str = Field(
        default="anonymous_session",
        description="会话 ID，用于多轮对话、短期记忆、trace 串联和排障。",
    )
    # user_id 用于长期偏好记忆；匿名场景允许为空。
    user_id: str | None = Field(
        default=None,
        description="用户 ID。登录态可传入真实 ID，匿名调试可为空。",
    )
    # tenant_id 用于多租户隔离，避免不同团队共享记忆或知识库。
    tenant_id: str = Field(
        default="local",
        description="租户 ID。用于企业、团队或渠道隔离，避免跨租户读取记忆和知识库。",
    )
    # workflow_name 为旧 Dify/API 客户端保留；当前统一执行图不会按它分叉保险逻辑。
    workflow_name: str = Field(
        default="universal_agent_workflow",
        description="兼容运行标签，默认 universal_agent_workflow；不参与保险意图或 Handler 选择。",
    )
    # domain_skill 是调用提示和响应字段，不能绕过 Input Guardrail 或意图白名单。
    domain_skill: str | None = Field(
        default=None,
        description="可选领域提示；最终 Skill 仍由安全检查后的白名单意图路由决定。",
    )
    # metadata 只保存不参与生成或数据选行的调用端信息；知识正文、新闻和业务 ID 都受保护。
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "请求扩展信息，仅允许 source、client、channel、experiment_group、eval_id、request_id、locale。"
            "公开请求不得通过该字段注入知识、新闻、对话模式、业务记录 ID 或内部信任标志。"
        ),
    )

    @field_validator("metadata")
    @classmethod
    def reject_protected_public_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """拒绝可影响生成内容或业务记录选择的客户 metadata 键。"""

        # 校验发生在公开请求契约边界；一旦发现受保护键，整次请求直接验证失败，
        # 而不是静默删除后继续运行，便于调用端发现误用或攻击并留下标准 422 记录。
        rejected_keys = rejected_public_metadata_keys(value)
        # 受保护键可以改变生成内容或业务记录选择，发现任一项就拒绝整个公开请求。
        if rejected_keys:
            # 错误只列键名，不回显恶意正文，避免把注入内容带入 API 日志。
            joined_keys = ", ".join(rejected_keys)
            # 以 Pydantic ValueError 形式返回标准请求体验证失败，不进入工作流。
            raise ValueError(f"metadata 包含禁止的受保护字段: {joined_keys}")
        # 对未识别键采用 default deny；否则以后新增内部 metadata 控制字段时，公开 API 会在未审计的
        # 情况下自动获得该能力。允许列表只包含不选数据、不改流程的观测标签。
        unknown_keys = unknown_public_metadata_keys(value)
        # 即使键当前没有已知风险，只要不在公开允许列表中也默认拒绝。
        if unknown_keys:
            # 对键名排序结果做稳定拼接，同样不回显对应值。
            joined_keys = ", ".join(unknown_keys)
            # 抛出契约验证错误，要求调用方删除未审计扩展字段。
            raise ValueError(f"metadata 包含未允许的字段: {joined_keys}")
        # 返回浅拷贝，避免外部调用方在模型创建后修改原 dict，绕过本次验证结果。
        return dict(value)


class AgentRunResponse(BaseModel):
    """进程内完整诊断响应；FastAPI 会进一步投影为客户安全 DTO。"""
    # trace_id 是外部排障入口，可以关联日志、LangSmith 和本地 eval 结果。
    trace_id: str = Field(
        ...,
        description="本次请求的追踪 ID。前端、日志、LangSmith 和 eval 都可以用它定位同一次运行。",
    )
    # session_id 回传给调用方，方便前端下一轮继续同一个会话。
    session_id: str = Field(
        ...,
        description="响应对应的会话 ID，便于调用方继续多轮对话或查询历史状态。",
    )
    # final_state 说明本轮最终停在哪个状态，便于识别 FINAL 或 ERROR。
    final_state: str = Field(
        ...,
        description="工作流最终状态，通常为 FINAL 或 ERROR。",
    )
    # answer 是最终用户可读文本；即使被阻断也会返回阻断说明。
    answer: str = Field(
        ...,
        description="最终返回给用户的可读回答。被合规拦截或降级时也会在这里说明原因。",
    )
    # intent 让前端和评估脚本知道本轮路由判断。
    intent: str | None = Field(
        default=None,
        description="意图识别结果，例如 weather_query、insurance_break_ice、insurance_objection_handling。",
    )
    # domain_skill 表示实际命中的业务能力，例如 insurance_advisor。
    domain_skill: str | None = Field(
        default=None,
        description="实际命中的业务 Skill。通用问题可能为空。",
    )
    # intent_routing_result 暴露阈值层、来源和候选分数，便于线上评估但不包含客户槽位原值。
    intent_routing_result: dict[str, Any] = Field(
        default_factory=dict,
        description="双层意图路由摘要，包含向量相似度、裁定置信度和执行档位。",
    )
    # active_intent 只返回控制信封，不返回 KYC 事实；前端可据此展示当前正在补充的方向。
    active_intent: dict[str, Any] = Field(
        default_factory=dict,
        description="当前保险活跃意图控制状态；任务完成或取消后为空。",
    )
    # insurance_kyc_status 只暴露缺失字段名和分数，不返回资产、家庭等槽位值。
    insurance_kyc_status: dict[str, Any] = Field(
        default_factory=dict,
        description="保险代码路径的状态摘要，不包含敏感槽位值。",
    )
    # guardrails 汇总输入、工具、输出风控结果。
    guardrails: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本次运行触发或通过的安全合规结果，供前端提示、审计和 eval 使用。",
    )
    # retrieved_context 返回本轮用过的知识证据，便于解释和引用。
    retrieved_context: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本次运行使用过的检索证据摘要，通常包含来源、分数、片段 ID 和 metadata。",
    )
    # trace_events 是完整事件流，适合回放和排障。
    trace_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="完整结构化 trace 事件，包含状态切换、工具、检索、风控、恢复和成本等事件。",
    )
    # stream_events 是面向未来 SSE 的事件骨架，比 trace_events 更适合前端按节点/工具展示进度。
    stream_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="流式事件骨架，包含节点开始/结束、工具调用、最终答案等可转 SSE 的事件。",
    )
    # state_transitions 只记录状态跳转路径，不混入工具和检索细节。
    state_transitions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="仅记录状态从哪里到哪里、为什么跳转的审计链路，不混入工具或检索细节。",
    )
    # tool_calls 是工具调用审计，强调输入、状态、耗时和错误。
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本次运行实际发生的工具调用审计记录，包含工具名、输入、状态、耗时和错误。",
    )
    # tool_results 是可用于回答生成和前端工具卡片的结构化工具结果。
    tool_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="工具执行后可供回答和前端展示消费的结构化结果。",
    )
    # query_understanding 暴露检索前处理过程，例如实体、时间范围和 filters。
    query_understanding: dict[str, Any] = Field(
        default_factory=dict,
        description="Query Understanding 结果，例如指代消解、实体、时间范围、改写 query 和 filters。",
    )
    # context_needs 解释本轮为什么需要或不需要 memory、RAG、tool、安全降级或拒答。
    context_needs: dict[str, bool] = Field(
        default_factory=dict,
        description="上下文需求规划结果，说明本轮是否需要 memory、rag、tool、safe_response、reject 或 clarify。",
    )
    # response_package 面向前端组件，比完整 AgentState 更轻。
    response_package: dict[str, Any] = Field(
        default_factory=dict,
        description="前端可展示的响应包，包含 answer、citations、tool_cards、next_actions、risk_level 和 trace_id。",
    )
    # grounding_result 返回事实校验结论，帮助调用方判断回答是否可靠。
    grounding_result: dict[str, Any] = Field(
        default_factory=dict,
        description="事实校验结果，说明回答是否有证据支撑、引用了哪些来源、是否存在冲突。",
    )
    # evaluation_result 暴露回答质量评估和重生成触发原因，便于生产排障与测试断言。
    evaluation_result: dict[str, Any] = Field(
        default_factory=dict,
        description="回答质量评估结果，记录是否触发重生成、触发原因和重生成次数。",
    )
    # output_pii_scan_result 暴露输出侧 PII 扫描摘要，绝不包含原始敏感文本。
    output_pii_scan_result: dict[str, Any] = Field(
        default_factory=dict,
        description="输出侧 PII 扫描结果摘要，只包含 PII 类型、位置摘要、动作和高敏标记。",
    )
    # cost 返回本轮资源消耗摘要，便于预算和性能分析。
    cost: dict[str, Any] = Field(
        default_factory=dict,
        description="本次运行的成本与资源消耗摘要，例如预算、工具调用次数、压缩后上下文长度。",
    )


class PublicAgentRunResponse(BaseModel):
    """Customer-safe HTTP response separated from the internal diagnostic DTO."""

    # trace_id 允许客服关联服务端日志，但不会把完整 trace payload 暴露给客户。
    trace_id: str = Field(..., description="供客户反馈问题时关联服务端日志的追踪 ID。")
    # session_id 用于继续下一轮；生产环境必须使用网关签发且绑定登录主体的值。
    session_id: str = Field(..., description="下一轮继续会话使用的 ID；生产部署必须由网关绑定到登录主体。")
    # final_state 只暴露最终状态，不泄露内部节点迁移路径。
    final_state: str = Field(..., description="本轮终态，通常为 FINAL 或 ERROR。")
    # answer 在封装响应前已经通过输出 PII 扫描和合规检查。
    answer: str = Field(..., description="已经过输出 PII 与合规检查的客户可读回答。")
    # intent 供 UI 展示路由结果，但不携带向量候选或模型置信分。
    intent: str | None = Field(default=None, description="本轮最终意图标签；不包含内部候选和分数。")
    # domain_skill 只返回最终命中的能力标签，不暴露内部领域路由过程。
    domain_skill: str | None = Field(default=None, description="本轮实际命中的领域 Skill。")
    # active_intent 仅包含字段名和控制时间，不包含任何 KYC 实际值。
    active_intent: dict[str, Any] = Field(
        default_factory=dict,
        description="保险多轮控制信封，只包含意图状态、待问字段名、已问字段名和过期时间。",
    )
    # insurance_kyc_status 只保留完整度控制信息，不返回家庭或资产答案。
    insurance_kyc_status: dict[str, Any] = Field(
        default_factory=dict,
        description="保险 KYC 状态摘要，只包含信息状态、缺失字段名、已问字段名和轮次。",
    )
    # citations 只返回证据标识，实际检索片段正文继续保留在服务端。
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="脱敏引用标识，只包含 source/chunk/risk 等轻量字段，不返回知识正文。",
    )
    # next_actions 是前端可直接展示的低风险后续建议列表。
    next_actions: list[str] = Field(default_factory=list, description="前端可展示的低风险下一步建议。")
    # warnings 汇总可公开的证据不足、降级和合规改写提示。
    warnings: list[str] = Field(default_factory=list, description="证据不足、降级或合规改写等公开提示。")
    # clarification_question 只在本轮需要继续补充信息时返回单个问题。
    clarification_question: str | None = Field(
        default=None,
        description="低置信或参数缺失时需要客户补充的一句澄清问题。",
    )

    @classmethod
    def from_internal(cls, response: AgentRunResponse) -> "PublicAgentRunResponse":
        """Project an internal response into the default-deny customer contract."""

        # response_package 在输出检查后生成，内部 citations/next_actions 已经过安全整理；
        # 此处刻意忽略工具卡、trace、检索正文、Query 改写、成本和模型评估细节。
        package = response.response_package
        # 活跃意图置信度/来源和 KYC 机会分/完整度分属于内部客户画像，不能返回客户；
        # 下面只投影 UI 继续多轮所需的控制字段。
        safe_active_intent = {
            # 对固定允许列表逐项取值，新增内部字段不会自动进入公开响应。
            key: response.active_intent[key]
            for key in ["intent", "status", "pending_focus", "asked_focuses", "expires_at"]
            # 仅包含内部响应实际存在的键，避免可选字段访问异常。
            if key in response.active_intent
        }
        # 同样以默认拒绝方式投影 KYC 状态，只保留缺失字段名和轮次等控制信息。
        safe_kyc_status = {
            # 从内部状态读取当前白名单键对应的值。
            key: response.insurance_kyc_status[key]
            for key in [
                "information_status",
                "missing_fields",
                "asked_focuses",
                "kyc_question_round_count",
            ]
            # 忽略本次响应没有生成的可选键。
            if key in response.insurance_kyc_status
        }
        # 构造客户专用 DTO；未显式列出的内部字段不会被序列化。
        return cls(
            trace_id=response.trace_id,
            session_id=response.session_id,
            final_state=response.final_state,
            answer=response.answer,
            intent=response.intent,
            domain_skill=response.domain_skill,
            active_intent=safe_active_intent,
            insurance_kyc_status=safe_kyc_status,
            citations=list(package.get("citations") or []),
            next_actions=list(package.get("next_actions") or []),
            warnings=list(package.get("warnings") or []),
            clarification_question=package.get("clarification_question"),
        )


class AgentRunExecutionContext(BaseModel):
    """代码侧受信运行参数；公开 API 不接收该对象。"""

    # request_token_budget 只允许受信调用方缩放本次运行预算，不能覆盖身份、权限或 Artifact Snapshot。
    request_token_budget: int | None = Field(
        default=None,
        ge=1,
        description="代码侧注入的单次请求 Token 预算；公开 AgentRunRequest 不暴露该字段。",
    )


# EVAL_RULE_NAMES 是本地确定性评分器支持的稳定规则 ID；数据集使用未知 ID 时在加载阶段立即失败。
EVAL_RULE_NAMES = frozenset(
    {
        "answer",
        "schema",
        "state",
        "intent",
        "sales_route",
        "tools",
        "guardrail",
        "trace",
        "cost",
        "trajectory",
        # judge 仅在 Runner 显式开启 --enable-llm-judge 时执行；未开启时声明了 judge 会记为 skip/失败策略见门禁配置。
        "judge",
    }
)

# EVAL_MATURITY_VALUES 区分“当前应过的回归”与“产品尚未齐备的前瞻样本”，避免混为一谈。
EVAL_MATURITY_VALUES = frozenset({"current", "aspirational"})


class EvalCase(BaseModel):
    """离线评估样本契约，声明输入、预期路径和通过/失败规则。"""
    # id 是离线评估样本的稳定主键，便于失败定位和报告关联。
    id: str = Field(..., description="评估样本 ID。用于定位失败 case 和生成评估报告。")
    # type 表示该样本主要验证的能力类别，供 evaluator 选择附加断言。
    type: str = Field(..., description="评估类型，例如 route、guardrail、rag、sales_quality 或 recovery。")
    # suite 把同类 Case 聚合成独立指标，避免总体通过率掩盖安全或路由子集退化。
    suite: str = Field(
        default="regression",
        description="评估套件名称，例如 safety、routing、tools、rag、memory 或 business_quality。",
    )
    # maturity 区分当前回归门禁与前瞻能力样本：aspirational 默认不阻断 Promote，但仍计入报告。
    maturity: str = Field(
        default="current",
        description="current=当前产品应通过的回归；aspirational=能力路线图样本，默认不阻断发布门禁。",
    )
    # input 是送入完整 Agent 链路的原始测试文本。
    input: str = Field(..., description="喂给 Agent 的用户输入，用于复现完整工作流。")
    # turns 非空时表示完整多轮用户输入；同一 Trial 内复用隔离 Engine 与 Session，并只评分最后一轮。
    turns: list[str] = Field(
        default_factory=list,
        description="可选多轮用户输入；非空时替代 input 作为按顺序执行的完整对话。",
    )
    # initial_state 仅允许映射到正式请求字段或受信执行预算，禁止任意修改 AgentState。
    initial_state: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "受控评估夹具，只允许 session_id、user_id、tenant_id、workflow_name、domain_skill、"
            "source、metadata 和 request_token_budget。"
        ),
    )
    # expected_state 声明预期终态，None 表示该样本不检查状态名称。
    expected_state: str | None = Field(
        default=None,
        description="期望最终进入的状态名称，例如 FINAL 或 ERROR。",
    )
    # expected_tools 列出预期实际执行的工具，空列表表示不强制检查工具集合。
    expected_tools: list[str] = Field(
        default_factory=list,
        description="必须出现的工具名称列表；允许实际轨迹包含额外工具。",
    )
    # forbidden_tools 明确声明绝不能调用的工具，适合测试越权、过度搜索和无需工具的普通回答。
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="实际工具轨迹中禁止出现的工具名称列表。",
    )
    # expected_intent 直接验证统一意图路由标签，避免仅靠回答关键词掩盖误路由。
    expected_intent: str | None = Field(
        default=None,
        description="期望的统一意图标签，例如 calculator_query 或 insurance_break_ice。",
    )
    # expected_domain_skill 验证实际领域处理器；None 表示不检查而不是强制通用链路。
    expected_domain_skill: str | None = Field(
        default=None,
        description="期望命中的领域 Skill；为空表示该 Case 不执行此项断言。",
    )
    # expected_sales_intelligence_route 验证销售智能内部路由标签是否符合预期。
    expected_sales_intelligence_route: str | None = Field(
        default=None,
        description="期望命中的销售智能路由，例如 break_ice、kyc_question、objection_handling。",
    )
    # must_include 保存最终答案至少应命中的关键文本集合。
    must_include: list[str] = Field(
        default_factory=list,
        description="最终回答中必须包含的关键词或关键表达。",
    )
    # must_include_any 用“同义组任一命中”降低单一关键词造成的假失败/假通过。
    must_include_any: list[list[str]] = Field(
        default_factory=list,
        description=(
            "最终回答必须命中的同义表达组列表；每一组内任一关键词出现即算该组通过。"
            "例如 [[\"资金\",\"资产\",\"理财\"],[\"低压\",\"不逼单\"]]。"
        ),
    )
    # must_not_include 保存最终答案绝不能出现的高风险或错误表达。
    must_not_include: list[str] = Field(
        default_factory=list,
        description="最终回答中禁止出现的词语，例如保证收益、避税避债等高风险表达。",
    )
    # judge_rubric 声明主观质量维度；只有启用 LLM Judge 且 pass_fail_rules 含 judge 时才评分。
    judge_rubric: str = Field(
        default="",
        description="LLM-as-Judge 评分量表说明，例如表达自然、策略得体、合规不越权。空表示不启用主观评分。",
    )
    # expected_guardrail 指定必须执行并记录的风控规则；None 时不做单规则断言。
    expected_guardrail: str | None = Field(
        default=None,
        description="期望执行并记录的风控规则名称；是否触发由 expected_guardrail_triggered 单独声明。",
    )
    # expected_guardrail_action 在命中规则后继续验证 pass、safe_fallback 或 block 等执行动作。
    expected_guardrail_action: str | None = Field(
        default=None,
        description="指定 Guardrail 应产生的动作，例如 pass、safe_fallback 或 block。",
    )
    # expected_guardrail_triggered 区分应该拦截和应该正常通过的正反安全样本。
    expected_guardrail_triggered: bool | None = Field(
        default=None,
        description="指定目标 Guardrail 的 triggered 值；为空表示只检查规则是否执行。",
    )
    # expected_trace_fields 列出结构化 trace 必须包含的字段名称。
    expected_trace_fields: list[str] = Field(
        default_factory=list,
        description="响应或任一 trace event 中必须出现的字段路径，用于验证可观测性完整性。",
    )
    # required_states 使用里程碑子序列约束必要节点，同时允许实现选择额外的合法中间节点。
    required_states: list[str] = Field(
        default_factory=list,
        description="状态轨迹必须按给定顺序经过的节点子序列。",
    )
    # forbidden_states 约束安全短路或无需工具的路径绝不能进入某些执行节点。
    forbidden_states: list[str] = Field(
        default_factory=list,
        description="状态轨迹中禁止出现的节点名称。",
    )
    # expected_cost 以键值方式检查预算决策，例如 request_token_budget=500 或 budget_pressure=true。
    expected_cost: dict[str, Any] = Field(
        default_factory=dict,
        description="期望在响应 cost 摘要中出现的精确键值。",
    )
    # max_tool_calls 为工具总调用数设置硬上限，防止循环或不必要的外部访问。
    max_tool_calls: int | None = Field(
        default=None,
        ge=0,
        description="允许的最大工具调用次数；为空表示不检查。",
    )
    # trials 对非确定性 Case 重复运行，并采用所有 Trial 均通过的稳定性门槛。
    trials: int = Field(
        default=1,
        ge=1,
        le=10,
        description="该 Case 的独立运行次数；所有 Trial 均通过时 Case 才通过。",
    )
    # pass_fail_rules 列出本 Case 启用的稳定评分器 ID；未知 ID 会在数据加载阶段报错。
    pass_fail_rules: list[str] = Field(
        default_factory=list,
        description="启用的确定性评分器 ID，例如 answer、state、tools、guardrail、trace 或 cost。",
    )

    @field_validator("maturity")
    @classmethod
    def validate_maturity(cls, value: str) -> str:
        """只允许 current / aspirational，防止拼写错误把门禁样本静默降级。"""

        # normalized 去掉首尾空白后做精确匹配，避免 YAML/JSON 粘贴引入不可见空格。
        normalized = value.strip()
        # 未知成熟度会让 Promote 阈值语义不确定，必须在加载阶段失败。
        if normalized not in EVAL_MATURITY_VALUES:
            # 错误只回显稳定枚举，不回显 Case 输入正文。
            raise ValueError(
                f"EvalCase.maturity 必须是 {', '.join(sorted(EVAL_MATURITY_VALUES))}，收到: {normalized}"
            )
        # 返回规范化后的成熟度标签。
        return normalized

    @field_validator("turns")
    @classmethod
    def validate_turns(cls, value: list[str]) -> list[str]:
        """拒绝空白多轮输入，避免 Runner 执行不可复现的空 Turn。"""

        # 任一 Turn 为空都会使会话轨迹和期望轮次含义不明确，因此整条 Case 直接无效。
        if any(not item.strip() for item in value):
            # 数据错误应在运行 Agent 前暴露，不能把空输入误记为模型或 Guardrail 失败。
            raise ValueError("EvalCase.turns 不能包含空白输入")
        # 返回原顺序副本，确保 Runner 按数据集声明顺序执行多轮输入。
        return list(value)

    @field_validator("must_include_any")
    @classmethod
    def validate_must_include_any(cls, value: list[list[str]]) -> list[list[str]]:
        """同义组必须非空，且每组至少有一个非空白关键词。"""

        # normalized 保存清洗后的组列表，供 Runner 和 Evaluator 稳定消费。
        normalized: list[list[str]] = []
        # 逐组校验，定位错误时保留组下标。
        for group_index, group in enumerate(value):
            # 空组无法表达“任一命中”语义。
            if not group:
                # 明确指出哪一组为空，方便修 JSONL。
                raise ValueError(f"EvalCase.must_include_any[{group_index}] 不能为空组")
            # terms 去掉空白词，避免数据集里混入无意义空串。
            terms = [term.strip() for term in group if term and term.strip()]
            # 清洗后若无有效词，同样视为无效组。
            if not terms:
                # 与空组使用相同错误级别，阻止脏数据进入执行。
                raise ValueError(f"EvalCase.must_include_any[{group_index}] 缺少有效关键词")
            # 保留组内首次出现顺序，便于报告 expected 字段可读。
            normalized.append(list(dict.fromkeys(terms)))
        # 返回全部合法同义组。
        return normalized

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        """只允许可安全映射到正式入口的受控评估夹具。"""

        # allowed_keys 不包含 profile、工具结果或内部信任标志，防止 Eval 悄悄绕过正式运行链路。
        allowed_keys = {
            "session_id",
            "user_id",
            "tenant_id",
            "workflow_name",
            "domain_skill",
            "source",
            "metadata",
            "request_token_budget",
        }
        # unknown_keys 使用稳定排序，让 JSONL 加载错误在本地和 CI 中保持可比较。
        unknown_keys = sorted(set(value) - allowed_keys)
        # 任意未知键都意味着 Case 试图注入尚未建模的内部状态，必须先扩展受信契约。
        if unknown_keys:
            # 只回显键名，不回显可能包含客户数据的夹具值。
            raise ValueError(f"EvalCase.initial_state 包含未允许字段: {', '.join(unknown_keys)}")
        # metadata 必须保持字典形态，后续仍会经过 AgentRunRequest 的 default-deny 校验。
        if "metadata" in value and not isinstance(value["metadata"], dict):
            # 非字典 metadata 无法安全合并 eval_id 和 source 标签。
            raise ValueError("EvalCase.initial_state.metadata 必须是字典")
        # 返回浅拷贝，避免 Runner 执行期间修改数据集模型持有的原始字典。
        return dict(value)

    @field_validator("pass_fail_rules")
    @classmethod
    def validate_pass_fail_rules(cls, value: list[str]) -> list[str]:
        """拒绝没有实现的自然语言规则，防止报告把未评分条件当作通过。"""

        # unsupported_rules 收集全部未知规则，一次性报告比逐个修复更适合维护 JSONL 数据集。
        unsupported_rules = sorted(set(value) - EVAL_RULE_NAMES)
        # 未知规则没有对应评分逻辑时整条 Case 不可执行，不能静默忽略。
        if unsupported_rules:
            # 错误只列稳定规则 ID，不包含用户输入或回答正文。
            raise ValueError(f"EvalCase.pass_fail_rules 包含未实现规则: {', '.join(unsupported_rules)}")
        # 去重同时保持首次出现顺序，避免同一评分器重复执行和重复计分。
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_rule_coverage(self) -> "EvalCase":
        """显式规则列表必须覆盖所有非空结构化期望，禁止静默漏评。"""

        # 空规则列表保留旧调用方的兼容语义：Evaluator 会执行全部适用的确定性断言。
        if not self.pass_fail_rules:
            # 返回已完成字段校验的当前模型。
            return self
        # required_rules 根据真正声明了期望的字段计算，不要求 Case 启用没有验收目标的维度。
        required_rules: set[str] = set()
        # 关键文本非空时必须启用 answer，确保业务与合规文案确实参与通过判定。
        if self.must_include or self.must_include_any or self.must_not_include:
            # 将答案评分器加入必要集合。
            required_rules.add("answer")
        # 声明了 Judge 量表时必须显式启用 judge，避免主观期望被静默忽略。
        if self.judge_rubric.strip():
            # 将 LLM Judge 评分器加入必要集合；是否真正调用模型由 Runner 开关控制。
            required_rules.add("judge")
        # 期望终态非空时必须启用 state，避免 ERROR/FINAL 约束被漏掉。
        if self.expected_state is not None:
            # 将终态评分器加入必要集合。
            required_rules.add("state")
        # 统一意图或领域 Skill 任一非空都由 intent 评分器负责。
        if self.expected_intent is not None or self.expected_domain_skill is not None:
            # 将统一路由评分器加入必要集合。
            required_rules.add("intent")
        # 旧销售场景期望使用独立兼容评分器，不能由 intent 自动替代。
        if self.expected_sales_intelligence_route is not None:
            # 将销售路由评分器加入必要集合。
            required_rules.add("sales_route")
        # 必调、禁调或次数上限任一存在时都必须启用工具评分器。
        if self.expected_tools or self.forbidden_tools or self.max_tool_calls is not None:
            # 将工具评分器加入必要集合。
            required_rules.add("tools")
        # Guardrail 名称、动作或触发状态任一被声明时必须启用安全评分器。
        if (
            self.expected_guardrail is not None
            or self.expected_guardrail_action is not None
            or self.expected_guardrail_triggered is not None
        ):
            # 将安全评分器加入必要集合。
            required_rules.add("guardrail")
        # Trace 字段列表非空时必须启用可观测性评分器。
        if self.expected_trace_fields:
            # 将 Trace 评分器加入必要集合。
            required_rules.add("trace")
        # 必经或禁止状态任一非空时必须启用轨迹评分器。
        if self.required_states or self.forbidden_states:
            # 将轨迹评分器加入必要集合。
            required_rules.add("trajectory")
        # 成本期望非空时必须启用成本评分器。
        if self.expected_cost:
            # 将成本评分器加入必要集合。
            required_rules.add("cost")
        # missing_rules 是数据声明了期望却不会参与评分的危险配置。
        missing_rules = sorted(required_rules - set(self.pass_fail_rules))
        # 任何漏评都在执行 Agent 前失败，不能生成虚假的绿色报告。
        if missing_rules:
            # 错误仅列规则 ID，JSONL Loader 会进一步补充精确文件与行号。
            raise ValueError(
                "EvalCase.pass_fail_rules 未覆盖已声明的结构化期望: "
                + ", ".join(missing_rules)
            )
        # 返回规则覆盖完整的模型。
        return self
