"""Workflow input/output and step-level contracts.

The production rule is simple: anything that crosses a workflow step boundary
must have a contract. These Pydantic models make each node's required inputs,
outputs, retries, guardrails, and trace fields explicit.
"""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_core.graph.state import AgentNode


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
    # 定义失败后进入哪个恢复状态，通常是 RECOVERY 或 HUMAN_APPROVAL。
    recovery_state: AgentNode = Field(
        default=AgentNode.RECOVERY,
        description="该 step 失败且可恢复时进入的状态节点，通常用于降级、重试或人工审批。",
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

    # 工作流名称用于区分通用 Agent、保险顾问、销售语料导入等不同图。
    name: str = Field(
        ...,
        description="工作流名称。用于区分通用 Agent、保险顾问、销售语料导入等不同执行图。",
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
    # workflow_name 允许调用方指定工作流，默认走通用 Agent 主链路。
    workflow_name: str = Field(
        default="universal_agent_workflow",
        description="希望执行的工作流名称。未指定时走通用 Agent 工作流。",
    )
    # domain_skill 可强制指定业务 Skill，也可留空让路由器自动判断。
    domain_skill: str | None = Field(
        default=None,
        description="显式指定的业务 Skill，例如 insurance_advisor；为空时由路由器自动判断。",
    )
    # metadata 保存调用端和实验信息，避免为了调试字段不断修改核心契约。
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="请求扩展信息，例如 client=dify、channel=api、debug=true 或实验分组。",
    )


class AgentRunResponse(BaseModel):
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
    # final_state 说明本轮最终停在哪个状态，便于识别 FINAL、ERROR 或 HUMAN_APPROVAL。
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
        description="意图识别结果，例如 weather_query、break_ice_help、objection_handling。",
    )
    # domain_skill 表示实际命中的业务能力，例如 insurance_advisor。
    domain_skill: str | None = Field(
        default=None,
        description="实际命中的业务 Skill。通用问题可能为空。",
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
    # context_needs 解释本轮为什么需要或不需要 memory、RAG、tool、人审或拒答。
    context_needs: dict[str, bool] = Field(
        default_factory=dict,
        description="上下文需求规划结果，说明本轮是否需要 memory、rag、tool、human、reject 或 clarify。",
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


class EvalCase(BaseModel):
    id: str = Field(..., description="评估样本 ID。用于定位失败 case 和生成评估报告。")
    type: str = Field(..., description="评估类型，例如 route、guardrail、rag、sales_quality 或 recovery。")
    input: str = Field(..., description="喂给 Agent 的用户输入，用于复现完整工作流。")
    initial_state: dict[str, Any] = Field(
        default_factory=dict,
        description="评估开始前预置到 AgentState 的状态字段，例如 profile、metadata 或 cost。",
    )
    expected_state: str | None = Field(
        default=None,
        description="期望最终进入的状态名称，例如 FINAL、ERROR 或 HUMAN_APPROVAL。",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="期望调用的工具名称列表。用于验证工具路由和权限控制。",
    )
    expected_sales_intelligence_route: str | None = Field(
        default=None,
        description="期望命中的销售智能路由，例如 break_ice、kyc_question、objection_handling。",
    )
    must_include: list[str] = Field(
        default_factory=list,
        description="最终回答中必须包含的关键词或关键表达。",
    )
    must_not_include: list[str] = Field(
        default_factory=list,
        description="最终回答中禁止出现的词语，例如保证收益、避税避债等高风险表达。",
    )
    expected_guardrail: str | None = Field(
        default=None,
        description="期望触发的风控规则名称。为空表示该样本不强制要求触发特定规则。",
    )
    expected_trace_fields: list[str] = Field(
        default_factory=list,
        description="trace_events 中必须出现的字段名，用于验证可观测性是否完整。",
    )
    pass_fail_rules: list[str] = Field(
        default_factory=list,
        description="该样本额外的通过/失败规则说明，供本地 evaluator 或人工复核使用。",
    )
