"""业务记忆系统的结构化数据模型。

本文件把 Dify 工作流里的字符串变量升级为可持久化、可审计、可评测的
Pydantic v2 schema。

设计原则：
1. 真实销售/客户对话不是普通 RAG 知识库；
2. 原始对话只能作为证据来源和训练/评测资产，不直接进入最终 Prompt；
3. 长期事实必须带来源、证据、置信度和租户边界；
4. 不确定线索必须标记为 uncertain，不能混入 confirmed facts；
5. PII 只保存引用或 hash，默认不进入生成上下文。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_core.utils.ids import new_id
from agent_core.utils.time import utc_now_iso


Certainty = Literal["confirmed", "uncertain"]
SensitivityLevel = Literal["public", "internal", "sensitive", "pii"]
FactSourceType = Literal["user_message", "advisor_message", "analysis", "manual_review", "imported_corpus"]
InformationStatus = Literal["insufficient", "matched", "unmatched"]
SubjectType = Literal["customer", "channel", "unclear"]
TargetPersona = Literal["enterprise_owner", "executive", "family_planner", "channel", "unknown"]
AdvisorStage = Literal["newbie", "transitioning", "part_time", "experienced", "unknown"]
ExternalGrade = Literal["A", "B", "C", "D"]
TriggerModule = Literal[
    "cashflow_pressure",
    "investment_fantasy",
    "compliance_boundary",
    "family_responsibility",
    "interest_rate_stability",
    "overseas_multi_currency",
    "channel_referral",
    "light_touch",
    "unknown",
]
CurrentStage = Literal["collect_kyc", "deep_conversation", "cultivate", "low_pressure_end"]
MemoryEventType = Literal[
    "customer_objection",
    "customer_positive_signal",
    "advisor_confidence_signal",
    "trigger_event",
    "relationship_update",
    "strategy_generated",
    "strategy_used",
    "outcome_update",
    "fact_correction",
]
CaseOutcomeType = Literal[
    "no_response",
    "replied",
    "call_booked",
    "meeting_booked",
    "info_requested",
    "objection",
    "rejected",
    "deal_created",
    "closed_won",
    "closed_lost",
    "long_term_nurture",
]


class Tenant(BaseModel):
    """租户模型，用于隔离不同团队、机构或渠道的数据。"""

    id: str = Field(default_factory=lambda: new_id("tenant"), description="租户唯一 ID。")
    tenant_id: str = Field(..., description="业务租户 ID，所有业务表必须携带该字段。")
    display_name: str = Field(default="", description="租户展示名称，例如分公司、团队或机构名称。")
    created_at: str = Field(default_factory=utc_now_iso, description="租户记录创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="租户记录最近更新时间。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="租户扩展信息。")


class Advisor(BaseModel):
    """使用本系统的顾问/从业者。"""

    id: str = Field(default_factory=lambda: new_id("advisor"), description="顾问唯一 ID。")
    tenant_id: str = Field(..., description="顾问所属租户 ID。")
    display_name: str = Field(default="", description="顾问展示名称，可为空。")
    display_alias: str = Field(default="", description="顾问对外别名，避免在 Prompt 中暴露真实姓名。")
    created_at: str = Field(default_factory=utc_now_iso, description="顾问记录创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="顾问记录最近更新时间。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="顾问扩展信息。")


class Customer(BaseModel):
    """被分析的客户或渠道对象。

    真实姓名、手机号、微信号、身份证、精确地址等 PII 不直接进入生成上下文。
    生产环境应保存 pii_ref_id 或 hash，并在权限允许时单独解析。
    """

    id: str = Field(default_factory=lambda: new_id("customer"), description="客户唯一 ID。")
    tenant_id: str = Field(..., description="客户所属租户 ID。")
    display_alias: str = Field(default="", description="客户展示别名，例如 客户A、制造业企业主。")
    pii_ref_id: str | None = Field(default=None, description="PII 引用 ID，默认不进入 Prompt。")
    phone_hash: str | None = Field(default=None, description="手机号 hash，用于去重，不进入 Prompt。")
    email_hash: str | None = Field(default=None, description="邮箱 hash，用于去重，不进入 Prompt。")
    created_at: str = Field(default_factory=utc_now_iso, description="客户记录创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="客户记录最近更新时间。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="客户扩展信息。")


class AdvisorProfileFact(BaseModel):
    """从业者画像事实，用于替代单一 practitioner_state 字符串。"""

    id: str = Field(default_factory=lambda: new_id("advisor_fact"), description="从业者事实唯一 ID。")
    tenant_id: str = Field(..., description="事实所属租户 ID。")
    advisor_id: str = Field(..., description="事实对应的从业者 ID。")
    fact_key: str = Field(..., description="事实键，例如 role、experience_years、confidence_barrier。")
    fact_value: Any = Field(..., description="事实值。可以是字符串、数字、列表或结构化对象。")
    confidence: float = Field(default=1.0, ge=0, le=1, description="事实置信度，0 到 1。")
    source_type: FactSourceType = Field(..., description="事实来源类型。")
    source_conversation_id: str | None = Field(default=None, description="来源会话 ID。")
    source_message_id: str | None = Field(default=None, description="来源消息 ID。")
    evidence_text: str = Field(..., description="支持该事实的原文证据摘录，不得为空。")
    is_current: bool = Field(default=True, description="是否为当前有效事实。冲突旧事实会置为 False。")
    valid_from: str = Field(default_factory=utc_now_iso, description="事实生效时间。")
    valid_to: str | None = Field(default=None, description="事实失效时间；当前事实为空。")
    created_at: str = Field(default_factory=utc_now_iso, description="事实创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="事实最近更新时间。")


class CustomerProfileFact(BaseModel):
    """客户画像事实，用于替代单一 profile_state 字符串。"""

    id: str = Field(default_factory=lambda: new_id("customer_fact"), description="客户事实唯一 ID。")
    tenant_id: str = Field(..., description="事实所属租户 ID。")
    customer_id: str = Field(..., description="事实对应的客户 ID。")
    fact_key: str = Field(..., description="事实键，例如 age、industry、children、financial_preference。")
    fact_value: Any = Field(..., description="原始事实值。")
    normalized_value: Any | None = Field(default=None, description="标准化事实值，例如年龄段或标签化结果。")
    confidence: float = Field(default=1.0, ge=0, le=1, description="事实置信度，0 到 1。")
    certainty: Certainty = Field(default="confirmed", description="confirmed 表示明确事实；uncertain 表示推测或转述。")
    sensitivity_level: SensitivityLevel = Field(default="internal", description="敏感等级；pii 默认不得进入 compact_context。")
    source_type: FactSourceType = Field(..., description="事实来源类型。")
    source_conversation_id: str | None = Field(default=None, description="来源会话 ID。")
    source_message_id: str | None = Field(default=None, description="来源消息 ID。")
    evidence_text: str = Field(..., description="支持该事实的明确证据摘录；无证据不得写入长期事实表。")
    extraction_run_id: str | None = Field(default=None, description="产生该事实的分析运行 ID。")
    is_current: bool = Field(default=True, description="是否为当前有效事实。冲突旧事实会置为 False。")
    valid_from: str = Field(default_factory=utc_now_iso, description="事实生效时间。")
    valid_to: str | None = Field(default=None, description="事实失效时间；当前事实为空。")
    created_at: str = Field(default_factory=utc_now_iso, description="事实创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="事实最近更新时间。")


class OpportunityCase(BaseModel):
    """一个客户或渠道机会的长期推进状态。"""

    id: str = Field(default_factory=lambda: new_id("case"), description="机会 case 唯一 ID。")
    tenant_id: str = Field(..., description="case 所属租户 ID。")
    advisor_id: str = Field(..., description="负责该机会的从业者 ID。")
    customer_id: str = Field(..., description="该机会对应的客户或渠道对象 ID。")
    subject_type: SubjectType = Field(default="unclear", description="分析对象类型：客户、渠道或不明确。")
    case_status: str = Field(default="active", description="case 状态，例如 active、closed、nurture。")
    target_persona: TargetPersona = Field(default="unknown", description="内部客群标签，严禁直接对客户展示。")
    trigger_module: TriggerModule = Field(default="unknown", description="当前最适合的销售切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="当前建议阶段。")
    relationship_strength: str = Field(default="", description="关系强度摘要，例如能微信聊、能约饭。")
    latest_kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="最近一次 KYC 完整度分。")
    latest_opportunity_score: int = Field(default=0, ge=0, le=100, description="最近一次机会推进分。")
    latest_external_grade: ExternalGrade = Field(default="D", description="对从业者展示的推进等级。")
    latest_missing_fields: list[str] = Field(default_factory=list, description="最近一次仍缺失的关键字段。")
    latest_support_note: str = Field(default="", description="最近一次给从业者的鼓励摘要。")
    next_best_action: str = Field(default="", description="下一最佳动作，例如补问、低压维护、生成策略。")
    workflow_version: str = Field(default="local-v1", description="产生该 case 状态的工作流版本。")
    created_at: str = Field(default_factory=utc_now_iso, description="case 创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="case 最近更新时间。")


class Conversation(BaseModel):
    """一次围绕某个机会的多轮对话。"""

    id: str = Field(default_factory=lambda: new_id("conversation"), description="会话唯一 ID。")
    tenant_id: str = Field(..., description="会话所属租户 ID。")
    advisor_id: str = Field(..., description="会话对应从业者 ID。")
    customer_id: str | None = Field(default=None, description="会话对应客户 ID；未知时可为空。")
    opportunity_case_id: str | None = Field(default=None, description="会话关联的机会 case ID。")
    created_at: str = Field(default_factory=utc_now_iso, description="会话创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="会话最近更新时间。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="会话扩展信息。")


class ConversationMessage(BaseModel):
    """会话消息证据，保存完整多轮交互。"""

    id: str = Field(default_factory=lambda: new_id("message"), description="消息唯一 ID。")
    tenant_id: str = Field(..., description="消息所属租户 ID。")
    conversation_id: str = Field(..., description="消息所属会话 ID。")
    seq_no: int = Field(..., ge=1, description="消息在会话内的顺序号。")
    speaker_role: Literal["user", "assistant", "tool", "system"] = Field(..., description="消息说话方角色。")
    content: str = Field(..., description="原始消息内容，仅作为证据归档，不默认进入 Prompt。")
    content_redacted: str = Field(default="", description="脱敏后的消息内容，可用于抽取或评测。")
    message_ts: str = Field(default_factory=utc_now_iso, description="消息发生时间。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="消息扩展信息。")


class AgentSessionState(BaseModel):
    """每一轮 KYC 分析后的工作记忆快照。"""

    id: str = Field(default_factory=lambda: new_id("session_state"), description="会话状态快照唯一 ID。")
    tenant_id: str = Field(..., description="快照所属租户 ID。")
    conversation_id: str = Field(..., description="快照所属会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="快照关联的机会 case ID。")
    profile_state: dict[str, Any] = Field(default_factory=dict, description="客户 KYC 画像快照。")
    practitioner_state: dict[str, Any] = Field(default_factory=dict, description="从业者画像快照。")
    information_status: InformationStatus = Field(default="insufficient", description="当前信息状态。")
    subject_type: SubjectType = Field(default="unclear", description="当前分析对象类型。")
    target_persona: TargetPersona = Field(default="unknown", description="内部客群标签。")
    advisor_stage: AdvisorStage = Field(default="unknown", description="从业者阶段。")
    trigger_module: TriggerModule = Field(default="unknown", description="当前切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="当前建议阶段。")
    missing_fields: list[str] = Field(default_factory=list, description="当前缺失字段。")
    asked_focuses: list[str] = Field(default_factory=list, description="已经问过的 KYC 焦点。")
    kyc_question_round_count: int = Field(default=0, ge=0, description="KYC 补问轮次。")
    kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="KYC 完整度分。")
    opportunity_score: int = Field(default=0, ge=0, le=100, description="机会推进分。")
    external_grade: ExternalGrade = Field(default="D", description="对从业者展示的推进等级。")
    objective_material_need: str = Field(default="", description="需要补充的公开素材方向。")
    support_note: str = Field(default="", description="给从业者的鼓励摘要。")
    created_at: str = Field(default_factory=utc_now_iso, description="快照创建时间。")


class KYCQuestion(BaseModel):
    """一条已提出或待回答的 KYC 补问记录。"""

    id: str = Field(default_factory=lambda: new_id("kyc_question"), description="KYC 问题唯一 ID。")
    tenant_id: str = Field(..., description="问题所属租户 ID。")
    opportunity_case_id: str = Field(..., description="问题关联的机会 case ID。")
    conversation_id: str = Field(..., description="问题所属会话 ID。")
    round_no: int = Field(..., ge=1, description="问题所属 KYC 轮次。")
    focus_key: str = Field(..., description="问题焦点，例如 available_long_term_funds。")
    question_text: str = Field(..., description="实际向从业者提出的问题文本。")
    question_status: Literal["asked", "answered", "skipped"] = Field(default="asked", description="问题状态。")
    answer_message_id: str | None = Field(default=None, description="回答该问题的消息 ID。")
    extracted_fact_ids: list[str] = Field(default_factory=list, description="从回答中抽取出的事实 ID。")
    asked_at: str = Field(default_factory=utc_now_iso, description="问题提出时间。")
    answered_at: str | None = Field(default=None, description="问题回答时间。")


class DifyKYCAnalysisOutput(BaseModel):
    """Dify KYC 分析节点的 18 个顶层字段工程化 schema。"""

    information_status: InformationStatus = Field(..., description="当前 KYC 与推进状态。")
    subject_type: SubjectType = Field(..., description="当前分析对象类型。")
    target_persona: TargetPersona = Field(..., description="内部客群标签。")
    profile_state: dict[str, Any] = Field(default_factory=dict, description="客户 KYC 画像。")
    practitioner_state: dict[str, Any] = Field(default_factory=dict, description="从业者画像。")
    advisor_stage: AdvisorStage = Field(..., description="从业者阶段。")
    missing_fields: list[str] = Field(default_factory=list, description="当前缺失字段。")
    match_evidence: str = Field(default="", description="明确事实证据，不得写推测。")
    route_reason: str = Field(default="", description="进入当前信息状态的原因。")
    kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="KYC 完整度分。")
    opportunity_score: int = Field(default=0, ge=0, le=100, description="机会推进分。")
    external_grade: ExternalGrade = Field(default="D", description="对从业者展示的推进等级。")
    trigger_module: TriggerModule = Field(default="unknown", description="当前切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="当前建议阶段。")
    objective_material_need: str = Field(default="", description="需要补充的公开素材方向。")
    support_note: str = Field(default="", description="给从业者的鼓励摘要。")
    kyc_question_round_count: int = Field(default=0, ge=0, description="KYC 补问轮次。")
    asked_focuses: list[str] = Field(default_factory=list, description="已经问过的 KYC 焦点。")


class AnalysisRun(BaseModel):
    """每一次 KYC 分析和评分运行的审计记录。"""

    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(default_factory=lambda: new_id("analysis"), description="分析运行唯一 ID。")
    tenant_id: str = Field(..., description="分析运行所属租户 ID。")
    conversation_id: str = Field(..., description="分析运行所属会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="分析关联的机会 case ID。")
    model_name: str = Field(default="configured-runtime", description="执行分析的模型名称。")
    workflow_version: str = Field(default="local-v1", description="工作流版本。")
    prompt_version: str = Field(default="kyc-analyzer-v1", description="分析 Prompt 或规则版本。")
    input_snapshot: dict[str, Any] = Field(default_factory=dict, description="分析输入快照。")
    output_json: dict[str, Any] = Field(default_factory=dict, description="分析输出 JSON。")
    information_status: InformationStatus = Field(default="insufficient", description="本次分析得到的信息状态。")
    target_persona: TargetPersona = Field(default="unknown", description="本次分析识别出的内部客群标签。")
    trigger_module: TriggerModule = Field(default="unknown", description="本次分析识别出的切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="本次分析建议的当前沟通阶段。")
    kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="本次分析给出的 KYC 完整度分。")
    opportunity_score: int = Field(default=0, ge=0, le=100, description="本次分析给出的机会推进分。")
    external_grade: ExternalGrade = Field(default="D", description="本次分析对从业者展示的推进等级。")
    match_evidence: str = Field(default="", description="本次分析使用的明确事实证据，不得写推测。")
    route_reason: str = Field(default="", description="本次分析进入 matched/insufficient/unmatched 的原因。")
    latency_ms: int = Field(default=0, ge=0, description="分析耗时，单位毫秒。")
    created_at: str = Field(default_factory=utc_now_iso, description="分析运行创建时间。")


class GeneratedOutput(BaseModel):
    """每次生成给从业者的话术、策略、补问或维护消息。"""

    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(default_factory=lambda: new_id("generated_output"), description="生成输出唯一 ID。")
    tenant_id: str = Field(..., description="生成输出所属租户 ID。")
    conversation_id: str = Field(..., description="生成输出所属会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="生成输出关联的机会 case ID。")
    output_type: Literal["kyc_question", "strategy", "low_pressure_nurture", "compliance_rewrite"] = Field(
        ...,
        description="输出类型：补问、策略、低压维护或合规改写。",
    )
    model_name: str = Field(default="configured-runtime", description="生成该输出的模型名称或运行时生成器标识。")
    workflow_version: str = Field(default="local-v1", description="生成输出使用的工作流版本。")
    prompt_version: str = Field(default="strategy-generator-v1", description="生成输出使用的 Prompt 或规则版本。")
    input_context: dict[str, Any] = Field(
        default_factory=dict,
        description="进入生成节点的 compact_context 或等价上下文，不应包含 PII 或原始客户对话全文。",
    )
    output_text: str = Field(..., description="最终返回给从业者的生成文本。")
    safety_flags: list[str] = Field(default_factory=list, description="生成输出触发的安全或合规标记。")
    used_news_ids: list[str] = Field(default_factory=list, description="生成时使用的外部新闻素材 ID。")
    used_case_pattern_ids: list[str] = Field(default_factory=list, description="生成时使用的已审核销售模式 ID。")
    created_at: str = Field(default_factory=utc_now_iso, description="生成输出创建时间。")


class MemoryEvent(BaseModel):
    """围绕客户机会发生的事件记忆。"""

    id: str = Field(default_factory=lambda: new_id("memory_event"), description="事件记忆唯一 ID。")
    tenant_id: str = Field(..., description="事件所属租户 ID。")
    conversation_id: str | None = Field(default=None, description="事件来源会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="事件关联的机会 case ID。")
    customer_id: str | None = Field(default=None, description="事件关联客户 ID。")
    advisor_id: str | None = Field(default=None, description="事件关联从业者 ID。")
    event_type: MemoryEventType = Field(..., description="事件类型，例如异议、正向信号、关系变化、结果更新。")
    event_payload: dict[str, Any] = Field(default_factory=dict, description="事件结构化内容，敏感值应先脱敏或只保存引用。")
    evidence_text: str = Field(default="", description="支持该事件的证据摘录；没有证据时只可作为低可信运行事件。")
    source_message_id: str | None = Field(default=None, description="事件来源消息 ID。")
    created_at: str = Field(default_factory=utc_now_iso, description="事件记录创建时间。")


class CaseOutcome(BaseModel):
    """机会推进结果，用于把生成策略和真实业务结果闭环。"""

    id: str = Field(default_factory=lambda: new_id("case_outcome"), description="结果记录唯一 ID。")
    tenant_id: str = Field(..., description="结果所属租户 ID。")
    opportunity_case_id: str = Field(..., description="结果关联的机会 case ID。")
    outcome_type: CaseOutcomeType = Field(..., description="结果类型，例如已回复、已约电话、成交、长期维护。")
    outcome_detail: str = Field(default="", description="结果补充说明，避免只保存粗粒度标签。")
    source_conversation_id: str | None = Field(default=None, description="结果来源会话 ID。")
    source_message_id: str | None = Field(default=None, description="结果来源消息 ID。")
    created_at: str = Field(default_factory=utc_now_iso, description="结果记录创建时间。")
