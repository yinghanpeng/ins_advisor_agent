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


# Certainty 区分已经由证据确认的事实与仍需核实的线索，避免生成阶段混用。
Certainty = Literal["confirmed", "uncertain"]
# SensitivityLevel 决定字段是否可进入通用上下文；pii 默认禁止进入 Prompt。
SensitivityLevel = Literal["public", "internal", "sensitive", "pii"]
# FactSourceType 保留事实产生渠道，使每条长期记忆都能追溯到来源类别。
FactSourceType = Literal["user_message", "advisor_message", "analysis", "manual_review", "imported_corpus"]
# InformationStatus 描述当前 KYC 是否足够进入匹配策略或明确不匹配。
InformationStatus = Literal["insufficient", "matched", "unmatched"]
# SubjectType 标记当前被分析对象是终端客户、渠道伙伴还是暂时无法判断。
SubjectType = Literal["customer", "channel", "unclear"]
# TargetPersona 是仅供内部路由使用的粗粒度客群标签，不应直接展示给客户。
TargetPersona = Literal["enterprise_owner", "executive", "family_planner", "channel", "unknown"]
# AdvisorStage 表示顾问成熟度，用于调整教练建议的详细程度和语气。
AdvisorStage = Literal["newbie", "transitioning", "part_time", "experienced", "unknown"]
# ExternalGrade 是面向顾问展示的机会等级，避免暴露内部连续评分细节。
ExternalGrade = Literal["A", "B", "C", "D"]
# TriggerModule 限定策略生成器可选择的业务切入模块，unknown 表示暂不判断。
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
# CurrentStage 描述当前建议采取的沟通阶段，控制补问、深聊或低压收尾分支。
CurrentStage = Literal["collect_kyc", "deep_conversation", "cultivate", "low_pressure_end"]
# MemoryEventType 为业务时间线限定事件种类，便于后续检索和离线评测。
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
# CaseOutcomeType 统一记录沟通后的真实结果，用于形成策略到结果的闭环。
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

    # 内部记录 ID 使用 tenant 前缀，便于日志中识别实体类型。
    id: str = Field(default_factory=lambda: new_id("tenant"), description="租户唯一 ID。")
    # tenant_id 是所有业务查询的强制隔离键，不允许由下游省略。
    tenant_id: str = Field(..., description="业务租户 ID，所有业务表必须携带该字段。")
    # display_name 仅用于受控界面展示，不参与授权或数据选择。
    display_name: str = Field(default="", description="租户展示名称，例如分公司、团队或机构名称。")
    # 创建与更新时间均使用 UTC ISO 字符串，保证跨时区审计一致。
    created_at: str = Field(default_factory=utc_now_iso, description="租户记录创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="租户记录最近更新时间。")
    # metadata 承载非核心扩展属性，核心授权字段不能放在这里。
    metadata: dict[str, Any] = Field(default_factory=dict, description="租户扩展信息。")


class Advisor(BaseModel):
    """使用本系统的顾问/从业者。"""

    # 顾问记录使用带实体前缀的全局唯一 ID。
    id: str = Field(default_factory=lambda: new_id("advisor"), description="顾问唯一 ID。")
    # 顾问始终归属于单一租户，所有读取必须同时匹配该边界。
    tenant_id: str = Field(..., description="顾问所属租户 ID。")
    # display_name 可保存内部展示名，但生成上下文优先使用脱敏别名。
    display_name: str = Field(default="", description="顾问展示名称，可为空。")
    display_alias: str = Field(default="", description="顾问对外别名，避免在 Prompt 中暴露真实姓名。")
    # 时间戳用于记录顾问实体的创建和最近一次属性更新。
    created_at: str = Field(default_factory=utc_now_iso, description="顾问记录创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="顾问记录最近更新时间。")
    # metadata 仅保存不影响租户隔离的附加标签。
    metadata: dict[str, Any] = Field(default_factory=dict, description="顾问扩展信息。")


class Customer(BaseModel):
    """被分析的客户或渠道对象。

    真实姓名、手机号、微信号、身份证、精确地址等 PII 不直接进入生成上下文。
    生产环境应保存 pii_ref_id 或 hash，并在权限允许时单独解析。
    """

    # 客户主键使用不可推断的唯一 ID，不直接采用手机号等 PII。
    id: str = Field(default_factory=lambda: new_id("customer"), description="客户唯一 ID。")
    # tenant_id 将客户限定在所属租户的数据域内。
    tenant_id: str = Field(..., description="客户所属租户 ID。")
    # 展示别名是生成上下文允许使用的低敏客户称谓。
    display_alias: str = Field(default="", description="客户展示别名，例如 客户A、制造业企业主。")
    # pii_ref_id 指向隔离的 PII Vault，而不是在本表存储真实身份内容。
    pii_ref_id: str | None = Field(default=None, description="PII 引用 ID，默认不进入 Prompt。")
    # 联系方式仅保存不可逆 hash 以支持去重，不可用于还原原始值。
    phone_hash: str | None = Field(default=None, description="手机号 hash，用于去重，不进入 Prompt。")
    email_hash: str | None = Field(default=None, description="邮箱 hash，用于去重，不进入 Prompt。")
    # 创建和更新时间支撑隐私导出、删除及审计时间线。
    created_at: str = Field(default_factory=utc_now_iso, description="客户记录创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="客户记录最近更新时间。")
    # metadata 只用于受控扩展，禁止存入未加密 PII。
    metadata: dict[str, Any] = Field(default_factory=dict, description="客户扩展信息。")


class AdvisorProfileFact(BaseModel):
    """从业者画像事实，用于替代单一 practitioner_state 字符串。"""

    # 每个事实版本拥有独立 ID，事实修正时关闭旧版本而非复用主键。
    id: str = Field(default_factory=lambda: new_id("advisor_fact"), description="从业者事实唯一 ID。")
    # tenant_id 与 advisor_id 共同限定事实所属主体。
    tenant_id: str = Field(..., description="事实所属租户 ID。")
    advisor_id: str = Field(..., description="事实对应的从业者 ID。")
    # fact_key 使用稳定业务键，支持同一维度的版本冲突检测。
    fact_key: str = Field(..., description="事实键，例如 role、experience_years、confidence_barrier。")
    # fact_value 允许结构化值，但上层必须控制内容和敏感级别。
    fact_value: Any = Field(..., description="事实值。可以是字符串、数字、列表或结构化对象。")
    # confidence 被限制在 0 到 1，表示抽取结果的证据强度。
    confidence: float = Field(default=1.0, ge=0, le=1, description="事实置信度，0 到 1。")
    # source_type 与来源会话/消息共同构成事实溯源信息。
    source_type: FactSourceType = Field(..., description="事实来源类型。")
    source_conversation_id: str | None = Field(default=None, description="来源会话 ID。")
    source_message_id: str | None = Field(default=None, description="来源消息 ID。")
    # evidence_text 保存支持事实的最小证据片段，写入策略要求非空。
    evidence_text: str = Field(..., description="支持该事实的原文证据摘录，不得为空。")
    # is_current 和有效期字段共同表达事实版本，不物理删除被修正内容。
    is_current: bool = Field(default=True, description="是否为当前有效事实。冲突旧事实会置为 False。")
    valid_from: str = Field(default_factory=utc_now_iso, description="事实生效时间。")
    valid_to: str | None = Field(default=None, description="事实失效时间；当前事实为空。")
    # created_at 保留初次入库时间，updated_at 记录证据增强或关闭时间。
    created_at: str = Field(default_factory=utc_now_iso, description="事实创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="事实最近更新时间。")


class CustomerProfileFact(BaseModel):
    """客户画像事实，用于替代单一 profile_state 字符串。"""

    # 客户事实采用版本化唯一 ID，便于完整追踪 KYC 变化历史。
    id: str = Field(default_factory=lambda: new_id("customer_fact"), description="客户事实唯一 ID。")
    # tenant_id 与 customer_id 组成强制主体边界。
    tenant_id: str = Field(..., description="事实所属租户 ID。")
    customer_id: str = Field(..., description="事实对应的客户 ID。")
    # fact_key 是稳定的 KYC 维度名，用于冲突检测和 compact context 构建。
    fact_key: str = Field(..., description="事实键，例如 age、industry、children、financial_preference。")
    # fact_value 保存抽取原值；normalized_value 保存便于比较和路由的标准值。
    fact_value: Any = Field(..., description="原始事实值。")
    normalized_value: Any | None = Field(default=None, description="标准化事实值，例如年龄段或标签化结果。")
    # confidence 和 certainty 分别表达模型把握度与事实是否已由用户确认。
    confidence: float = Field(default=1.0, ge=0, le=1, description="事实置信度，0 到 1。")
    certainty: Certainty = Field(default="confirmed", description="confirmed 表示明确事实；uncertain 表示推测或转述。")
    # sensitivity_level 控制该事实能否进入 Prompt，pii 在写入策略层直接阻断。
    sensitivity_level: SensitivityLevel = Field(default="internal", description="敏感等级；pii 默认不得进入 compact_context。")
    # 以下来源字段将每个事实关联回具体消息和抽取运行。
    source_type: FactSourceType = Field(..., description="事实来源类型。")
    source_conversation_id: str | None = Field(default=None, description="来源会话 ID。")
    source_message_id: str | None = Field(default=None, description="来源消息 ID。")
    evidence_text: str = Field(..., description="支持该事实的明确证据摘录；无证据不得写入长期事实表。")
    extraction_run_id: str | None = Field(default=None, description="产生该事实的分析运行 ID。")
    # 当前标记和有效期用于保留冲突事实的历史版本。
    is_current: bool = Field(default=True, description="是否为当前有效事实。冲突旧事实会置为 False。")
    valid_from: str = Field(default_factory=utc_now_iso, description="事实生效时间。")
    valid_to: str | None = Field(default=None, description="事实失效时间；当前事实为空。")
    # 创建/更新时间为事实生命周期审计提供统一 UTC 时间。
    created_at: str = Field(default_factory=utc_now_iso, description="事实创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="事实最近更新时间。")


class OpportunityCase(BaseModel):
    """一个客户或渠道机会的长期推进状态。"""

    # Case ID 标识一段独立销售任务；意图变化创建新 Case 而不是复用旧轮次。
    id: str = Field(default_factory=lambda: new_id("case"), description="机会 case 唯一 ID。")
    # tenant/advisor/customer 三个字段限定 Case 的归属和可见范围。
    tenant_id: str = Field(..., description="case 所属租户 ID。")
    advisor_id: str = Field(..., description="负责该机会的从业者 ID。")
    customer_id: str = Field(..., description="该机会对应的客户或渠道对象 ID。")
    # subject_type 与 case_status 描述分析对象和 Case 生命周期。
    subject_type: SubjectType = Field(default="unclear", description="分析对象类型：客户、渠道或不明确。")
    case_status: str = Field(default="active", description="case 状态，例如 active、closed、nurture。")
    # persona、trigger 和 stage 是策略路由字段，均使用受限枚举避免任意字符串漂移。
    target_persona: TargetPersona = Field(default="unknown", description="内部客群标签，严禁直接对客户展示。")
    trigger_module: TriggerModule = Field(default="unknown", description="当前最适合的销售切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="当前建议阶段。")
    # 关系摘要用于策略语气，不应包含真实姓名或联系方式。
    relationship_strength: str = Field(default="", description="关系强度摘要，例如能微信聊、能约饭。")
    # 两个连续评分限定为 0-100，外部只展示离散等级。
    latest_kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="最近一次 KYC 完整度分。")
    latest_opportunity_score: int = Field(default=0, ge=0, le=100, description="最近一次机会推进分。")
    latest_external_grade: ExternalGrade = Field(default="D", description="对从业者展示的推进等级。")
    # 缺失字段、支持说明与下一动作构成下一轮恢复所需的最小任务状态。
    latest_missing_fields: list[str] = Field(default_factory=list, description="最近一次仍缺失的关键字段。")
    latest_support_note: str = Field(default="", description="最近一次给从业者的鼓励摘要。")
    next_best_action: str = Field(default="", description="下一最佳动作，例如补问、低压维护、生成策略。")
    # workflow_version 标明 Case 状态由哪一版本代码产生，便于升级回溯。
    workflow_version: str = Field(default="local-v1", description="产生该 case 状态的工作流版本。")
    # 时间字段记录 Case 创建和最近一次聚合状态更新。
    created_at: str = Field(default_factory=utc_now_iso, description="case 创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="case 最近更新时间。")


class Conversation(BaseModel):
    """一次围绕某个机会的多轮对话。"""

    # 会话 ID 将多条消息、Session Snapshot 和分析运行关联到同一轮对话。
    id: str = Field(default_factory=lambda: new_id("conversation"), description="会话唯一 ID。")
    # tenant 与 advisor 必填，customer/Case 在尚未识别时允许为空。
    tenant_id: str = Field(..., description="会话所属租户 ID。")
    advisor_id: str = Field(..., description="会话对应从业者 ID。")
    customer_id: str | None = Field(default=None, description="会话对应客户 ID；未知时可为空。")
    opportunity_case_id: str | None = Field(default=None, description="会话关联的机会 case ID。")
    # 时间戳支撑会话排序和生命周期审计。
    created_at: str = Field(default_factory=utc_now_iso, description="会话创建时间。")
    updated_at: str = Field(default_factory=utc_now_iso, description="会话最近更新时间。")
    # metadata 保存不进入核心关系模型的低敏扩展标签。
    metadata: dict[str, Any] = Field(default_factory=dict, description="会话扩展信息。")


class AgentSessionState(BaseModel):
    """每一轮 KYC 分析后的工作记忆快照。"""

    # 快照 ID 唯一标识一次轮次状态，后续写入追加新快照而不覆盖旧版本。
    id: str = Field(default_factory=lambda: new_id("session_state"), description="会话状态快照唯一 ID。")
    # tenant 与 conversation 是读取快照时必须同时匹配的隔离键。
    tenant_id: str = Field(..., description="快照所属租户 ID。")
    conversation_id: str = Field(..., description="快照所属会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="快照关联的机会 case ID。")
    # profile_state 与 practitioner_state 是本轮结构化画像，不应包含原始 PII。
    profile_state: dict[str, Any] = Field(default_factory=dict, description="客户 KYC 画像快照。")
    practitioner_state: dict[str, Any] = Field(default_factory=dict, description="从业者画像快照。")
    # 以下枚举字段保存本轮路由判断，支持下一轮无需重复分析即可恢复任务。
    information_status: InformationStatus = Field(default="insufficient", description="当前信息状态。")
    subject_type: SubjectType = Field(default="unclear", description="当前分析对象类型。")
    target_persona: TargetPersona = Field(default="unknown", description="内部客群标签。")
    advisor_stage: AdvisorStage = Field(default="unknown", description="从业者阶段。")
    trigger_module: TriggerModule = Field(default="unknown", description="当前切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="当前建议阶段。")
    # missing_fields 与 asked_focuses 共同决定下一轮应该补问哪个尚未覆盖的焦点。
    missing_fields: list[str] = Field(default_factory=list, description="当前缺失字段。")
    asked_focuses: list[str] = Field(default_factory=list, description="已经问过的 KYC 焦点。")
    # 补问轮次单调递增且不允许负数，用于限制持续追问。
    kyc_question_round_count: int = Field(default=0, ge=0, description="KYC 补问轮次。")
    # 完整度与机会分限制在 0-100，external_grade 是其对外离散表达。
    kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="KYC 完整度分。")
    opportunity_score: int = Field(default=0, ge=0, le=100, description="机会推进分。")
    external_grade: ExternalGrade = Field(default="D", description="对从业者展示的推进等级。")
    # 素材需求和支持摘要供后续策略生成，不应包含未经验证的事实。
    objective_material_need: str = Field(default="", description="需要补充的公开素材方向。")
    support_note: str = Field(default="", description="给从业者的鼓励摘要。")
    # created_at 固化该快照产生的 UTC 时间，便于选取最新状态。
    created_at: str = Field(default_factory=utc_now_iso, description="快照创建时间。")


class KYCQuestion(BaseModel):
    """一条已提出或待回答的 KYC 补问记录。"""

    # 问题 ID 用于将回答和抽取事实回链到实际展示过的补问。
    id: str = Field(default_factory=lambda: new_id("kyc_question"), description="KYC 问题唯一 ID。")
    # tenant、Case 和 conversation 三层关联保证问题不会跨任务复用。
    tenant_id: str = Field(..., description="问题所属租户 ID。")
    opportunity_case_id: str = Field(..., description="问题关联的机会 case ID。")
    conversation_id: str = Field(..., description="问题所属会话 ID。")
    # round_no 从 1 开始记录补问次序，focus_key 是防重复的业务维度键。
    round_no: int = Field(..., ge=1, description="问题所属 KYC 轮次。")
    focus_key: str = Field(..., description="问题焦点，例如 available_long_term_funds。")
    # question_text 只在问题已经实际呈现给用户后持久化。
    question_text: str = Field(..., description="实际向从业者提出的问题文本。")
    # question_status 表达问题从已问到已回答或跳过的生命周期。
    question_status: Literal["asked", "answered", "skipped"] = Field(default="asked", description="问题状态。")
    # extracted_fact_ids 将回答产生的长期事实与问题证据链连接起来。
    extracted_fact_ids: list[str] = Field(default_factory=list, description="从回答中抽取出的事实 ID。")
    # asked_at 必填生成，answered_at 仅在收到有效回答时设置。
    asked_at: str = Field(default_factory=utc_now_iso, description="问题提出时间。")
    answered_at: str | None = Field(default=None, description="问题回答时间。")


class DifyKYCAnalysisOutput(BaseModel):
    """Dify KYC 分析节点的 18 个顶层字段工程化 schema。"""

    # 前三个字段确定整体信息状态、分析对象和内部客群路由。
    information_status: InformationStatus = Field(..., description="当前 KYC 与推进状态。")
    subject_type: SubjectType = Field(..., description="当前分析对象类型。")
    target_persona: TargetPersona = Field(..., description="内部客群标签。")
    # profile/practitioner 保存本轮合并后的客户与顾问结构化画像。
    profile_state: dict[str, Any] = Field(default_factory=dict, description="客户 KYC 画像。")
    practitioner_state: dict[str, Any] = Field(default_factory=dict, description="从业者画像。")
    # advisor_stage 用于调整教练建议，missing_fields 驱动后续 KYC 补问。
    advisor_stage: AdvisorStage = Field(..., description="从业者阶段。")
    missing_fields: list[str] = Field(default_factory=list, description="当前缺失字段。")
    # match_evidence 必须来自明确输入，route_reason 解释为何选择当前分支。
    match_evidence: str = Field(default="", description="明确事实证据，不得写推测。")
    route_reason: str = Field(default="", description="进入当前信息状态的原因。")
    # 三个评分字段分离内部连续判断与对顾问展示的离散等级。
    kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="KYC 完整度分。")
    opportunity_score: int = Field(default=0, ge=0, le=100, description="机会推进分。")
    external_grade: ExternalGrade = Field(default="D", description="对从业者展示的推进等级。")
    # trigger_module 与 current_stage 决定后续知识检索和生成路径。
    trigger_module: TriggerModule = Field(default="unknown", description="当前切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="当前建议阶段。")
    # 素材与支持文本是辅助说明，不参与事实置信度计算。
    objective_material_need: str = Field(default="", description="需要补充的公开素材方向。")
    support_note: str = Field(default="", description="给从业者的鼓励摘要。")
    # 补问轮次和已问焦点用于从历史工作流迁移时保持多轮状态。
    kyc_question_round_count: int = Field(default=0, ge=0, description="KYC 补问轮次。")
    asked_focuses: list[str] = Field(default_factory=list, description="已经问过的 KYC 焦点。")


class AnalysisRun(BaseModel):
    """每一次 KYC 分析和评分运行的审计记录。"""

    # 允许使用 model_name 字段，避免 Pydantic 的 protected namespace 与业务字段冲突。
    model_config = ConfigDict(protected_namespaces=())

    # 每次分析运行都生成独立 ID，并关联租户、会话及可选 Case。
    id: str = Field(default_factory=lambda: new_id("analysis"), description="分析运行唯一 ID。")
    tenant_id: str = Field(..., description="分析运行所属租户 ID。")
    conversation_id: str = Field(..., description="分析运行所属会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="分析关联的机会 case ID。")
    # 模型、工作流和 Prompt 版本共同保证一次结果能够被准确复现。
    model_name: str = Field(default="configured-runtime", description="执行分析的模型名称。")
    workflow_version: str = Field(default="local-v1", description="工作流版本。")
    prompt_version: str = Field(default="kyc-analyzer-v1", description="分析 Prompt 或规则版本。")
    # 输入和输出快照仅保存脱敏结构化数据，用于离线回放和质量分析。
    input_snapshot: dict[str, Any] = Field(default_factory=dict, description="分析输入快照。")
    output_json: dict[str, Any] = Field(default_factory=dict, description="分析输出 JSON。")
    # 路由枚举固化本次分析结论，避免后续代码重新解释自由文本输出。
    information_status: InformationStatus = Field(default="insufficient", description="本次分析得到的信息状态。")
    target_persona: TargetPersona = Field(default="unknown", description="本次分析识别出的内部客群标签。")
    trigger_module: TriggerModule = Field(default="unknown", description="本次分析识别出的切入模块。")
    current_stage: CurrentStage = Field(default="collect_kyc", description="本次分析建议的当前沟通阶段。")
    # 分数和等级保存本轮量化结果，便于比较策略变化。
    kyc_completeness_score: int = Field(default=0, ge=0, le=100, description="本次分析给出的 KYC 完整度分。")
    opportunity_score: int = Field(default=0, ge=0, le=100, description="本次分析给出的机会推进分。")
    external_grade: ExternalGrade = Field(default="D", description="本次分析对从业者展示的推进等级。")
    # 证据和路由原因解释模型为何得出当前结论，写入策略要求证据非空。
    match_evidence: str = Field(default="", description="本次分析使用的明确事实证据，不得写推测。")
    route_reason: str = Field(default="", description="本次分析进入 matched/insufficient/unmatched 的原因。")
    # latency_ms 记录非负耗时，created_at 固化运行完成时间。
    latency_ms: int = Field(default=0, ge=0, description="分析耗时，单位毫秒。")
    created_at: str = Field(default_factory=utc_now_iso, description="分析运行创建时间。")


class GeneratedOutput(BaseModel):
    """每次生成给从业者的话术、策略、补问或维护消息。"""

    # 关闭 protected namespace 检查，以合法使用业务字段 model_name。
    model_config = ConfigDict(protected_namespaces=())

    # 输出记录关联租户、会话及可选 Case，支持从生成结果追溯到业务任务。
    id: str = Field(default_factory=lambda: new_id("generated_output"), description="生成输出唯一 ID。")
    tenant_id: str = Field(..., description="生成输出所属租户 ID。")
    conversation_id: str = Field(..., description="生成输出所属会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="生成输出关联的机会 case ID。")
    # output_type 限制可保存的输出用途，避免任意类型绕过对应安全策略。
    output_type: Literal["kyc_question", "strategy", "low_pressure_nurture", "compliance_rewrite"] = Field(
        ...,
        description="输出类型：补问、策略、低压维护或合规改写。",
    )
    # 运行时、工作流与 Prompt 版本构成输出可复现所需的版本信息。
    model_name: str = Field(default="configured-runtime", description="生成该输出的模型名称或运行时生成器标识。")
    workflow_version: str = Field(default="local-v1", description="生成输出使用的工作流版本。")
    prompt_version: str = Field(default="strategy-generator-v1", description="生成输出使用的 Prompt 或规则版本。")
    # input_context 只能是 compact context 或等价脱敏结构，不能保存原始对话全文。
    input_context: dict[str, Any] = Field(
        default_factory=dict,
        description="进入生成节点的 compact_context 或等价上下文，不应包含 PII 或原始客户对话全文。",
    )
    # output_text 是最终呈现文本，写入前还会经过 PII 和合规校验。
    output_text: str = Field(..., description="最终返回给从业者的生成文本。")
    # safety_flags 保存本轮安全命中；两个 used_* 列表记录引用素材的溯源 ID。
    safety_flags: list[str] = Field(default_factory=list, description="生成输出触发的安全或合规标记。")
    used_news_ids: list[str] = Field(default_factory=list, description="生成时使用的外部新闻素材 ID。")
    used_case_pattern_ids: list[str] = Field(default_factory=list, description="生成时使用的已审核销售模式 ID。")
    # created_at 表示输出生成时间，支持与 Case Outcome 做时间关联。
    created_at: str = Field(default_factory=utc_now_iso, description="生成输出创建时间。")


class MemoryEvent(BaseModel):
    """围绕客户机会发生的事件记忆。"""

    # 每个事件拥有独立 ID，并强制关联租户以形成隔离的时间线。
    id: str = Field(default_factory=lambda: new_id("memory_event"), description="事件记忆唯一 ID。")
    tenant_id: str = Field(..., description="事件所属租户 ID。")
    # 以下四个业务 ID 均可选，但写入策略要求至少提供一个关联主体。
    conversation_id: str | None = Field(default=None, description="事件来源会话 ID。")
    opportunity_case_id: str | None = Field(default=None, description="事件关联的机会 case ID。")
    customer_id: str | None = Field(default=None, description="事件关联客户 ID。")
    advisor_id: str | None = Field(default=None, description="事件关联从业者 ID。")
    # event_type 使用受限枚举，event_payload 承载经过脱敏的结构化细节。
    event_type: MemoryEventType = Field(..., description="事件类型，例如异议、正向信号、关系变化、结果更新。")
    event_payload: dict[str, Any] = Field(default_factory=dict, description="事件结构化内容，敏感值应先脱敏或只保存引用。")
    # evidence_text 保存支持事件的最小证据，不依赖重复的业务消息归档表。
    evidence_text: str = Field(default="", description="支持该事件的证据摘录；没有证据时只可作为低可信运行事件。")
    # created_at 固定事件入库时间，列表读取可据此解释顺序。
    created_at: str = Field(default_factory=utc_now_iso, description="事件记录创建时间。")


class CaseOutcome(BaseModel):
    """机会推进结果，用于把生成策略和真实业务结果闭环。"""

    # Outcome 独立于 GeneratedOutput 保存，用于衡量策略产生的真实业务结果。
    id: str = Field(default_factory=lambda: new_id("case_outcome"), description="结果记录唯一 ID。")
    # tenant 和 opportunity_case_id 共同将结果限定到单一业务机会。
    tenant_id: str = Field(..., description="结果所属租户 ID。")
    opportunity_case_id: str = Field(..., description="结果关联的机会 case ID。")
    # outcome_type 使用有限枚举支持聚合分析，detail 保存必要补充说明。
    outcome_type: CaseOutcomeType = Field(..., description="结果类型，例如已回复、已约电话、成交、长期维护。")
    outcome_detail: str = Field(default="", description="结果补充说明，避免只保存粗粒度标签。")
    # 来源会话允许将结果回链到对应客户沟通任务。
    source_conversation_id: str | None = Field(default=None, description="结果来源会话 ID。")
    # created_at 用于计算从策略生成到业务结果发生的时延。
    created_at: str = Field(default_factory=utc_now_iso, description="结果记录创建时间。")
