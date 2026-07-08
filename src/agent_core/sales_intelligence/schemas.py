"""Sales Intelligence schemas."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_core.utils.ids import new_id
from agent_core.utils.time import utc_now_iso


# 业务阶段枚举：用于把访谈经验放到新客破冰、老客维护、加保、转介绍、计划书、成交等不同销售阶段。
BusinessStage = Literal[
    "new_customer",
    "old_customer_maintenance",
    "add_on",
    "referral",
    "proposal",
    "closing",
    "unknown",
]

# 销售洞察卡片的风险等级；high 内容默认不能直接进入最终生成。
RiskLevel = Literal["low", "medium", "high"]

# 销售对话模式类型；真实语料必须先抽成这些模式，不能直接作为普通 RAG chunk 进入生成。
PatternType = Literal[
    "opening",
    "kyc_question",
    "objection_handling",
    "transition",
    "close_next_step",
    "low_pressure_nurture",
    "risk_reframing",
    "product_bridge",
    "next_best_action",
]


class CustomerKYC(BaseModel):
    """客户 KYC 摘要，用于把访谈中的客户背景转成可检索、可复用的结构。"""

    # age 帮助检索器匹配人生阶段，例如成长期家庭、临近退休、企业接班等。
    age: str | None = Field(
        default=None,
        description="客户年龄或年龄段，例如 45岁左右、60后、刚成家。未知时为空。",
    )
    # family 反映家庭责任，是保险沟通中比产品收益更稳妥的切入点。
    family: str | None = Field(
        default=None,
        description="客户家庭结构和责任信息，例如已婚、有两个孩子、父母赡养压力。",
    )
    # occupation 帮助区分企业主、医生、高管等客户类型，影响沟通关注点。
    occupation: str | None = Field(
        default=None,
        description="客户职业或收入来源，例如制造业企业主、医生、互联网高管、全职太太。",
    )
    # asset_preference 保存客户既有资产偏好，用于异议处理和破冰策略匹配。
    asset_preference: str | None = Field(
        default=None,
        description="客户资产偏好或既有配置习惯，例如偏银行理财、重视现金流、抗拒长期锁定。",
    )
    # decision_style 帮助生成更合适的追问节奏，例如谨慎客户不适合强推。
    decision_style: str | None = Field(
        default=None,
        description="客户决策风格，例如谨慎型、听配偶意见、重视熟人案例、讨厌被推销。",
    )


class SalesInsightCard(BaseModel):
    """销售洞察卡片，是销售访谈资产化后的核心结构。"""

    # source_id 追溯原始访谈或素材，保证每条经验都能回到来源。
    source_id: str = Field(
        ...,
        description="原始访谈、转写稿或销售素材的来源 ID，用于审计和回溯。",
    )
    # chunk_id 定位原文片段，方便人工复核某条卡片是否抽取准确。
    chunk_id: str = Field(
        ...,
        description="该卡片对应的访谈片段 ID。一个 source_id 可拆分出多个 chunk。",
    )
    # interviewee_role 标识经验来源身份，用于评估洞察可信度。
    interviewee_role: str = Field(
        ...,
        description="受访者角色，例如资深保险顾问、团队长、银保渠道经理。",
    )
    # sales_experience_years 帮助区分新手建议和资深顾问沉淀经验。
    sales_experience_years: float | None = Field(
        default=None,
        description="受访者销售从业年限。用于区分经验来源成熟度，未知时为空。",
    )
    # channel 保存渠道环境，因为银行、转介绍、私董会的话术边界不同。
    channel: str | None = Field(
        default=None,
        description="销售场景或渠道，例如高净值转介绍、银行网点、私董会、老客户加保。",
    )
    # business_stage 用于检索时匹配业务阶段，避免把成交话术用在初次破冰。
    business_stage: BusinessStage = Field(
        default="unknown",
        description="业务阶段，例如新客户破冰、老客户维护、计划书、成交收口。",
    )
    # scene 是最具体的沟通场景，最终回答会围绕它组织建议。
    scene: str = Field(
        ...,
        description="具体沟通场景，例如饭局破冰、KYC 深挖、异议处理、计划书讲解。",
    )
    # customer_type 是检索主标签之一，例如企业主客户和医生客户的触发点不同。
    customer_type: str = Field(
        ...,
        description="客户类型标签，例如企业主、医生、家庭主妇、退休客户、高净值二代。",
    )
    # customer_kyc 保存可匹配的客户背景，支持更精准的销售洞察检索。
    customer_kyc: CustomerKYC = Field(
        default_factory=CustomerKYC,
        description="该经验卡片对应的客户 KYC 信息，用于检索匹配和话术个性化。",
    )
    # sales_pain_solved 描述这张卡片解决的销售难点，是 query rewrite 的重要输入。
    sales_pain_solved: str = Field(
        ...,
        description="这条经验解决的销售痛点，例如不会自然切入保险、客户抗拒谈保障。",
    )
    # root_cause 保存痛点根因，避免回答只给话术不解释为什么。
    root_cause: str = Field(
        ...,
        description="痛点背后的原因分析，例如客户防御感强、销售过早讲产品、信任未建立。",
    )
    # effective_strategy 保存可迁移策略，最终回答优先输出策略再输出话术。
    effective_strategy: str = Field(
        ...,
        description="被访谈验证有效的沟通策略，应该是可迁移的方法，不只是单句口号。",
    )
    # usable_script 保存可改写话术，但必须经过合规过滤后才能进入回答。
    usable_script: str = Field(
        ...,
        description="可直接改写给顾问使用的安全话术。不得包含保证收益、避税避债等违规表达。",
    )
    # wrong_way 保存反例，帮助新手知道哪些行为会提高客户防御。
    wrong_way: str = Field(
        ...,
        description="该场景下不建议采用的错误做法，用于提醒新手避坑。",
    )
    # why_it_works 解释策略有效机制，避免输出看起来像死记硬背的话术。
    why_it_works: str = Field(
        ...,
        description="解释该策略为什么有效，例如降低防御、先共情再引导、让客户先谈用途。",
    )
    # next_question 给出下一句低压追问，帮助沟通自然推进。
    next_question: str = Field(
        ...,
        description="建议销售下一句追问，用于把沟通自然推进到更深层 KYC 或下一步动作。",
    )
    # customer_response 保存真实客户反应，可作为可信案例线索但不能编造成承诺。
    customer_response: str | None = Field(
        default=None,
        description="访谈中记录的客户典型反应。没有明确记录时为空。",
    )
    # follow_up_action 保存下一步动作建议，最终 response_package 可转成 next_actions。
    follow_up_action: str | None = Field(
        default=None,
        description="建议的后续动作，例如准备资金分层图、约 15 分钟复盘、补充家庭责任信息。",
    )
    # tags 用于 hybrid search 的 metadata 过滤和 rerank。
    tags: list[str] = Field(
        default_factory=list,
        description="检索和评估标签，例如 破冰、企业主、资金分层、异议处理。",
    )
    # risk_level 控制这张卡片能否直接进入生成链路。
    risk_level: RiskLevel = Field(
        default="low",
        description="卡片风险等级。high 卡片默认不直接用于生成，应先合规审查或人工审批。",
    )
    # compliance_notes 保存人工或规则审查结论，供生成前再次提醒。
    compliance_notes: str = Field(
        default="",
        description="合规备注，记录需要避免的表达、引用限制或审查结论。",
    )
    # suitable_for_rag 表示这张卡片是否适合作为检索语料进入 RAG。
    suitable_for_rag: bool = Field(
        default=True,
        description="该卡片是否适合作为 RAG 检索资料。噪声过大或未结构化内容应设为 False。",
    )
    # suitable_for_eval 表示是否能从这张卡片生成评估样本。
    suitable_for_eval: bool = Field(
        default=True,
        description="该卡片是否适合作为评估样本来源。用于自动生成 sales quality eval case。",
    )
    # approved_for_generation 是最终生成准入开关，默认 False 防止未审访谈直接被模型引用。
    approved_for_generation: bool = Field(
        default=False,
        description="该卡片是否已通过审查、允许进入最终生成。默认 False，防止原始访谈直接出现在回答中。",
    )
    # created_at 记录卡片创建时间，方便版本审计。
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="卡片创建时间，ISO 字符串，用于版本追踪和审计。",
    )
    # updated_at 记录最近更新时间，后续人工修订时应更新。
    updated_at: str = Field(
        default_factory=utc_now_iso,
        description="卡片最近更新时间，ISO 字符串，用于后续人工修订或再审查。",
    )


class SalesInsightDigest(BaseModel):
    """检索到的销售洞察压缩摘要，供最终回答生成使用。"""

    # applicable_scene 告诉生成节点这些洞察适合哪个沟通场景。
    applicable_scene: str = Field(
        ...,
        description="这份摘要适用的沟通场景，例如饭局破冰、宏观共鸣、异议处理。",
    )
    # insight_summary 是多张卡片融合后的策略摘要，不直接堆原文。
    insight_summary: str = Field(
        ...,
        description="对多条销售经验卡片压缩后的核心洞察。应保留策略，不堆砌原文。",
    )
    # usable_scripts 是可以改写给用户的候选话术集合。
    usable_scripts: list[str] = Field(
        default_factory=list,
        description="可以安全改写给用户的候选话术列表。",
    )
    # forbidden_expressions 是生成阶段必须避开的表达清单。
    forbidden_expressions: list[str] = Field(
        default_factory=list,
        description="该场景下禁止或不建议使用的表达，例如保证收益、避税避债、恐吓式营销。",
    )
    # next_actions 是建议顾问继续做什么，而不是只给一句话术。
    next_actions: list[str] = Field(
        default_factory=list,
        description="建议顾问接下来执行的动作，例如补问 KYC、准备图表、安排轻量复访。",
    )
    # sources 保存摘要所依据的来源，最终可进入 citations。
    sources: list[dict] = Field(
        default_factory=list,
        description="摘要引用的来源信息，建议包含 source_id、chunk_id、score 和 scene。",
    )
    # compliance_notes 汇总多张卡片的合规提醒。
    compliance_notes: list[str] = Field(
        default_factory=list,
        description="压缩摘要层面的合规提醒，用于最终回答前的审查。",
    )


class CorpusBatch(BaseModel):
    """一次销售语料导入批次。"""

    id: str = Field(default_factory=lambda: new_id("corpus_batch"), description="语料批次唯一 ID。")
    tenant_id: str = Field(..., description="语料批次所属租户 ID。")
    batch_name: str = Field(..., description="批次名称，例如 2026Q2 高客访谈导入。")
    source_type: str = Field(..., description="语料来源类型，例如 interview、wechat_export、call_transcript。")
    upload_by: str = Field(default="", description="上传人或导入任务 ID。")
    raw_file_uri: str = Field(default="", description="原始文件 URI，只用于归档和审计，不进入生成 Prompt。")
    total_conversations: int = Field(default=0, ge=0, description="该批次包含的对话数量。")
    pii_status: Literal["raw", "redacted", "reviewed"] = Field(
        default="raw",
        description="PII 处理状态：原始、已脱敏或已人工复核。",
    )
    created_at: str = Field(default_factory=utc_now_iso, description="语料批次创建时间。")


class CorpusCase(BaseModel):
    """一个从真实语料中整理出的销售案例资产。"""

    id: str = Field(default_factory=lambda: new_id("corpus_case"), description="语料 case 唯一 ID。")
    tenant_id: str = Field(..., description="语料 case 所属租户 ID。")
    batch_id: str = Field(..., description="语料 case 所属导入批次 ID。")
    case_title: str = Field(..., description="case 标题，用于内部检索和人工复核。")
    scene_type: str = Field(default="", description="沟通场景类型，例如 KYC 补问、异议处理、低压维护。")
    target_persona: str = Field(default="unknown", description="该案例涉及的内部客群标签。")
    trigger_module: str = Field(default="unknown", description="该案例主要体现的销售切入模块。")
    advisor_stage: str = Field(default="unknown", description="案例中从业者阶段。")
    customer_stage: str = Field(default="unknown", description="案例中客户所处阶段或反应阶段。")
    relationship_strength: str = Field(default="", description="顾问与客户关系强度摘要。")
    final_outcome: str = Field(default="", description="案例最终结果标签或摘要。")
    quality_score: int = Field(default=0, ge=0, le=100, description="该案例作为训练/评测资产的质量分。")
    raw_conversation_uri: str = Field(default="", description="原始对话归档 URI，不进入最终 Prompt。")
    redacted_conversation_uri: str = Field(default="", description="脱敏对话归档 URI，可用于抽取和评测。")
    created_at: str = Field(default_factory=utc_now_iso, description="语料 case 创建时间。")


class CorpusMessage(BaseModel):
    """脱敏后的语料消息，只能作为抽取和评测来源。"""

    id: str = Field(default_factory=lambda: new_id("corpus_message"), description="语料消息唯一 ID。")
    tenant_id: str = Field(..., description="语料消息所属租户 ID。")
    corpus_case_id: str = Field(..., description="语料消息所属 case ID。")
    seq_no: int = Field(..., ge=1, description="消息在语料 case 中的顺序号。")
    speaker_role: Literal["advisor", "customer", "observer", "system"] = Field(..., description="消息说话方角色。")
    content_redacted: str = Field(..., description="脱敏后的消息内容，仅用于抽取模式或构造 eval，不直接进入生成。")
    message_type: str = Field(default="", description="消息类型，例如 question、answer、objection、closing。")
    detected_intent: str = Field(default="", description="从该消息中识别出的意图。")
    sentiment: str = Field(default="", description="该消息的情绪或态度标签。")
    created_at: str = Field(default_factory=utc_now_iso, description="语料消息创建时间。")


class DialoguePattern(BaseModel):
    """从真实对话中抽取、脱敏、审查后的销售动作模式。"""

    id: str = Field(default_factory=lambda: new_id("dialogue_pattern"), description="销售对话模式唯一 ID。")
    tenant_id: str = Field(..., description="模式所属租户 ID。")
    pattern_type: PatternType = Field(..., description="模式类型，例如 opening、kyc_question、objection_handling。")
    scene_type: str = Field(default="", description="模式适用的沟通场景。")
    target_persona: str = Field(default="unknown", description="模式适用的内部客群标签。")
    trigger_module: str = Field(default="unknown", description="模式适用的销售切入模块。")
    advisor_stage: str = Field(default="unknown", description="模式适合的从业者阶段。")
    situation_summary: str = Field(..., description="抽象后的情境摘要，不包含真实客户身份或完整故事。")
    customer_signal: str = Field(default="", description="客户在该场景下表现出的典型信号。")
    recommended_move: str = Field(..., description="建议的销售动作，必须是可迁移模式而非单个客户事实。")
    bad_move: str = Field(default="", description="该场景下不建议采取的动作。")
    example_wording: str = Field(default="", description="可参考的话术样本，必须已脱敏并通过合规审查。")
    outcome_label: str = Field(default="", description="该模式对应的结果标签，例如 replied、meeting_booked。")
    confidence: float = Field(default=0.8, ge=0, le=1, description="模式抽取或人工确认的置信度。")
    source_corpus_case_ids: list[str] = Field(default_factory=list, description="该模式来源的语料 case ID 列表。")
    approved_for_generation: bool = Field(
        default=False,
        description="该模式是否已允许进入最终生成。默认 False，避免未审真实语料被直接引用。",
    )
    risk_level: RiskLevel = Field(default="medium", description="模式风险等级；high 默认不能进入最终生成。")
    compliance_notes: str = Field(default="", description="模式审查备注，例如禁止承诺收益或杜绝避税避债表达。")
    created_at: str = Field(default_factory=utc_now_iso, description="模式创建时间。")
