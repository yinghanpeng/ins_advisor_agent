"""Sales Intelligence schemas."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_core.utils.time import utc_now_iso


BusinessStage = Literal[
    "new_customer",
    "old_customer_maintenance",
    "add_on",
    "referral",
    "proposal",
    "closing",
    "unknown",
]

RiskLevel = Literal["low", "medium", "high"]


class CustomerKYC(BaseModel):
    """客户 KYC 摘要，用于把访谈中的客户背景转成可检索、可复用的结构。"""

    age: str | None = Field(
        default=None,
        description="客户年龄或年龄段，例如 45岁左右、60后、刚成家。未知时为空。",
    )
    family: str | None = Field(
        default=None,
        description="客户家庭结构和责任信息，例如已婚、有两个孩子、父母赡养压力。",
    )
    occupation: str | None = Field(
        default=None,
        description="客户职业或收入来源，例如制造业企业主、医生、互联网高管、全职太太。",
    )
    asset_preference: str | None = Field(
        default=None,
        description="客户资产偏好或既有配置习惯，例如偏银行理财、重视现金流、抗拒长期锁定。",
    )
    decision_style: str | None = Field(
        default=None,
        description="客户决策风格，例如谨慎型、听配偶意见、重视熟人案例、讨厌被推销。",
    )


class SalesInsightCard(BaseModel):
    """销售洞察卡片，是销售访谈资产化后的核心结构。"""

    source_id: str = Field(
        ...,
        description="原始访谈、转写稿或销售素材的来源 ID，用于审计和回溯。",
    )
    chunk_id: str = Field(
        ...,
        description="该卡片对应的访谈片段 ID。一个 source_id 可拆分出多个 chunk。",
    )
    interviewee_role: str = Field(
        ...,
        description="受访者角色，例如资深保险顾问、团队长、银保渠道经理。",
    )
    sales_experience_years: float | None = Field(
        default=None,
        description="受访者销售从业年限。用于区分经验来源成熟度，未知时为空。",
    )
    channel: str | None = Field(
        default=None,
        description="销售场景或渠道，例如高净值转介绍、银行网点、私董会、老客户加保。",
    )
    business_stage: BusinessStage = Field(
        default="unknown",
        description="业务阶段，例如新客户破冰、老客户维护、计划书、成交收口。",
    )
    scene: str = Field(
        ...,
        description="具体沟通场景，例如饭局破冰、KYC 深挖、异议处理、计划书讲解。",
    )
    customer_type: str = Field(
        ...,
        description="客户类型标签，例如企业主、医生、家庭主妇、退休客户、高净值二代。",
    )
    customer_kyc: CustomerKYC = Field(
        default_factory=CustomerKYC,
        description="该经验卡片对应的客户 KYC 信息，用于检索匹配和话术个性化。",
    )
    sales_pain_solved: str = Field(
        ...,
        description="这条经验解决的销售痛点，例如不会自然切入保险、客户抗拒谈保障。",
    )
    root_cause: str = Field(
        ...,
        description="痛点背后的原因分析，例如客户防御感强、销售过早讲产品、信任未建立。",
    )
    effective_strategy: str = Field(
        ...,
        description="被访谈验证有效的沟通策略，应该是可迁移的方法，不只是单句口号。",
    )
    usable_script: str = Field(
        ...,
        description="可直接改写给顾问使用的安全话术。不得包含保证收益、避税避债等违规表达。",
    )
    wrong_way: str = Field(
        ...,
        description="该场景下不建议采用的错误做法，用于提醒新手避坑。",
    )
    why_it_works: str = Field(
        ...,
        description="解释该策略为什么有效，例如降低防御、先共情再引导、让客户先谈用途。",
    )
    next_question: str = Field(
        ...,
        description="建议销售下一句追问，用于把沟通自然推进到更深层 KYC 或下一步动作。",
    )
    customer_response: str | None = Field(
        default=None,
        description="访谈中记录的客户典型反应。没有明确记录时为空。",
    )
    follow_up_action: str | None = Field(
        default=None,
        description="建议的后续动作，例如准备资金分层图、约 15 分钟复盘、补充家庭责任信息。",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="检索和评估标签，例如 破冰、企业主、资金分层、异议处理。",
    )
    risk_level: RiskLevel = Field(
        default="low",
        description="卡片风险等级。high 卡片默认不直接用于生成，应先合规审查或人工审批。",
    )
    compliance_notes: str = Field(
        default="",
        description="合规备注，记录需要避免的表达、引用限制或审查结论。",
    )
    suitable_for_rag: bool = Field(
        default=True,
        description="该卡片是否适合作为 RAG 检索资料。噪声过大或未结构化内容应设为 False。",
    )
    suitable_for_eval: bool = Field(
        default=True,
        description="该卡片是否适合作为评估样本来源。用于自动生成 sales quality eval case。",
    )
    approved_for_generation: bool = Field(
        default=False,
        description="该卡片是否已通过审查、允许进入最终生成。默认 False，防止原始访谈直接出现在回答中。",
    )
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="卡片创建时间，ISO 字符串，用于版本追踪和审计。",
    )
    updated_at: str = Field(
        default_factory=utc_now_iso,
        description="卡片最近更新时间，ISO 字符串，用于后续人工修订或再审查。",
    )


class SalesInsightDigest(BaseModel):
    """检索到的销售洞察压缩摘要，供最终回答生成使用。"""

    applicable_scene: str = Field(
        ...,
        description="这份摘要适用的沟通场景，例如饭局破冰、宏观共鸣、异议处理。",
    )
    insight_summary: str = Field(
        ...,
        description="对多条销售经验卡片压缩后的核心洞察。应保留策略，不堆砌原文。",
    )
    usable_scripts: list[str] = Field(
        default_factory=list,
        description="可以安全改写给用户的候选话术列表。",
    )
    forbidden_expressions: list[str] = Field(
        default_factory=list,
        description="该场景下禁止或不建议使用的表达，例如保证收益、避税避债、恐吓式营销。",
    )
    next_actions: list[str] = Field(
        default_factory=list,
        description="建议顾问接下来执行的动作，例如补问 KYC、准备图表、安排轻量复访。",
    )
    sources: list[dict] = Field(
        default_factory=list,
        description="摘要引用的来源信息，建议包含 source_id、chunk_id、score 和 scene。",
    )
    compliance_notes: list[str] = Field(
        default_factory=list,
        description="压缩摘要层面的合规提醒，用于最终回答前的审查。",
    )


def sample_card() -> SalesInsightCard:
    """返回一张已合规审批的示例销售洞察卡片，供本地检索、测试和演示使用。"""
    return SalesInsightCard(
        source_id="sample_interview_001",
        chunk_id="chunk_001",
        interviewee_role="资深保险顾问",
        sales_experience_years=8,
        channel="高净值客户转介绍",
        business_stage="new_customer",
        scene="饭局破冰",
        customer_type="企业主",
        customer_kyc=CustomerKYC(
            age="45岁左右",
            family="两个孩子",
            occupation="制造业企业主",
            asset_preference="偏好银行理财",
            decision_style="谨慎，重视现金流",
        ),
        sales_pain_solved="不知道如何从闲聊自然切入长期资金规划",
        root_cause="从业者过早讲产品，客户还没有建立资金分层意识",
        effective_strategy="先围绕经营现金流和家庭责任共情，再用资金分层把话题转到长期稳定安排。",
        usable_script="最近很多老板不是没有赚钱能力，而是更在意哪些钱不能被经营波动打乱。",
        wrong_way="一上来讲保险收益、港保优势或资产隔离。",
        why_it_works="低压共情能降低防御感，资金分层让客户先讨论用途而不是产品。",
        next_question="这笔钱更偏企业备用，还是家庭长期不能动的钱？",
        customer_response="客户愿意聊资金用途",
        follow_up_action="准备一张资金分层图，约15分钟只聊资金用途。",
        tags=["破冰", "企业主", "资金分层", "低压沟通"],
        risk_level="low",
        compliance_notes="不得承诺收益；不得使用避债避税表达。",
        approved_for_generation=True,
    )
