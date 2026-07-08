"""业务记忆写入策略。

MemoryWriteProposal 是节点和持久化 store 之间的缓冲层：
- 分析节点只提出“建议写什么”；
- validator 决定这些写入是否符合证据、PII 和事实边界；
- persist 节点只执行通过校验的写入。

这样可以避免模型把自己生成的策略误写成客户事实，也避免没有证据的推测污染长期记忆。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_core.memory.business_schemas import (
    AdvisorProfileFact,
    AgentSessionState,
    AnalysisRun,
    CustomerProfileFact,
    GeneratedOutput,
    KYCQuestion,
    MemoryEvent,
)


GENERATED_ADVICE_KEYS = {
    "strategy",
    "recommended_move",
    "next_best_action",
    "proposal",
    "script",
    "talking_points",
}

PII_FACT_KEYS = {
    "name",
    "real_name",
    "phone",
    "wechat",
    "id_card",
    "passport",
    "email",
    "address",
    "exact_address",
}


class MemoryWriteProposal(BaseModel):
    """一次节点运行提出的记忆写入计划。"""

    facts_to_upsert: list[AdvisorProfileFact | CustomerProfileFact] = Field(
        default_factory=list,
        description="建议写入或更新的长期画像事实。必须有 source_type 和 evidence_text。",
    )
    events_to_insert: list[MemoryEvent] = Field(
        default_factory=list,
        description="建议写入的事件记忆，例如客户异议、正向信号、策略已生成。",
    )
    questions_to_record: list[KYCQuestion] = Field(
        default_factory=list,
        description="建议记录的 KYC 补问，用于避免后续重复追问同一焦点。",
    )
    session_state_to_insert: AgentSessionState | None = Field(
        default=None,
        description="建议保存的一轮会话工作记忆快照；为空表示本轮不保存快照。",
    )
    analysis_run_to_insert: AnalysisRun | None = Field(
        default=None,
        description="建议保存的一次 KYC 分析运行记录；为空表示本轮没有分析输出。",
    )
    generated_output_to_insert: GeneratedOutput | None = Field(
        default=None,
        description="建议保存的一次生成输出；策略和话术只能写这里，不能写成客户事实。",
    )
    do_not_store: list[dict[str, Any]] = Field(
        default_factory=list,
        description="明确不写入长期记忆的内容和原因，例如 PII、无证据推测、一次性情绪。",
    )


class MemoryWriteValidationResult(BaseModel):
    """记忆写入计划校验结果。"""

    is_valid: bool = Field(..., description="整体写入计划是否可执行。")
    errors: list[str] = Field(default_factory=list, description="阻断写入的错误原因列表。")
    warnings: list[str] = Field(default_factory=list, description="不阻断但需要审计关注的提醒。")
    blocked_fact_ids: list[str] = Field(default_factory=list, description="被策略阻断写入的事实 ID。")
    allowed_fact_ids: list[str] = Field(default_factory=list, description="通过策略校验、允许写入的事实 ID。")


def validate_memory_write_proposal(proposal: MemoryWriteProposal) -> MemoryWriteValidationResult:
    """校验记忆写入计划是否符合业务事实边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    blocked_fact_ids: list[str] = []
    allowed_fact_ids: list[str] = []

    for fact in proposal.facts_to_upsert:
        fact_errors = _validate_fact(fact)
        if fact_errors:
            errors.extend(f"{fact.id}: {error}" for error in fact_errors)
            blocked_fact_ids.append(fact.id)
        else:
            allowed_fact_ids.append(fact.id)
            if isinstance(fact, CustomerProfileFact) and fact.certainty == "uncertain":
                warnings.append(f"{fact.id}: 不确定线索已按 uncertain 写入，生成时不得当作明确事实。")

    return MemoryWriteValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        blocked_fact_ids=blocked_fact_ids,
        allowed_fact_ids=allowed_fact_ids,
    )


def filter_allowed_facts(
    proposal: MemoryWriteProposal,
    validation: MemoryWriteValidationResult,
) -> list[AdvisorProfileFact | CustomerProfileFact]:
    """返回通过校验的事实，供 persist 节点执行写入。"""
    allowed = set(validation.allowed_fact_ids)
    return [fact for fact in proposal.facts_to_upsert if fact.id in allowed]


def _validate_fact(fact: AdvisorProfileFact | CustomerProfileFact) -> list[str]:
    """校验单条长期事实是否可写入。"""
    errors: list[str] = []
    if not fact.source_type:
        errors.append("长期事实缺少 source_type。")
    if not fact.evidence_text or not fact.evidence_text.strip():
        errors.append("长期事实缺少 evidence_text。")
    if fact.fact_key in GENERATED_ADVICE_KEYS:
        errors.append("模型生成的建议不能写成客户或从业者画像事实。")
    if isinstance(fact, CustomerProfileFact):
        if fact.sensitivity_level == "pii" or fact.fact_key in PII_FACT_KEYS:
            errors.append("PII 默认不得写入长期 prompt 记忆。")
        if fact.certainty not in {"confirmed", "uncertain"}:
            errors.append("客户事实 certainty 必须是 confirmed 或 uncertain。")
    return errors
