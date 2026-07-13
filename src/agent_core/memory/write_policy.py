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

from agent_core.guardrails.output_pii import scan_and_redact_output_pii
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
    # 这些键表示模型生成的销售建议，绝不能被当作客户既有事实长期保存。
    "strategy",
    "recommended_move",
    "next_best_action",
    "proposal",
    "script",
    "talking_points",
}

PII_FACT_KEYS = {
    # 这些常见字段名代表可直接识别个人的信息，通用业务记忆默认拒绝存储。
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

    # 待 upsert 的长期事实；执行前必须逐条通过证据和敏感性校验。
    facts_to_upsert: list[AdvisorProfileFact | CustomerProfileFact] = Field(
        default_factory=list,
        description="建议写入或更新的长期画像事实。必须有 source_type 和 evidence_text。",
    )
    # 待追加的业务事件；事件只描述已发生信息，不能充当事实覆盖操作。
    events_to_insert: list[MemoryEvent] = Field(
        default_factory=list,
        description="建议写入的事件记忆，例如客户异议、正向信号、策略已生成。",
    )
    # 实际已经展示给用户的 KYC 问题，用于后续防止重复追问。
    questions_to_record: list[KYCQuestion] = Field(
        default_factory=list,
        description="建议记录的 KYC 补问，用于避免后续重复追问同一焦点。",
    )
    # 可选工作快照；每个 Proposal 最多持久化一份本轮 Session 状态。
    session_state_to_insert: AgentSessionState | None = Field(
        default=None,
        description="建议保存的一轮会话工作记忆快照；为空表示本轮不保存快照。",
    )
    # 可选分析审计记录，保存本轮 KYC 分类和评分的输入输出快照。
    analysis_run_to_insert: AnalysisRun | None = Field(
        default=None,
        description="建议保存的一次 KYC 分析运行记录；为空表示本轮没有分析输出。",
    )
    # 可选生成结果记录；与 facts 分离可避免模型建议污染客户画像。
    generated_output_to_insert: GeneratedOutput | None = Field(
        default=None,
        description="建议保存的一次生成输出；策略和话术只能写这里，不能写成客户事实。",
    )
    # 明确拒绝持久化的候选及原因，便于审计“为何没有写入”。
    do_not_store: list[dict[str, Any]] = Field(
        default_factory=list,
        description="明确不写入长期记忆的内容和原因，例如 PII、无证据推测、一次性情绪。",
    )


class MemoryWriteValidationResult(BaseModel):
    """记忆写入计划校验结果。"""

    # 只要 errors 非空便为 False；warnings 不会单独阻断整个 Proposal。
    is_valid: bool = Field(..., description="整体写入计划是否可执行。")
    # errors 保存需要阻断的确定性违规原因。
    errors: list[str] = Field(default_factory=list, description="阻断写入的错误原因列表。")
    # warnings 保存允许写入但需要后续观察的低风险提醒。
    warnings: list[str] = Field(default_factory=list, description="不阻断但需要审计关注的提醒。")
    # 以下四组 ID 将校验结果与原 Proposal 精确关联，持久化节点据此过滤记录。
    blocked_fact_ids: list[str] = Field(default_factory=list, description="被策略阻断写入的事实 ID。")
    allowed_fact_ids: list[str] = Field(default_factory=list, description="通过策略校验、允许写入的事实 ID。")
    blocked_record_ids: list[str] = Field(default_factory=list, description="被阻断的事件、问题、快照或输出 ID。")
    allowed_record_ids: list[str] = Field(default_factory=list, description="通过校验的非事实记录 ID。")


def validate_memory_write_proposal(proposal: MemoryWriteProposal) -> MemoryWriteValidationResult:
    """校验记忆写入计划是否符合业务事实边界。"""
    # 分别收集错误、提醒及事实/非事实记录的允许与阻断集合，避免混淆处理。
    errors: list[str] = []
    # warnings 保存允许继续但生成阶段必须谨慎使用的提醒。
    warnings: list[str] = []
    # blocked_fact_ids 保存未通过严格事实边界的 ID。
    blocked_fact_ids: list[str] = []
    # allowed_fact_ids 保存可以执行 Upsert 的事实 ID。
    allowed_fact_ids: list[str] = []
    # blocked_record_ids 保存被拒绝的事件、问题、快照或输出 ID。
    blocked_record_ids: list[str] = []
    # allowed_record_ids 保存通过验证的非事实记录 ID。
    allowed_record_ids: list[str] = []

    # 长期事实约束最严格，逐条验证来源、证据、PII 和“建议冒充事实”等风险。
    for fact in proposal.facts_to_upsert:
        # 对当前事实执行全部业务边界校验。
        fact_errors = _validate_fact(fact)
        # 任一事实错误都会阻断该事实并使整个 Proposal 标记为不可执行。
        if fact_errors:
            # 将事实 ID 前缀加入每条错误，便于定位 Proposal 项。
            errors.extend(f"{fact.id}: {error}" for error in fact_errors)
            # 当前事实加入阻断 ID 集合。
            blocked_fact_ids.append(fact.id)
        # 当前事实无错误时进入允许分支。
        else:
            # 没有错误的事实加入允许集合，供 persist 节点做白名单过滤。
            allowed_fact_ids.append(fact.id)
            # uncertain 可以保存为线索，但必须显式提醒生成阶段不可当作确定事实。
            if isinstance(fact, CustomerProfileFact) and fact.certainty == "uncertain":
                # 记录 uncertain 可存但不可当作明确事实的警告。
                warnings.append(f"{fact.id}: 不确定线索已按 uncertain 写入，生成时不得当作明确事实。")

    # Event 必须有证据，Payload 不允许携带未脱敏 PII。
    for event in proposal.events_to_insert:
        # 验证当前事件的证据、PII 和业务关联。
        record_errors = _validate_event(event)
        # 将事件校验结果写入统一记录 ID 白/黑名单。
        _collect_record_validation(
            event.id,
            record_errors,
            errors,
            blocked_record_ids,
            allowed_record_ids,
        )

    # KYCQuestion 是可回放业务记录，也需要租户、关联 ID 和 PII 检查。
    for question in proposal.questions_to_record:
        # 验证当前问题的焦点、正文和 PII。
        record_errors = _validate_question(question)
        # 将问题校验结果写入统一记录 ID 白/黑名单。
        _collect_record_validation(
            question.id,
            record_errors,
            errors,
            blocked_record_ids,
            allowed_record_ids,
        )

    # 三种单例记录用“记录 + 对应验证器”配对，复用统一汇总逻辑。
    optional_records = [
        (proposal.session_state_to_insert, _validate_session_state),
        (proposal.analysis_run_to_insert, _validate_analysis_run),
        (proposal.generated_output_to_insert, _validate_generated_output),
    ]
    # 仅验证本轮实际存在的可选记录；None 代表该轮没有相应产物。
    for record, validator in optional_records:
        # 本轮未产生该类记录时直接跳过对应验证器。
        if record is None:
            # 继续处理下一个可选记录类型。
            continue
        # 调用与当前记录类型配对的验证器并汇总结果。
        _collect_record_validation(
            record.id,
            validator(record),
            errors,
            blocked_record_ids,
            allowed_record_ids,
        )

    # do_not_store 必须带原因，且引用的记录不能同时出现在写入列表中。
    # 汇总所有计划写入 ID，用于检测 do_not_store 与写入列表的自相矛盾。
    proposal_ids = {
        *[item.id for item in proposal.facts_to_upsert],
        *[item.id for item in proposal.events_to_insert],
        *[item.id for item in proposal.questions_to_record],
    }
    # 将三个可能存在的单例记录 ID 合并到同一集合。
    proposal_ids.update(
        record.id
        for record in [
            proposal.session_state_to_insert,
            proposal.analysis_run_to_insert,
            proposal.generated_output_to_insert,
        ]
        if record is not None
    )
    # do_not_store 的每一项必须解释拒绝原因，否则审计记录没有决策依据。
    for index, item in enumerate(proposal.do_not_store):
        # 非字典或缺少非空 reason 的拒存项无法形成有效审计依据。
        if not isinstance(item, dict) or not str(item.get("reason") or "").strip():
            # 缺少拒存理由时追加阻断错误。
            errors.append(f"do_not_store[{index}]: 必须提供非空 reason。")
            # 当前项无法继续做 record_id 一致性检查。
            continue
        # record_id 可为空；非空时检查它是否又出现在允许写入候选中。
        blocked_id = str(item.get("record_id") or "")
        # 同一 ID 不能既声明拒存又出现在执行写入集合中。
        if blocked_id and blocked_id in proposal_ids:
            # 同一 ID 同时拒存和写入时追加契约冲突错误。
            errors.append(f"do_not_store[{index}]: {blocked_id} 同时出现在写入计划中。")
            # 将冲突 ID 加入非事实阻断集合。
            blocked_record_ids.append(blocked_id)

    # errors 是否为空决定整体有效性，同时返回细粒度 ID 便于安全持久化。
    return MemoryWriteValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        blocked_fact_ids=blocked_fact_ids,
        allowed_fact_ids=allowed_fact_ids,
        blocked_record_ids=blocked_record_ids,
        allowed_record_ids=allowed_record_ids,
    )


def filter_allowed_facts(
    proposal: MemoryWriteProposal,
    validation: MemoryWriteValidationResult,
) -> list[AdvisorProfileFact | CustomerProfileFact]:
    """返回通过校验的事实，供 persist 节点执行写入。"""
    # 使用 set 将允许 ID 查询降为常数时间，避免逐条列表扫描。
    allowed = set(validation.allowed_fact_ids)
    # 保持 Proposal 原始顺序，只移除未进入允许集合的事实。
    return [fact for fact in proposal.facts_to_upsert if fact.id in allowed]


def _validate_fact(fact: AdvisorProfileFact | CustomerProfileFact) -> list[str]:
    """校验单条长期事实是否可写入。"""
    # 同一事实可能同时违反多个约束，因此累积全部错误而不是首错即返回。
    errors: list[str] = []
    # 来源类型用于区分用户明示、分析抽取或导入语料，缺失时不可追溯。
    if not fact.source_type:
        # 记录来源缺失错误。
        errors.append("长期事实缺少 source_type。")
    # evidence_text 必须包含非空白原文，防止模型凭空生成画像事实。
    if not fact.evidence_text or not fact.evidence_text.strip():
        # 记录证据缺失错误。
        errors.append("长期事实缺少 evidence_text。")
    # 建议类键属于生成产物而非主体事实，必须走 GeneratedOutput 表。
    if fact.fact_key in GENERATED_ADVICE_KEYS:
        # 记录生成建议越界写入事实表错误。
        errors.append("模型生成的建议不能写成客户或从业者画像事实。")
    # 客户事实额外检查 PII 与 certainty；顾问事实不含这两个字段。
    if isinstance(fact, CustomerProfileFact):
        # 标记为 pii 或使用显式 PII 键名的客户事实均默认阻断。
        if fact.sensitivity_level == "pii" or fact.fact_key in PII_FACT_KEYS:
            # 记录 PII 长期事实阻断错误。
            errors.append("PII 默认不得写入长期 prompt 记忆。")
        # certainty 只允许确认/待确认两态，未知值不能进入生成上下文。
        if fact.certainty not in {"confirmed", "uncertain"}:
            # 记录未知 certainty 错误。
            errors.append("客户事实 certainty 必须是 confirmed 或 uncertain。")
    # 返回该事实命中的全部错误，空列表表示允许写入。
    return errors


def _validate_event(event: MemoryEvent) -> list[str]:
    """校验事件证据和脱敏 Payload。"""
    # 事件同样允许一次返回多条错误，方便调用方集中修复 Proposal。
    errors: list[str] = []
    # 没有证据的事件无法证明真实发生，不能进入长期业务时间线。
    if not event.evidence_text.strip():
        # 记录事件无证据错误。
        errors.append("事件记忆缺少 evidence_text。")
    # 结构化事件 Payload 命中 PII 时整条事件阻断，不能只删除局部值后静默保存。
    if _contains_pii(event.event_payload):
        # 记录事件 Payload PII 错误。
        errors.append("事件 Payload 包含未脱敏 PII。")
    # 四个业务关联均为空时事件无法归属任何主体或任务。
    if not any([event.conversation_id, event.opportunity_case_id, event.customer_id, event.advisor_id]):
        # 记录事件无业务归属错误。
        errors.append("事件至少需要一个业务关联 ID。")
    # 返回事件证据、PII 和业务关联的完整错误集合。
    return errors


def _validate_question(question: KYCQuestion) -> list[str]:
    """校验 KYC 问题的关联关系与输出安全。"""
    # 聚合字段完整性和 PII 两类错误，避免只暴露第一个问题。
    errors: list[str] = []
    # 焦点键和实际问题正文都不能为空白，否则无法去重或回放。
    if not question.focus_key.strip() or not question.question_text.strip():
        # 记录问题关键字段缺失错误。
        errors.append("KYC 问题缺少 focus_key 或 question_text。")
    # 用户可见问题命中 PII 时阻断记录，避免敏感文本进入后续回放。
    if _contains_pii(question.question_text):
        # 记录问题正文 PII 错误。
        errors.append("KYC 问题包含未脱敏 PII。")
    # 返回问题字段完整性与 PII 错误集合。
    return errors


def _validate_session_state(state: AgentSessionState) -> list[str]:
    """校验 Session Snapshot 不保存 PII 或越界生成建议。"""
    # Session Snapshot 会被后续轮次恢复，因此必须在写入前一次性清理越界内容。
    errors: list[str] = []
    # 同时检查客户与顾问画像的嵌套值和敏感字段名。
    if _contains_pii({"profile": state.profile_state, "practitioner": state.practitioner_state}):
        # 记录 Session Snapshot PII 错误。
        errors.append("Session Snapshot 包含未脱敏 PII。")
    # 客户画像快照出现建议键表示模型产物越界混入事实状态。
    if any(key in GENERATED_ADVICE_KEYS for key in state.profile_state):
        # 记录画像中混入生成建议的错误。
        errors.append("Session profile_state 不能混入模型生成建议。")
    # 返回 Snapshot 中 PII/建议越界的全部错误。
    return errors


def _validate_analysis_run(run: AnalysisRun) -> list[str]:
    """校验分析快照、证据和输出结构。"""
    # 分析记录必须既可解释又不泄露 PII，因此分别检查证据和输入输出快照。
    errors: list[str] = []
    # 分析运行没有明确证据时不可审计，必须阻断。
    if not run.match_evidence.strip():
        # 记录分析运行缺少匹配证据错误。
        errors.append("AnalysisRun 缺少 match_evidence。")
    # 输入和输出快照作为整体递归扫描，任一侧命中 PII 都阻断记录。
    if _contains_pii({"input": run.input_snapshot, "output": run.output_json}):
        # 记录分析输入/输出包含 PII 错误。
        errors.append("AnalysisRun 包含未脱敏 PII。")
    # 返回 AnalysisRun 的证据和 PII 校验错误。
    return errors


def _validate_generated_output(output: GeneratedOutput) -> list[str]:
    """校验生成输出和输入上下文的 PII 边界。"""
    # 最终输出需要非空且无 PII，两个约束都在持久化前确定性执行。
    errors: list[str] = []
    # 空白生成结果没有保存价值，也不能形成策略到结果的映射。
    if not output.output_text.strip():
        # 记录生成正文为空错误。
        errors.append("GeneratedOutput 为空。")
    # 输入上下文或最终正文任一命中 PII 均阻断整条生成记录。
    if _contains_pii(output.input_context) or _contains_pii(output.output_text):
        # 记录生成上下文或正文 PII 错误。
        errors.append("GeneratedOutput 或其输入上下文包含未脱敏 PII。")
    # 返回 GeneratedOutput 的非空与 PII 校验错误。
    return errors


def _collect_record_validation(
    record_id: str,
    record_errors: list[str],
    errors: list[str],
    blocked_ids: list[str],
    allowed_ids: list[str],
) -> None:
    """把单条非事实记录校验结果汇总到 Proposal 结果。"""
    # 有错误时附加 record_id，确保多条记录并行验证时仍可定位来源。
    if record_errors:
        # 将记录 ID 加入每条错误并合并到总错误列表。
        errors.extend(f"{record_id}: {error}" for error in record_errors)
        # 当前记录加入阻断 ID 列表。
        blocked_ids.append(record_id)
    # 没有错误时进入允许记录分支。
    else:
        # 完全无错误的记录进入允许 ID 白名单，供持久化阶段执行。
        allowed_ids.append(record_id)


def _contains_pii(value: Any) -> bool:
    """递归扫描结构化值是否包含输出 PII 规则命中。"""
    # 字典同时检查键和值：敏感键即便值已替换也不应进入通用快照。
    if isinstance(value, dict):
        # 即使值已被占位符替换，PII 字段名本身也不应进入通用快照。
        if any(str(key).casefold() in PII_FACT_KEYS for key in value):
            # 命中显式敏感键名时立即返回 True。
            return True
        # 未命中敏感键时递归检查字典全部值。
        return any(_contains_pii(item) for item in value.values())
    # 列表递归检查每个元素，支持任意深度的嵌套结构。
    if isinstance(value, list):
        # 任一列表元素递归命中即可判定整体包含 PII。
        return any(_contains_pii(item) for item in value)
    # 字符串复用统一输出 PII 扫描器，triggered 表示命中任一敏感模式。
    if isinstance(value, str):
        # 复用输出扫描器并将 triggered 标志转换为布尔返回值。
        return bool(scan_and_redact_output_pii(value)[1]["triggered"])
    # 数字、布尔和 None 等标量不携带可被当前规则识别的文本 PII。
    return False
