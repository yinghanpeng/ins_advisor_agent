"""生成节点使用的紧凑业务上下文。

Dify 原 workflow 中策略生成节点依赖大量散落变量，容易出现：
- 忘记带上某个评分或阶段；
- 把 uncertain 线索当成 confirmed 事实；
- 把客户 PII 或原始对话全文塞进 Prompt；
- 无法审计最终回答到底用了哪些业务依据。

本模块提供 build_compact_context，把业务记忆 store、KYC 分析输出、
销售模式摘要和外部素材统一压缩成一个结构化对象。最终生成节点应优先使用
compact_context，而不是直接拼接散落变量。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent_core.memory.business_schemas import (
    AdvisorProfileFact,
    CustomerProfileFact,
    OpportunityCase,
)


PII_FACT_KEYS = {
    # 即使上游 sensitivity_level 标错，命中这些显式键名的事实仍会被二次过滤。
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


def build_compact_context(
    *,
    confirmed_customer_facts: list[CustomerProfileFact],
    uncertain_customer_facts: list[CustomerProfileFact],
    advisor_facts: list[AdvisorProfileFact],
    opportunity_case: OpportunityCase | None,
    kyc_completeness_score: int,
    opportunity_score: int,
    external_grade: str,
    asked_focuses: list[str],
    missing_fields: list[str],
    support_note: str,
    retrieved_dialogue_patterns: list[Any] | None = None,
    method_knowledge: list[dict[str, Any]] | None = None,
    compliance_knowledge: list[dict[str, Any]] | None = None,
    news_digest: str = "",
) -> dict[str, Any]:
    """构建策略生成节点可使用的紧凑上下文。

    关键边界：
    1. confirmed 和 uncertain 分开输出；
    2. PII 或敏感事实默认不进入上下文；
    3. 不输出评分公式，只输出分数结果；
    4. 不输出原始历史全文；
    5. 销售语料只允许已审核模式摘要进入，而不是 CorpusMessage 原文。
    """
    # 先从可选 Case 提取稳定阶段字段；没有 Case 时 helper 返回相同结构的空默认值。
    case_state = _case_state_from_case(opportunity_case)
    # 本轮 KYC 分析结果优先于 Case 中的历史分数，覆盖三个可能已经变化的字段。
    case_state.update(
        {
            "kyc_completeness_score": kyc_completeness_score,
            "opportunity_score": opportunity_score,
            "external_grade": external_grade,
        }
    )

    # 返回结构固定的最小生成上下文，调用方无需接触原始消息或数据库模型。
    return {
        # confirmed/uncertain 保持两个独立分区，生成器可以采用不同措辞和证据等级。
        "customer_profile": {
            "confirmed": _facts_to_mapping(confirmed_customer_facts),
            "uncertain": _facts_to_mapping(uncertain_customer_facts),
        },
        # 顾问画像只保留当前有效键值，帮助教练调整建议深度。
        "advisor_profile": _advisor_facts_to_mapping(advisor_facts),
        # Case 聚合本轮路由、阶段及评分结果。
        "case_state": case_state,
        # dict.fromkeys 在保持原顺序的同时去重，避免重复字段造成重复补问。
        "missing_fields": list(dict.fromkeys(missing_fields)),
        "asked_focuses": list(dict.fromkeys(asked_focuses)),
        # support_note 是面向顾问的简短支持文本，不含评分公式。
        "support_note": support_note,
        # 销售模式在 helper 内执行生成准入和风险过滤后才进入上下文。
        "retrieved_patterns": _safe_pattern_summaries(retrieved_dialogue_patterns or []),
        # 方法库与合规库保持独立分区，生成节点不能把案例建议误当成合同事实。
        "method_knowledge": _safe_knowledge_summaries(method_knowledge or []),
        "compliance_knowledge": _safe_knowledge_summaries(compliance_knowledge or []),
        # news_digest 只接受内部新闻节点生成的摘要；信任来源由图节点边界校验。
        "news_digest": news_digest,
    }


def _case_state_from_case(case: OpportunityCase | None) -> dict[str, Any]:
    """把 OpportunityCase 压缩成生成节点需要的阶段字段。"""
    # 首轮没有 OpportunityCase 时仍返回完整字段集合，避免 Prompt 模板 KeyError。
    if case is None:
        # 没有 Case 时返回与正常分支相同的固定键集合和空默认值。
        return {
            "subject_type": "",
            "target_persona": "",
            "trigger_module": "",
            "current_stage": "",
            "kyc_completeness_score": 0,
            "opportunity_score": 0,
            "external_grade": "",
        }
    # 已有 Case 时仅复制生成所需字段，不暴露客户/顾问 ID 或内部说明。
    return {
        "subject_type": case.subject_type,
        "target_persona": case.target_persona,
        "trigger_module": case.trigger_module,
        "current_stage": case.current_stage,
        "kyc_completeness_score": case.latest_kyc_completeness_score,
        "opportunity_score": case.latest_opportunity_score,
        "external_grade": case.latest_external_grade,
    }


def _facts_to_mapping(facts: list[CustomerProfileFact]) -> dict[str, Any]:
    """客户事实映射；PII、失效事实和错误 certainty 不会进入对应分区。"""
    # 结果按 fact_key 索引；同键多版本由 Store 保证最多一个 current 事实。
    result: dict[str, Any] = {}
    # 逐条应用 current 与 PII 双重过滤，不能直接 model_dump 整个事实对象。
    for fact in facts:
        # 非当前版本或 PII 事实不允许进入任何客户画像摘要分区。
        if not fact.is_current or _is_pii_fact(fact):
            # 跳过失效或敏感事实，继续处理下一条。
            continue
        # 有标准化值时优先使用，便于下游比较同义表达；否则使用抽取原值。
        value = fact.normalized_value if fact.normalized_value is not None else fact.fact_value
        # 用稳定 fact_key 写入客户摘要映射。
        result[fact.fact_key] = value
    # 返回已过滤并按 fact_key 索引的客户事实映射。
    return result


def _advisor_facts_to_mapping(facts: list[AdvisorProfileFact]) -> dict[str, Any]:
    """从业者事实映射；只保留当前有效事实。"""
    # 顾问事实没有 normalized_value，仅按当前版本过滤后保留原始业务值。
    result: dict[str, Any] = {}
    # 逐条复制当前顾问事实；同键冲突应已在 Store 层关闭旧版本。
    for fact in facts:
        # 仅 current 事实允许进入生成上下文。
        if fact.is_current:
            # 将顾问当前事实写入摘要映射。
            result[fact.fact_key] = fact.fact_value
    # 返回只含 current 顾问事实的键值映射。
    return result


def _is_pii_fact(fact: CustomerProfileFact) -> bool:
    """判断客户事实是否属于默认不应进入 Prompt 的 PII。"""
    # sensitivity_level 是主判定，显式敏感键集合提供防御性兜底。
    return fact.sensitivity_level == "pii" or fact.fact_key in PII_FACT_KEYS


def _safe_pattern_summaries(patterns: list[Any]) -> list[dict[str, Any]]:
    """把销售模式压缩成可生成摘要，过滤未通过生成准入和高风险模式。"""
    # 只构建允许进入 Prompt 的低敏摘要列表，绝不返回原始语料对象。
    summaries: list[dict[str, Any]] = []
    # 对每条检索模式独立执行类型转换、准入和风险检查。
    for pattern in patterns:
        # 同时兼容 Pydantic 模型与普通字典，非法类型会转换为空字典并被默认拒绝。
        item = _to_plain_dict(pattern)
        # 生成准入采用默认拒绝；缺字段或显式 False 都不能进入 Prompt。
        if not item.get("approved_for_generation", False):
            # 未准入模式直接跳过。
            continue
        # 高风险模式即使已准入也不参与在线生成，避免危险销售表达被复用。
        if item.get("risk_level") == "high":
            # 高风险模式直接跳过。
            continue
        # 白名单复制业务字段，不携带原始消息、PII 或任意 metadata。
        summaries.append(
            {
                "id": item.get("id"),
                "pattern_type": item.get("pattern_type"),
                "scene_type": item.get("scene_type"),
                "target_persona": item.get("target_persona"),
                "trigger_module": item.get("trigger_module"),
                "situation_summary": item.get("situation_summary"),
                "customer_signal": item.get("customer_signal"),
                "recommended_move": item.get("recommended_move"),
                "bad_move": item.get("bad_move"),
                "example_wording": item.get("example_wording"),
                "outcome_label": item.get("outcome_label"),
                "confidence": item.get("confidence"),
            }
        )
    # 返回通过准入与风险过滤的白名单模式摘要。
    return summaries


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """兼容 Pydantic 模型和普通 dict，方便节点和测试复用。"""
    # Pydantic 模型使用官方序列化接口，确保别名和嵌套类型按 Schema 输出。
    if isinstance(value, BaseModel):
        # Pydantic 输入转换为普通字典后返回。
        return value.model_dump()
    # 普通字典已满足调用要求，直接复用而不做深拷贝。
    if isinstance(value, dict):
        # 已是字典的输入直接返回。
        return value
    # 未知对象返回空字典，后续 default-deny 校验会安全过滤。
    return dict()


def _safe_knowledge_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """限制双知识库内容长度并保留 Grounding 所需来源字段。"""
    # 最终列表只保留经过准入且具有非空内容的知识片段。
    summaries: list[dict[str, Any]] = []
    # 对每条知识片段应用 mapping 类型、准入标记及内容非空校验。
    for item in items:
        # 非 mapping 项不进入生成上下文。
        if not isinstance(item, dict):
            # 结构异常知识项无法安全检查 metadata，默认拒绝。
            continue
        # metadata 类型异常时回退空字典，使 approved_for_generation 缺失并默认拒绝。
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        # 生成准入字段采用 default deny：历史数据缺字段时视为未准入，不能因检索分高进入 Prompt。
        if metadata.get("approved_for_generation", False) is not True:
            # 未获得字面量 True 准入的知识项直接跳过。
            continue
        # 统一转换并清理内容；空内容无法提供 Grounding 证据，直接排除。
        content = str(item.get("content") or "").strip()
        # 空白内容无法提供任何 Grounding 依据，直接跳过。
        if not content:
            # 空内容跳过，不生成无证据摘要。
            continue
        # 内容限制为 1200 字符，并只复制 Grounding 所需的文档、分块和来源字段。
        summaries.append(
            {
                "content": content[:1200],
                "score": float(item.get("score") or 0.0),
                "document_id": str(item.get("document_id") or ""),
                "chunk_id": str(item.get("chunk_id") or ""),
                "source_uri": item.get("source_uri"),
                "metadata": {
                    "knowledge_type": metadata.get("knowledge_type"),
                    "library": metadata.get("library"),
                    "version": metadata.get("version"),
                },
            }
        )
    # 返回通过生成准入、非空和长度限制的知识摘要列表。
    return summaries
