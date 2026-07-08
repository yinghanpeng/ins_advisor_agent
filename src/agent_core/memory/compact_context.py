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
    case_state = _case_state_from_case(opportunity_case)
    case_state.update(
        {
            "kyc_completeness_score": kyc_completeness_score,
            "opportunity_score": opportunity_score,
            "external_grade": external_grade,
        }
    )

    return {
        "customer_profile": {
            "confirmed": _facts_to_mapping(confirmed_customer_facts),
            "uncertain": _facts_to_mapping(uncertain_customer_facts),
        },
        "advisor_profile": _advisor_facts_to_mapping(advisor_facts),
        "case_state": case_state,
        "missing_fields": list(dict.fromkeys(missing_fields)),
        "asked_focuses": list(dict.fromkeys(asked_focuses)),
        "support_note": support_note,
        "retrieved_patterns": _safe_pattern_summaries(retrieved_dialogue_patterns or []),
        "news_digest": news_digest,
    }


def _case_state_from_case(case: OpportunityCase | None) -> dict[str, Any]:
    """把 OpportunityCase 压缩成生成节点需要的阶段字段。"""
    if case is None:
        return {
            "subject_type": "",
            "target_persona": "",
            "trigger_module": "",
            "current_stage": "",
            "kyc_completeness_score": 0,
            "opportunity_score": 0,
            "external_grade": "",
        }
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
    result: dict[str, Any] = {}
    for fact in facts:
        if not fact.is_current or _is_pii_fact(fact):
            continue
        value = fact.normalized_value if fact.normalized_value is not None else fact.fact_value
        result[fact.fact_key] = value
    return result


def _advisor_facts_to_mapping(facts: list[AdvisorProfileFact]) -> dict[str, Any]:
    """从业者事实映射；只保留当前有效事实。"""
    result: dict[str, Any] = {}
    for fact in facts:
        if fact.is_current:
            result[fact.fact_key] = fact.fact_value
    return result


def _is_pii_fact(fact: CustomerProfileFact) -> bool:
    """判断客户事实是否属于默认不应进入 Prompt 的 PII。"""
    return fact.sensitivity_level == "pii" or fact.fact_key in PII_FACT_KEYS


def _safe_pattern_summaries(patterns: list[Any]) -> list[dict[str, Any]]:
    """把销售模式压缩成可生成摘要，过滤未审批和高风险模式。"""
    summaries: list[dict[str, Any]] = []
    for pattern in patterns:
        item = _to_plain_dict(pattern)
        if not item.get("approved_for_generation", False):
            continue
        if item.get("risk_level") == "high":
            continue
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
    return summaries


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """兼容 Pydantic 模型和普通 dict，方便节点和测试复用。"""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return dict()
