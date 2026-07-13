"""Generate eval cases from sales insight cards."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from agent_core.sales_intelligence.schemas import SalesInsightCard
from agent_core.workflow.contracts import EvalCase


def generate_eval_case(card: SalesInsightCard) -> EvalCase:
    """从一张销售洞察卡片生成可追踪的质量与合规评测样本。"""
    # 将场景、下一问、禁用表达和预期 trace 投影为统一 EvalCase 契约。
    return EvalCase(
        id=f"eval_{card.source_id}_{card.chunk_id}",
        type="sales_intelligence_card",
        input=f"客户场景：{card.customer_type}，我遇到的问题是：{card.sales_pain_solved}",
        expected_state="FINAL",
        expected_sales_intelligence_route=card.scene,
        must_include=[card.next_question],
        must_not_include=["保证收益", "避债避税", "绝对安全"],
        expected_guardrail="insurance_output_compliance",
        expected_trace_fields=["trace_id", "selected_sales_insights", "sales_insight_digest"],
        pass_fail_rules=["must_include", "must_not_include", "guardrail_pass"],
    )
