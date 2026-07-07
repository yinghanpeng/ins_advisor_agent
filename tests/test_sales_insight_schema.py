# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.sales_intelligence.schemas import SalesInsightCard, sample_card


def test_sales_insight_card_schema_validates_sample():
    card = sample_card()
    dumped = card.model_dump()
    reloaded = SalesInsightCard.model_validate(dumped)
    assert reloaded.source_id == "sample_interview_001"
    assert reloaded.approved_for_generation is True
    assert reloaded.risk_level == "low"


def test_sales_insight_card_json_schema_has_required_fields():
    schema = SalesInsightCard.model_json_schema()
    assert "source_id" in schema["properties"]
    assert "approved_for_generation" in schema["properties"]

