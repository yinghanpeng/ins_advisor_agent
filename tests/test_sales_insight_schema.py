# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.sales_intelligence.schemas import CustomerKYC, SalesInsightCard


def test_sales_insight_card_schema_validates_sample():
    card = SalesInsightCard(
        source_id="test_interview_001",
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
        root_cause="从业者过早讲产品",
        effective_strategy="先围绕经营现金流和家庭责任共情。",
        usable_script="最近很多老板更在意哪些钱不能被经营波动打乱。",
        wrong_way="一上来讲保险收益。",
        why_it_works="低压共情能降低防御感。",
        next_question="这笔钱更偏企业备用，还是家庭长期不能动的钱？",
        customer_response="客户愿意聊资金用途",
        tags=["破冰", "企业主"],
        risk_level="low",
        approved_for_generation=True,
    )
    dumped = card.model_dump()
    reloaded = SalesInsightCard.model_validate(dumped)
    assert reloaded.source_id == "test_interview_001"
    assert reloaded.approved_for_generation is True
    assert reloaded.risk_level == "low"


def test_sales_insight_card_json_schema_has_required_fields():
    schema = SalesInsightCard.model_json_schema()
    assert "source_id" in schema["properties"]
    assert "approved_for_generation" in schema["properties"]
