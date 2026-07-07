"""Insurance Advisor skill workflow."""

# 文件说明：
# - 本文件属于 Domain Skill 层，当前服务保险顾问业务场景。
# - 业务 Skill 只写业务逻辑，不拥有通用工具、Memory、Trace、Recovery。
from __future__ import annotations

from agent_core.context.builder import ContextBuilder
from agent_core.guardrails.output import OutputGuardrail
from agent_core.sales_intelligence.retriever import SalesIntelligenceRetriever


class InsuranceAdvisorWorkflow:
    """Domain workflow that consumes Sales Intelligence through a clear boundary."""

    def __init__(self, retriever: SalesIntelligenceRetriever | None = None) -> None:
        """初始化保险顾问工作流依赖的检索器、上下文构建器和输出风控。"""
        self.retriever = retriever or SalesIntelligenceRetriever()
        self.context_builder = ContextBuilder()
        self.output_guardrail = OutputGuardrail()

    def break_ice_assistant_workflow(self, user_input: str) -> dict:
        """执行保险顾问破冰工作流：检索销售洞察、构建摘要、生成合规话术。"""
        cards = self.retriever.retrieve(user_input)
        digest = self.context_builder.build_sales_digest([card.model_dump() for card in cards])
        answer = (
            "当前适合先做破冰和资金用途确认。"
            "可以先从客户的经营/家庭责任聊起，再轻轻问："
            "这笔钱更偏企业备用，还是家庭长期不能动的钱？"
        )
        guardrail = self.output_guardrail.review(answer)
        return {
            "workflow": "break_ice_assistant_workflow",
            "sales_insight_digest": digest,
            "answer": answer if guardrail["action"] == "pass" else "需要人工确认后继续。",
            "guardrail": guardrail,
        }
