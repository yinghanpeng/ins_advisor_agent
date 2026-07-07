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
        # retriever 允许测试注入；默认使用 SalesIntelligenceRetriever 检索已审核销售洞察卡片。
        self.retriever = retriever or SalesIntelligenceRetriever()
        # context_builder 负责把检索到的卡片压缩成 digest，不把原始访谈直接塞给生成逻辑。
        self.context_builder = ContextBuilder()
        # output_guardrail 在返回前检查保险销售高风险表达。
        self.output_guardrail = OutputGuardrail()

    def break_ice_assistant_workflow(self, user_input: str) -> dict:
        """执行保险顾问破冰工作流：检索销售洞察、构建摘要、生成合规话术。"""
        # 根据用户输入检索销售洞察卡片；这里只拿已审核、低/中风险卡片。
        cards = self.retriever.retrieve(user_input)
        # 将卡片转换成 dict 后交给 ContextBuilder 生成可用于回答的摘要。
        digest = self.context_builder.build_sales_digest([card.model_dump() for card in cards])
        # 本地 deterministic answer 用低压破冰方式示范，不直接推产品或承诺收益。
        answer = (
            "当前适合先做破冰和资金用途确认。"
            "可以先从客户的经营/家庭责任聊起，再轻轻问："
            "这笔钱更偏企业备用，还是家庭长期不能动的钱？"
        )
        # 输出前做合规审查，命中风险时不直接返回原回答。
        guardrail = self.output_guardrail.review(answer)
        # 返回 Skill 层结构化结果；主 Agent 可继续封装 response_package。
        return {
            # workflow 标识当前执行的业务子流程。
            "workflow": "break_ice_assistant_workflow",
            # sales_insight_digest 是证据摘要和来源边界。
            "sales_insight_digest": digest,
            # answer 只有通过合规审查才返回，否则提示人工确认。
            "answer": answer if guardrail["action"] == "pass" else "需要人工确认后继续。",
            # guardrail 保存输出合规审查结果。
            "guardrail": guardrail,
        }
