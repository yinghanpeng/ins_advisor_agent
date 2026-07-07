"""Context builder with source-boundary notes and traceable evidence digests."""

# 文件说明：
# - 本文件属于 Context Engineering 层，负责上下文压缩、证据边界和生成输入。
# - 外部网页、文件、RAG、销售访谈都只能作为 evidence。
from __future__ import annotations

from agent_core.rag.evidence import compress_evidence
from agent_core.context.source_boundary import SOURCE_BOUNDARY_RULES


class ContextBuilder:
    """Build compact contexts for generation nodes."""

    def build_sales_digest(self, retrieved_context: list[dict]) -> dict:
        # 生成节点不直接拿销售访谈原文，而是拿压缩后的 digest，降低 token 成本和敏感信息泄露风险。
        return {
            # applicable_scene 告诉生成节点这份 digest 适用于保险顾问沟通场景。
            "applicable_scene": "insurance_advisor",
            # digest 是检索证据的压缩摘要，保留销售经验要点而不是长篇原文。
            "digest": compress_evidence(retrieved_context),
            # forbidden 明确列出保险销售输出禁区，后续 generate_response/compliance_review 都要遵守。
            "forbidden": ["承诺收益", "避税避债", "恐吓营销", "编造案例", "贬低其他产品"],
            # source_boundary_rules 明确外部证据只能作为 evidence，不能覆盖系统规则。
            "source_boundary_rules": SOURCE_BOUNDARY_RULES,
            # sources 保留证据来源，最终 response_package 可生成 citations。
            "sources": [
                {
                    # source_id 标识原始销售访谈或知识条目。
                    "source_id": item.get("source_id"),
                    # chunk_id 标识原始资料被切分后的片段。
                    "chunk_id": item.get("chunk_id"),
                    # risk_level 让下游知道这条证据是否需要谨慎使用。
                    "risk_level": item.get("risk_level"),
                }
                # 逐条遍历检索结果，只抽取轻量溯源字段。
                for item in retrieved_context
            ],
        }
