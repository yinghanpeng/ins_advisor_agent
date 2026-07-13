"""Compliance review for sales insight cards."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from agent_core.sales_intelligence.schemas import SalesInsightCard


# 高风险词命中后卡片必须同步拒绝生成，不能通过提示词软约束放行。
HIGH_RISK_TERMS = ["保证收益", "避债避税", "绝对安全", "谁都动不了", "一定成交", "制造焦虑"]
# 中风险词同样默认不准入，需要离线内容修订后重新跑自动准入。
MEDIUM_RISK_TERMS = ["资产隔离", "税务安排", "收益率", "港保更安全"]


def review_card(card: SalesInsightCard) -> SalesInsightCard:
    """执行同步静态生成准入；不创建客户请求中的人工审批状态。"""
    # 重点逻辑：把策略、话术、错误方式、合规说明拼在一起统一扫描风险词。
    text = " ".join(
        [
            card.effective_strategy,
            card.usable_script,
            card.wrong_way,
            card.compliance_notes,
        ]
    )
    # 分别保存高/中风险命中，保证高风险优先级始终高于中风险。
    high_hits = [term for term in HIGH_RISK_TERMS if term in text]
    # 中风险命中单独保存，仅在没有高风险时才用于最终分类。
    medium_hits = [term for term in MEDIUM_RISK_TERMS if term in text]
    # 任一高风险表达命中时将卡片标为 high 且禁止在线生成。
    if high_hits:
        # 重点逻辑：高风险卡片必须禁止生成，不能靠下游 Prompt 自觉避开。
        card.risk_level = "high"
        # 高风险同步关闭生成准入，不进入人工审批或等待分支。
        card.approved_for_generation = False
        # 仅记录命中的规则词用于离线治理，不复制整段销售语料。
        card.compliance_notes = f"High risk terms: {','.join(high_hits)}"
    # 没有高风险但命中中风险时标为 medium，并同样保持 default deny。
    elif medium_hits:
        # 中风险卡片由自动准入策略直接拒绝进入生成，不创建人工审批或等待状态。
        card.risk_level = "medium"
        # 中风险同样执行 default deny，修订后需重新跑自动准入。
        card.approved_for_generation = False
        # 写入稳定的自动拒绝原因，便于内容治理定位和重新发布。
        card.compliance_notes = f"Generation gate rejected medium-risk terms: {','.join(medium_hits)}"
    # 高风险和中风险词都未命中时进入低风险分支，但不自动改变发布准入。
    else:
        # 未命中词表时风险降为 low；是否发布仍保留卡片原有准入布尔值。
        card.risk_level = "low"
    # 返回已原地更新风险和准入状态的卡片，供索引链路继续处理。
    return card
