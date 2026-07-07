"""Sales Intelligence Layer."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from agent_core.sales_intelligence.schemas import SalesInsightCard

__all__ = ["SalesInsightCard"]

