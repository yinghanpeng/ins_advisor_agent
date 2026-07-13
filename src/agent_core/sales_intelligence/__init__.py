"""Sales Intelligence Layer."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from agent_core.sales_intelligence.schemas import SalesInsightCard

# 只公开核心洞察卡片契约，其他离线治理组件由具体模块显式导入。
__all__ = ["SalesInsightCard"]
