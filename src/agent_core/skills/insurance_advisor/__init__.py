"""Insurance advisor domain skill."""

# 文件说明：
# - 本文件属于 Domain Skill 层，当前服务保险顾问业务场景。
# - 业务 Skill 只写业务逻辑，不拥有通用工具、Memory、Trace、Recovery。
"""保险顾问领域代码服务。"""

from agent_core.skills.insurance_advisor.kyc import InsuranceKycExtractor
from agent_core.skills.insurance_advisor.knowledge import InsuranceKnowledgeProvider

__all__ = ["InsuranceKnowledgeProvider", "InsuranceKycExtractor"]
