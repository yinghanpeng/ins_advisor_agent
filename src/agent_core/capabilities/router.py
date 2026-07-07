"""General capability routing facade."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from agent_core.tools.router import ToolRouter


class GeneralCapabilityRouter(ToolRouter):
    """Alias for tool routing at the capability layer."""

