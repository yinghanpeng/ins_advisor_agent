"""General capability registry."""

# 文件说明：
# - 本文件属于通用能力层，封装天气、搜索、计算、文件解析等可复用能力。
# - 需要外部服务的能力以 adapter 形式保留，便于生产替换 provider。
from agent_core.tools.registry import ToolRegistry


def build_general_capability_registry() -> ToolRegistry:
    return ToolRegistry.with_defaults()

