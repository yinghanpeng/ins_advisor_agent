"""Tool router."""

# 文件说明：
# - 本文件属于工具系统，负责工具 schema、权限、注册或路由。
# - 工具调用必须有风险等级、权限 scope、超时、重试和错误结构。
from __future__ import annotations

from agent_core.tools.registry import ToolRegistry
from agent_core.tools.schemas import ToolSpec


class ToolRouter:
    """Select a tool from a natural language request."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        """初始化工具注册表；测试可注入自定义 registry。"""
        # 允许外部注入 registry；测试可以替换工具集合，生产可以接入企业工具目录。
        self.registry = registry or ToolRegistry.with_defaults()

    def route(self, text: str) -> ToolSpec | None:
        """按用户输入选择一个低风险优先的工具规格。"""
        # 转小写后做关键词匹配，保证英文 weather/news/search 等大小写不影响路由。
        lower = text.lower()
        # 天气问题路由到 weather_query，后续 _build_tool_arguments 会抽取地点。
        if any(token in lower for token in ["天气", "weather"]):
            # 返回注册表中的规格而非执行函数，使权限与 Schema 校验仍可统一执行。
            return self.registry.get("weather_query")
        # 时间/日期问题路由到 time_query，不需要外部网络。
        if any(token in lower for token in ["几点", "日期", "time", "date"]):
            # 返回本地只读时间工具规格，不在路由阶段直接产生结果。
            return self.registry.get("time_query")
        # 数学表达式或“计算”关键词路由到 calculator，避免模型心算造成错误。
        if any(token in lower for token in ["计算", "calculator"]) or any(op in text for op in ["+", "-", "*", "/"]):
            # 返回受限计算器规格，后续仍需从输入构造并校验 expression 参数。
            return self.registry.get("calculator")
        # “新闻/最新/热点/news”优先路由到 news_search，适合有时效性的报道查询。
        if any(token in lower for token in ["新闻", "最新", "热点", "news"]):
            # 返回新闻工具规格，让执行器按中风险互联网只读策略审查。
            return self.registry.get("news_search")
        # “搜索/查一下/search”路由到通用 web_search。
        if any(token in lower for token in ["搜索", "查一下", "search"]):
            # 返回通用搜索规格，外部内容将在执行后经过来源边界清洗。
            return self.registry.get("web_search")
        # 无法识别具体工具时，默认走低风险 summarizer，避免误调中高风险网络工具。
        return self.registry.get("summarizer")
