"""Tool registry."""

# 文件说明：
# - 本文件属于工具系统，负责工具 schema、权限、注册或路由。
# - 工具调用必须有风险等级、权限 scope、超时、重试和错误结构。
from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.tools.schemas import ToolPermissionSpec, ToolSpec


def default_tool_specs() -> list[ToolSpec]:
    """返回本地默认工具清单，每个工具都带 schema、权限和风险元数据。"""
    # 本地 demo 使用宽松 JSON Schema；生产环境应替换成每个工具的严格参数 schema。
    base_input = {"type": "object", "additionalProperties": True}
    # 输出 schema 同样先保持宽松，后续可按工具收敛到结构化返回。
    base_output = {"type": "object", "additionalProperties": True}
    # 下面每个 ToolSpec 都显式声明风险等级、权限 scope 和执行策略，供 ToolGuardrail 审查。
    return [
        # web_search 读取公开互联网，属于中风险 tenant 级只读工具。
        ToolSpec(
            name="web_search",
            description="Search the public web through an approved provider.",
            input_schema=base_input,
            output_schema=base_output,
            risk_level="medium",
            permission=ToolPermissionSpec(level="tenant", scope="internet.read"),
            permission_scope="internet.read",
        ),
        # web_page_reader 读取指定 URL，风险同样来自互联网内容和 prompt injection。
        ToolSpec(
            name="web_page_reader",
            description="Read and summarize a specific URL.",
            input_schema=base_input,
            output_schema=base_output,
            risk_level="medium",
            permission=ToolPermissionSpec(level="tenant", scope="internet.read"),
            permission_scope="internet.read",
        ),
        # weather_query 是只读天气工具，风险较低但仍按 tenant scope 记录。
        ToolSpec(
            name="weather_query",
            description="Query weather for a location.",
            input_schema=base_input,
            output_schema=base_output,
            permission_scope="weather.read",
            permission=ToolPermissionSpec(level="tenant", scope="weather.read"),
        ),
        # time_query 只读取本地时间，不需要重试，超时设置更短。
        ToolSpec(
            name="time_query",
            description="Return current date/time information.",
            input_schema=base_input,
            output_schema=base_output,
            retryable=False,
            timeout_seconds=2,
            permission_scope="local.time",
            permission=ToolPermissionSpec(level="public", scope="local.time"),
        ),
        # calculator 只做本地安全算术，属于 public 低风险能力。
        ToolSpec(
            name="calculator",
            description="Evaluate safe arithmetic expressions.",
            input_schema=base_input,
            output_schema=base_output,
            retryable=False,
            timeout_seconds=2,
            permission_scope="local.compute",
            permission=ToolPermissionSpec(level="public", scope="local.compute"),
        ),
        # unit_converter 是本地单位换算，默认低风险。
        ToolSpec(
            name="unit_converter",
            description="Convert supported units.",
            permission=ToolPermissionSpec(level="public", scope="local.compute"),
        ),
        # file_parser 读取上传文件，必须保持 tenant 边界，避免跨用户文件泄露。
        ToolSpec(
            name="file_parser",
            description="Parse approved uploaded files.",
            risk_level="medium",
            permission=ToolPermissionSpec(level="tenant", scope="files.read"),
            permission_scope="files.read",
        ),
        # knowledge_search 读取内部知识库，属于 tenant 级只读能力。
        ToolSpec(
            name="knowledge_search",
            description="Search internal knowledge indexes.",
            permission=ToolPermissionSpec(level="tenant", scope="knowledge.read"),
            permission_scope="knowledge.read",
        ),
        # news_search 读取近期新闻，事实时效性强，因此标记为 medium 风险。
        ToolSpec(
            name="news_search",
            description="Search recent news.",
            risk_level="medium",
            permission=ToolPermissionSpec(level="tenant", scope="internet.read"),
            permission_scope="internet.read",
        ),
        # translation 是文本变换工具，不访问外部业务系统。
        ToolSpec(
            name="translation",
            description="Translate provided text.",
            permission=ToolPermissionSpec(level="tenant", scope="llm.transform"),
            permission_scope="llm.transform",
        ),
        # summarizer 是低风险兜底工具，可用于无法匹配其他工具时做文本摘要。
        ToolSpec(
            name="summarizer",
            description="Summarize provided text.",
            permission=ToolPermissionSpec(level="tenant", scope="llm.transform"),
            permission_scope="llm.transform",
        ),
    ]


@dataclass
class ToolRegistry:
    """管理工具注册和查找，所有工具路由都通过该对象取 ToolSpec。"""

    # tools 使用工具名作为 key，确保路由和执行时都能 O(1) 查找 ToolSpec。
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    @classmethod
    def with_defaults(cls) -> "ToolRegistry":
        """创建包含默认工具清单的 registry。"""
        # 创建空 registry，再逐个注册默认 ToolSpec。
        registry = cls()
        # 逐个注册默认工具，register 可复用覆盖逻辑。
        for spec in default_tool_specs():
            registry.register(spec)
        # 返回带默认工具集合的 registry，供 ToolRouter 直接使用。
        return registry

    def register(self, spec: ToolSpec) -> None:
        """注册或覆盖一个工具规格。"""
        # 以 spec.name 作为稳定键；同名注册会覆盖旧定义，便于测试替换工具。
        self.tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """按工具名称查找 ToolSpec；不存在时返回 None。"""
        # 找不到时返回 None，让路由或执行层显式处理“未注册工具”。
        return self.tools.get(name)

    def names(self) -> list[str]:
        """返回已注册工具名称，按字母序稳定输出。"""
        # 排序输出保证测试快照和文档生成稳定。
        return sorted(self.tools)
