"""Tool execution adapter.

# 文件说明：
# - 本文件属于工具系统，负责把 ToolSpec 映射到本地 capability adapter 并执行。
# - 这里不做自由函数调用，只允许执行 CAPABILITY_RUNNERS 白名单里的工具。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agent_core.capabilities import (
    calculator,
    file_parser,
    knowledge_search,
    news_search,
    summarizer,
    time_date,
    translation,
    unit_converter,
    weather,
    web_page_reader,
    web_search,
)
from agent_core.tools.schemas import ToolCall, ToolResult


# 工具执行白名单：ToolCall.name 只能映射到这里列出的本地 capability adapter。
# 这层设计用来阻断“模型随便编一个 Python 函数名然后执行”的风险。
CAPABILITY_RUNNERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    # calculator 只处理受限算术表达式，用于低风险本地计算。
    "calculator": calculator.run,
    # file_parser 只解析允许的上传文件，不允许任意读本地路径。
    "file_parser": file_parser.run,
    # knowledge_search 代表内部知识库检索能力。
    "knowledge_search": knowledge_search.run,
    # news_search 代表新闻检索能力；未配置 provider 时返回可解释降级信息。
    "news_search": news_search.run,
    # summarizer 代表文本摘要能力。
    "summarizer": summarizer.run,
    # time_query 读取本地时间，不产生外部副作用。
    "time_query": time_date.run,
    # translation 代表翻译能力，生产可接真实模型或翻译服务。
    "translation": translation.run,
    # unit_converter 代表单位换算能力。
    "unit_converter": unit_converter.run,
    # weather_query 代表天气查询能力；本地 demo 不调用真实天气 API。
    "weather_query": weather.run,
    # web_page_reader 代表网页读取能力；生产需配 URL 白名单和网络 provider。
    "web_page_reader": web_page_reader.run,
    # web_search 代表公开网页搜索能力；生产需配合搜索 provider。
    "web_search": web_search.run,
}


def execute_tool_call(call: ToolCall) -> ToolResult:
    """执行一个白名单工具调用，并返回结构化 ToolResult。

    设计约束：
    1. 不允许根据模型输出动态 import 任意模块；
    2. 所有工具都必须先注册到 CAPABILITY_RUNNERS；
    3. 异常被包装为 ToolResult(status="error")，交给 recovery/verification 节点处理。
    """
    # 记录工具开始时间，用于返回 latency_ms，方便观测工具耗时。
    started_at = time.perf_counter()
    # 只从白名单里取 runner；如果工具名不在白名单，就不会执行任何动态代码。
    runner = CAPABILITY_RUNNERS.get(call.name)
    # 找不到 runner 说明工具规划产生了未注册工具，直接返回结构化 error。
    if runner is None:
        return ToolResult(
            name=call.name,
            status="error",
            error=f"tool runner not registered: {call.name}",
            latency_ms=0,
        )
    # 工具内部可能失败，所以统一包在 try/except 中，避免异常炸穿整个 Agent 主链路。
    try:
        # capability adapter 只接收 arguments 字典，不接收完整 AgentState，降低越权读取状态的风险。
        output = runner(call.arguments)
        # 成功时返回标准 ToolResult，后续 verify/grounding/response_package 都消费这个结构。
        return ToolResult(
            name=call.name,
            status="success",
            output=output,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    # 任意工具异常都降级为 ToolResult(status="error")，由 verify_tool_result 决定恢复策略。
    except Exception as exc:
        return ToolResult(
            name=call.name,
            status="error",
            error=str(exc),
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
