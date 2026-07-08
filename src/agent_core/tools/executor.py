"""Tool execution adapter.

# 文件说明：
# - 本文件属于工具系统，负责把 ToolSpec 映射到本地 capability adapter 并执行。
# - 这里不做自由函数调用，只允许执行 CAPABILITY_RUNNERS 白名单里的工具。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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
from agent_core.guardrails.tool_guardrails import ToolGuardrail
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.sanitizer import sanitize_tool_output
from agent_core.tools.schemas import ToolCall, ToolResult, ToolSpec
from agent_core.tools.verifier import ToolResultVerifier


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


DEFAULT_TOOL_REGISTRY = ToolRegistry.with_defaults()


def execute_tool_call(call: ToolCall, spec: ToolSpec | None = None) -> ToolResult:
    """执行一个白名单工具调用，并返回结构化 ToolResult。

    设计约束：
    1. 不允许根据模型输出动态 import 任意模块；
    2. 所有工具都必须先注册到 CAPABILITY_RUNNERS；
    3. 异常被包装为 ToolResult(status="error")，交给 recovery/verification 节点处理。
    """
    started_at = time.perf_counter()
    spec = spec or DEFAULT_TOOL_REGISTRY.get(call.name)
    if spec is None:
        return ToolResult(
            name=call.name,
            status="error",
            error=f"tool spec not registered: {call.name}",
            latency_ms=0,
        )
    guardrail_result = ToolGuardrail().review(spec)
    if guardrail_result.get("triggered"):
        return ToolResult(
            name=call.name,
            status="blocked",
            error=guardrail_result.get("reason", "tool permission denied"),
            latency_ms=0,
        )
    if spec.side_effect_level in {"write", "external_action", "financial"} or spec.requires_approval:
        return ToolResult(
            name=call.name,
            status="blocked",
            error="tool requires human approval before execution",
            latency_ms=0,
        )

    runner = CAPABILITY_RUNNERS.get(call.name)
    if runner is None:
        return ToolResult(
            name=call.name,
            status="error",
            error=f"tool runner not registered: {call.name}",
            latency_ms=0,
        )

    max_attempts = int(spec.retry_policy.get("max_attempts", 1))
    max_attempts = max(1, max_attempts if spec.retryable else 1)
    backoff_ms = int(spec.retry_policy.get("backoff_ms", 200))
    last_error: str | None = None
    verifier = ToolResultVerifier()

    for attempt in range(max_attempts):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(runner, call.arguments)
                raw_output = future.result(timeout=spec.timeout_ms / 1000)
            if not isinstance(raw_output, dict):
                raw_output = {"value": raw_output}
            sanitized = sanitize_tool_output(call.name, raw_output)
            verification = verifier.verify(spec, sanitized.output)
            if not verification.ok:
                return ToolResult(
                    name=call.name,
                    status="error",
                    output={"safety_flags": sanitized.safety_flags},
                    error="; ".join(verification.errors),
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    retry_count=attempt,
                )
            return ToolResult(
                name=call.name,
                status="success",
                output={
                    **sanitized.output,
                    "_safety_flags": sanitized.safety_flags,
                    "_removed_fragments": sanitized.removed_fragments,
                },
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                retry_count=attempt,
            )
        except TimeoutError:
            last_error = f"tool timeout after {spec.timeout_ms}ms"
        except Exception as exc:
            last_error = str(exc)
        if attempt < max_attempts - 1:
            time.sleep(backoff_ms / 1000)

    return ToolResult(
        name=call.name,
        status="error",
        error=last_error or "tool execution failed",
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        retry_count=max_attempts - 1,
    )
