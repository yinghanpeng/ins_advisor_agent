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
from agent_core.tools.verifier import ToolInputValidator, ToolResultVerifier


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


# 默认 Registry 只包含代码注册的白名单工具，执行器不会从请求动态扩展。
DEFAULT_TOOL_REGISTRY = ToolRegistry.with_defaults()


def execute_tool_call(call: ToolCall, spec: ToolSpec | None = None) -> ToolResult:
    """执行一个白名单工具调用，并返回结构化 ToolResult。

    设计约束：
    1. 不允许根据模型输出动态 import 任意模块；
    2. 所有工具都必须先注册到 CAPABILITY_RUNNERS；
    3. 异常被包装为 ToolResult(status="error")，交给 recovery/verification 节点处理。
    """
    # 单调时钟用于计算工具总耗时，不受系统时间调整影响。
    started_at = time.perf_counter()
    # 调用方显式传入规格优先；否则从固定 Registry 按工具名解析。
    spec = spec or DEFAULT_TOOL_REGISTRY.get(call.name)
    # 未注册规格立即返回结构化错误，不能继续查找或动态导入 Runner。
    if spec is None:
        # 以 error 状态返回且耗时记零，明确失败发生在真正执行之前。
        return ToolResult(
            name=call.name,
            status="error",
            error=f"tool spec not registered: {call.name}",
            latency_ms=0,
        )
    # 在参数与 Runner 处理前执行权限/副作用 Guardrail。
    guardrail_result = ToolGuardrail().review(spec)
    # Guardrail 触发时同步阻断，不创建审批或等待状态。
    if guardrail_result.get("triggered"):
        # blocked 与运行错误分离，便于上层向客户解释为能力边界而非系统故障。
        return ToolResult(
            name=call.name,
            status="blocked",
            error=guardrail_result.get("reason", "tool permission denied"),
            latency_ms=0,
        )
    # 执行器再次硬阻断所有副作用级别，防止错误配置绕过 ToolGuardrail。
    if spec.side_effect or spec.side_effect_level in {"write", "external_action", "financial"}:
        # 使用稳定英文错误供 API 与评测断言，且绝不进入人工审批分支。
        return ToolResult(
            name=call.name,
            status="blocked",
            error="side-effecting tools are not available in this customer-facing system",
            latency_ms=0,
        )

    # Tool Schema 是参数的唯一契约；即使调用方绕过 routing，执行器也会二次校验。
    input_validation = ToolInputValidator().validate(spec, call.arguments)
    # 参数错误返回稳定 error，不调用 Runner，也不尝试猜测缺失值。
    if not input_validation.ok:
        # 优先使用具体类型/枚举错误；仅缺字段时生成统一缺参说明。
        details = input_validation.errors or [
            f"工具入参缺少必需字段：{field_name}"
            for field_name in input_validation.missing_fields
        ]
        # 把全部参数问题合并成一次结构化错误，避免无效 Runner 调用。
        return ToolResult(
            name=call.name,
            status="error",
            error="; ".join(details),
            latency_ms=0,
        )

    # ToolSpec 通过后仍需命中本地 Runner 白名单，规格存在不等于执行函数存在。
    runner = CAPABILITY_RUNNERS.get(call.name)
    # Runner 缺失属于部署错误，返回结构化失败而非执行任意同名函数。
    if runner is None:
        # 返回错误并保持零延迟，表明执行函数白名单在部署时未完成注册。
        return ToolResult(
            name=call.name,
            status="error",
            error=f"tool runner not registered: {call.name}",
            latency_ms=0,
        )

    # 最大尝试次数来自 ToolSpec；不可重试工具强制为一次。
    max_attempts = int(spec.retry_policy.get("max_attempts", 1))
    # 下限固定为一，防止错误配置导致工具完全不执行且无结果。
    max_attempts = max(1, max_attempts if spec.retryable else 1)
    # 重试等待同样来自规格，业务代码不硬编码不同工具的退避时间。
    backoff_ms = int(spec.retry_policy.get("backoff_ms", 200))
    # 保存最后一次异常摘要，全部尝试失败后写入最终 ToolResult。
    last_error: str | None = None
    # 同一个 Verifier 复用当前工具的所有尝试，校验口径保持一致。
    verifier = ToolResultVerifier()

    # 在明确预算内逐次执行；每轮都创建独立线程池以支持请求级超时返回。
    for attempt in range(max_attempts):
        # 不能使用 ThreadPoolExecutor 的 with 语句：超时后 __exit__ 会 wait=True，再次阻塞到任务结束。
        pool = ThreadPoolExecutor(max_workers=1)
        # future 在提交前保持 None，Timeout 分支据此判断是否存在可取消任务。
        future = None
        # Runner、Sanitizer 与 Verifier 共同位于异常边界，任何异常都变成 ToolResult。
        try:
            # 在线程池提交白名单 Runner，使同步函数也能受请求级超时约束。
            future = pool.submit(runner, call.arguments)
            # 按 ToolSpec 毫秒预算等待结果，超时会进入独立恢复分支。
            raw_output = future.result(timeout=spec.timeout_ms / 1000)
            # 非对象返回包装为 value，随后仍要经过 Sanitizer 与 output_schema。
            if not isinstance(raw_output, dict):
                # 用 value 键包装标量，确保清洗器和结果校验器接收统一对象结构。
                raw_output = {"value": raw_output}
            # 清除外部注入与 PII，并附加 untrusted source boundary。
            sanitized = sanitize_tool_output(call.name, raw_output)
            # 校验清洗后的结构和来源边界，失败结果不能进入生成上下文。
            verification = verifier.verify(spec, sanitized.output)
            # 输出结构或 source boundary 不合格时立即失败，禁止进入生成上下文。
            if not verification.ok:
                # 返回校验错误和安全标记，不透传不合格的原始工具正文。
                return ToolResult(
                    name=call.name,
                    status="error",
                    output={"safety_flags": sanitized.safety_flags},
                    error="; ".join(verification.errors),
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    retry_count=attempt,
                )
            # 校验通过后返回清洗输出，并附加安全事件供后续合规和 trace 使用。
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
        # 超时与普通异常分开处理，便于返回稳定 timeout 原因并决定是否重试。
        except TimeoutError:
            # Python 线程无法强制终止正在运行的函数；cancel 阻止尚未开始的任务，shutdown(wait=False)
            # 保证当前客户请求按配置超时返回。外部 HTTP Runner 仍必须设置自身网络超时。
            # Future 已提交时尝试取消尚未开始的任务；运行中线程由 wait=False 脱离当前请求。
            if future is not None:
                # cancel 只能取消尚未开始的任务；运行中任务由 wait=False 与请求解耦。
                future.cancel()
            # 保存稳定超时摘要，循环结束后写入最终 ToolResult 或进入下一次重试。
            last_error = f"tool timeout after {spec.timeout_ms}ms"
        # 其它 Runner 异常只保存字符串摘要，不把堆栈或内部对象返回客户。
        except Exception as exc:
            # 仅保留异常文本供内部恢复，不向客户暴露堆栈和实现对象。
            last_error = str(exc)
        # 无论前序逻辑成功或失败都执行资源清理，避免连接或执行器泄漏。
        finally:
            # cancel_futures 清理队列中的未执行任务；wait=False 避免把 timeout 重新变成阻塞等待。
            pool.shutdown(wait=False, cancel_futures=True)
        # 仅在仍有剩余尝试时等待退避，最后一次失败直接返回。
        if attempt < max_attempts - 1:
            # 仅在确有下一次尝试时按配置退避，避免最后一次失败额外延迟响应。
            time.sleep(backoff_ms / 1000)

    # 所有尝试都失败后返回最终结构化错误和实际总耗时。
    return ToolResult(
        name=call.name,
        status="error",
        error=last_error or "tool execution failed",
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        retry_count=max_attempts - 1,
    )
