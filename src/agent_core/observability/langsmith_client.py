"""LangSmith 远程 Run Tree 适配器。

本模块支持仅控制面和完整业务内容两种策略。完整模式可上传客户原文、KYC、Prompt、模型响应、
工具与知识正文，但任何模式都强制递归清除 API Key、密码、Cookie、Token 等认证凭据。
LangSmith 不可用时，所有方法都按可观测性降级处理，不影响 Agent 主业务结果。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

from agent_core.observability.logger import StructuredLogger
from agent_core.utils.time import utc_now_iso


# REMOTE_TRACE_SAFE_FIELDS 是允许离开本机的控制面字段白名单，任何正文类字段默认拒绝。
REMOTE_TRACE_SAFE_FIELDS = frozenset(
    {
        "trace_id",
        "workflow_name",
        "domain_skill",
        "node_name",
        "from_state",
        "to_state",
        "reason",
        "intent",
        "route",
        "risk_level",
        "decision_action",
        "status",
        "action",
        "tool_name",
        "attempt",
        "confidence",
        "count",
        "error_count",
        "final_state",
        "response_ready",
        "fallback",
        "fields",
        "keys",
        "step_index",
        "step_name",
        "step_code",
        "trace_event_name",
    }
)

# RETRIEVER_STEPS 在 LangSmith 中显示为 retriever Run，便于和普通 Chain 节点区分。
RETRIEVER_STEPS = frozenset(
    {
        "RESTORE_MEMORY",
        "LOAD_BUSINESS_MEMORY",
        "SALES_INSIGHT_RETRIEVAL",
        "RETRIEVE_DIALOGUE_PATTERNS",
        "RETRIEVE_INSURANCE_KNOWLEDGE",
        "RETRIEVE_EXTERNAL_CONTEXT_IF_NEEDED",
        "RETRIEVE_CONTEXT",
    }
)

# TOOL_STEPS 在 LangSmith 中显示为 tool Run，支持工具链耗时与失败过滤。
TOOL_STEPS = frozenset({"GENERAL_TOOL_CALL", "AGENTIC_TOOL_LOOP"})

# MODEL_STEPS 在 LangSmith 中显示为 llm Run；只记录模型节点边界，不上传 Prompt 或回答正文。
MODEL_STEPS = frozenset(
    {
        "CLASSIFY_INTENT",
        "EXTRACT_INSURANCE_KYC",
        "GENERATE_STRATEGY",
        "MODEL_ROUTING",
        "GENERATE_RESPONSE",
        "GENERAL_RESPONSE_GENERATION",
        "REGENERATE_RESPONSE",
        "EVALUATE_RESPONSE_QUALITY",
    }
)

# SAFE_LABEL_PATTERN 只允许稳定的 ASCII 枚举标签，阻止公共 workflow/domain 字段夹带自然语言正文。
SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")

# FULL_CONTENT_POLICY 是显式允许上传业务正文的策略值，避免布尔开关含义不清。
FULL_CONTENT_POLICY = "full_business_content"

# SECRET_FIELD_NAMES 覆盖常见认证 Header、连接密码和令牌字段；这些字段永远不能上传。
SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "set_cookie",
        "password",
        "passwd",
        "secret",
        "token",
        "client_secret",
        "access_token",
        "refresh_token",
        "id_token",
        "credential",
        "credentials",
    }
)

# SECRET_ENV_SUFFIXES 用于发现当前进程已配置的凭据值，并从任意业务字符串中替换掉它们。
SECRET_ENV_SUFFIXES = (
    "_API_KEY",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
    "_CREDENTIAL",
    "_CREDENTIALS",
)

# INLINE_SECRET_PATTERNS 清理正文中常见 Bearer、LangSmith/OpenAI Key 和含密码数据库 URL。
INLINE_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\blsv2_pt_[A-Za-z0-9_-]+\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(postgresql(?:\+\w+)?|redis)://[^\s/@:]+:[^\s/@]+@"),
)


@dataclass
class _RemoteTraceContext:
    """保存一个本地 Trace 对应的 LangSmith 根 Run 和当前步骤 Run。"""

    # root 是一次 Agent 请求在 LangSmith 中的顶层 chain Run。
    root: Any
    # active_step 是当前正在执行的节点子 Run，下一次状态迁移时完成。
    active_step: Any | None = None
    # active_step_code 保存当前内部节点码，便于失败收尾和测试断言。
    active_step_code: str | None = None
    # active_step_event_count 统计当前节点收到的安全 Trace Event 数量。
    active_step_event_count: int = 0
    # total_event_count 统计整个根 Run 收到的安全 Trace Event 数量。
    total_event_count: int = 0
    # latest_state_snapshot 保存完整模式下当前节点最近一次状态，用作节点结束输出。
    latest_state_snapshot: dict[str, Any] | None = None
    # active_model_run 保存当前真实 Chat Completion 的嵌套 LLM Run。
    active_model_run: Any | None = None
    # active_model_name 保存供应商实际模型名，供无 usage 的异常路径收尾。
    active_model_name: str | None = None


def _stable_reference(value: str) -> str:
    """把租户或 Session 标识转换成不可逆短引用，避免上传原始业务标识。"""

    # SHA-256 只用于不可逆日志引用，不承担密码存储或认证用途。
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    # 十六位前缀足以在单项目内关联排障，同时减少不必要标识长度。
    return digest[:16]


def _bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    """安全读取有上下界的浮点环境变量，非法值回退默认值。"""

    # 读取原始字符串；未配置时直接使用调用方给出的稳定默认值。
    raw_value = os.getenv(name, str(default))
    # LangSmith 属于可选观测层，错误数字不能导致 Agent Runtime 启动失败。
    try:
        # 把部署平台注入的字符串转换成浮点数，供采样或关闭等待逻辑使用。
        parsed_value = float(raw_value)
    # 非数字、空字符串或其它格式错误统一退回安全默认值。
    except (TypeError, ValueError):
        # 返回默认值维持可预测行为，且不把可能含敏感内容的原值写日志。
        return default
    # 把合法数字限制在调用方声明的区间内，避免负超时或异常采样率进入 SDK。
    return min(maximum, max(minimum, parsed_value))


def _safe_label(value: str | None, fallback: str) -> str:
    """只保留满足枚举格式的短标签，否则返回固定替代值。"""

    # None 和空值没有可观测标签意义，直接使用不会泄露正文的固定回退值。
    if not value:
        # fallback 由代码常量提供，供 LangSmith UI 稳定筛选。
        return fallback
    # 仅允许 ASCII 字母、数字和有限分隔符，中文句子、空格和控制符不会进入标签字段。
    if SAFE_LABEL_PATTERN.fullmatch(value):
        # 返回已完整匹配且长度不超过正则上限的标签。
        return value
    # 非法公共标签统一归类为 custom，不截取或记录原值。
    return fallback


def _is_secret_field(key: str) -> bool:
    """判断结构化字段名是否表示认证凭据，而不是普通 token 计数。"""

    # 统一小写并把连字符转成下划线，兼容 HTTP Header 与 JSON 常见命名风格。
    normalized = key.strip().lower().replace("-", "_")
    # 精确命中认证字段时必须整值替换；input_tokens 等计数字段不会命中。
    if normalized in SECRET_FIELD_NAMES:
        # 返回 True 通知递归投影器不要读取或转换原始值。
        return True
    # 以 api_key、password 或 client_secret 结尾的供应商自定义字段同样视为凭据。
    return normalized.endswith(("_api_key", "_password", "_client_secret"))


def _configured_secret_values() -> tuple[str, ...]:
    """收集当前进程环境中的非空凭据值，只用于正文替换。"""

    # values 使用集合去重，避免同一个企业 Key 被多个兼容环境变量重复扫描。
    values: set[str] = set()
    # 遍历当前进程环境；只读取名称表现为凭据的变量，不收集普通业务配置。
    for name, value in os.environ.items():
        # 过短值容易误伤普通文本，凭据至少要求六个字符且变量名符合固定后缀。
        if len(value) >= 6 and name.upper().endswith(SECRET_ENV_SUFFIXES):
            # 仅把值保存在当前函数内存中用于替换，任何日志和返回对象都不会包含它。
            values.add(value)
    # 按长度倒序替换，防止较短凭据是较长凭据子串时留下残片。
    return tuple(sorted(values, key=len, reverse=True))


def _redact_secret_text(value: str, *, max_chars: int) -> str:
    """清除字符串中的已配置凭据和常见内联认证格式，并限制字段体积。"""

    # copied 保存逐轮替换后的副本，原始字符串不会被修改或写入其它对象。
    copied = value
    # 环境中的真实 Key 可能被用户粘贴到 Prompt 或异常文本中，必须按值再次替换。
    for secret_value in _configured_secret_values():
        # exact replace 覆盖凭据出现在 JSON、自然语言或 URL 查询参数中的情况。
        copied = copied.replace(secret_value, "[REDACTED_CREDENTIAL]")
    # 正则用于兜底处理不在当前环境中的 Bearer、sk/lsv2 Key 和带密码连接串。
    for pattern in INLINE_SECRET_PATTERNS:
        # 数据库 URL 保留协议类别，其余凭据统一替换成固定标记。
        copied = pattern.sub("[REDACTED_CREDENTIAL]", copied)
    # 单字段达到上限时保留前缀并标明截断，避免第三方 Trace 请求无限膨胀。
    if len(copied) > max_chars:
        # 截断标记明确告诉排障人员远端不是完整存储，不应误判为模型原始输出。
        return f"{copied[:max_chars]}…[TRUNCATED]"
    # 未超长时返回完成凭据清理的业务正文。
    return copied


def _full_payload(
    value: Any,
    *,
    max_chars: int,
    max_items: int,
    depth: int = 0,
) -> Any:
    """递归复制完整业务内容，同时强制脱敏凭据并限制异常对象规模。"""

    # 递归深度超过二十层通常表示循环式供应商对象，使用标记阻止栈耗尽。
    if depth > 20:
        # 固定标记不调用对象字符串方法，因此不会意外展开内部凭据。
        return "[MAX_DEPTH_REACHED]"
    # 空值、布尔和数字可直接进入 Trace；token 计数、分数和向量数值都能保留。
    if value is None or isinstance(value, (bool, int, float)):
        # 返回 JSON 原生标量，LangSmith 可直接筛选或渲染。
        return value
    # 所有字符串先执行环境凭据和内联凭据清除，再应用字段长度上限。
    if isinstance(value, str):
        # max_chars 来自受限环境配置，防止单字段超过远端请求限制。
        return _redact_secret_text(value, max_chars=max_chars)
    # Pydantic 对象先投影为 JSON 模式字典，避免上传私有属性或客户端连接对象。
    if hasattr(value, "model_dump"):
        # model_dump 的结果继续经过相同凭据字段检查，不能绕过递归保护层。
        return _full_payload(
            value.model_dump(mode="json"),
            max_chars=max_chars,
            max_items=max_items,
            depth=depth + 1,
        )
    # 字典保留业务字段，但任何凭据字段都只写固定脱敏标记。
    if isinstance(value, dict):
        # result 按原迭代顺序保存前 max_items 项，便于阅读且控制单事件体积。
        result: dict[str, Any] = {}
        # enumerate 同时提供数量边界，避免巨大供应商响应拖慢客户线程。
        for index, (raw_key, raw_item) in enumerate(value.items()):
            # 超出配置上限时增加明确截断字段并停止遍历。
            if index >= max_items:
                # 固定字段告诉控制台仍有多少项未上传，而不是静默丢失。
                result["__truncated_items__"] = len(value) - max_items
                # 已满足远程体积上限，结束当前字典遍历。
                break
            # 键统一转字符串，兼容工具返回整数键等非标准 JSON 对象。
            key = str(raw_key)
            # 字段名命中凭据规则时绝不访问其字符串表示，直接替换整值。
            if _is_secret_field(key):
                # 固定标记可证明脱敏生效，同时不暴露长度、前后缀或凭据类型。
                result[key] = "[REDACTED_CREDENTIAL]"
                # 当前字段已经安全处理，继续投影下一个业务字段。
                continue
            # 非凭据字段递归保留完整业务内容。
            result[key] = _full_payload(
                raw_item,
                max_chars=max_chars,
                max_items=max_items,
                depth=depth + 1,
            )
        # 返回新字典，调用方原始对象不会因脱敏被修改。
        return result
    # 列表与元组保留前 max_items 个元素，并递归清理每个业务对象。
    if isinstance(value, (list, tuple)):
        # projected 保存远程副本，不把潜在可变列表直接交给后台上传线程。
        projected = [
            _full_payload(
                item,
                max_chars=max_chars,
                max_items=max_items,
                depth=depth + 1,
            )
            for item in value[:max_items]
        ]
        # 超出集合上限时追加截断信息，排障人员可以看到本地实际规模。
        if len(value) > max_items:
            # 使用结构化标记而不是字符串拼接原始内容。
            projected.append({"__truncated_items__": len(value) - max_items})
        # 返回完成深复制与凭据清理的集合。
        return projected
    # 未识别对象只记录类型名，不调用可能包含连接串或密钥的 repr/str。
    return f"<{type(value).__name__}>"


def _safe_value(value: Any) -> Any:
    """递归限制远程字段类型、长度和集合规模。"""

    # 空值、布尔、整数和浮点数不携带正文，可直接保留。
    if value is None or isinstance(value, (bool, int, float)):
        # 返回已验证的简单标量，保持数字指标可筛选。
        return value
    # 字符串只允许有限长度，防止异常原因或扩展状态码夹带长正文。
    if isinstance(value, str):
        # 最多保留两百字符，流程名称和原因码都远低于该上限。
        return value[:200]
    # 列表和元组只保留前二十个安全值，避免无限扩张 Trace 体积。
    if isinstance(value, (list, tuple)):
        # 对集合元素继续执行相同的标量和长度限制。
        return [_safe_value(item) for item in value[:20]]
    # 字典只保留允许上传的键，阻断调用方把嵌套正文绕过顶层白名单。
    if isinstance(value, dict):
        # 递归构造安全子对象，未知键不会进入远程 Trace。
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if str(key) in REMOTE_TRACE_SAFE_FIELDS
        }
    # 未识别对象只记录类型名，不调用可能泄露内部数据的 str(value)。
    return f"<{type(value).__name__}>"


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """对单条 Trace Event 执行默认拒绝的远程字段投影。"""

    # 只有白名单键进入结果，客户输入、Prompt、检索正文和模型输出会被直接丢弃。
    return {
        key: _safe_value(value)
        for key, value in payload.items()
        if key in REMOTE_TRACE_SAFE_FIELDS
    }


def _run_type_for_step(step_code: str) -> str:
    """把内部状态节点映射成 LangSmith 支持的 Run 类型。"""

    # 检索节点使用 retriever，LangSmith UI 可按检索类型筛选和聚合耗时。
    if step_code in RETRIEVER_STEPS:
        # 返回 LangSmith 标准 retriever 类型。
        return "retriever"
    # 工具执行与有界工具循环使用 tool 类型。
    if step_code in TOOL_STEPS:
        # 返回 LangSmith 标准 tool 类型。
        return "tool"
    # 模型裁定、抽取、生成和评估节点使用 llm 类型。
    if step_code in MODEL_STEPS:
        # 返回 LangSmith 标准 llm 类型。
        return "llm"
    # 其余编排、风控、记忆写入和收尾节点统一使用 chain。
    return "chain"


@dataclass
class LangSmithAdapter:
    """把一次 Agent 请求安全地投影成 LangSmith 根 Run 与动态节点子 Run。"""

    # enabled 表示用户是否通过环境变量请求启用远程追踪。
    enabled: bool
    # project 是 LangSmith 中接收 Trace 的项目名。
    project: str | None = None
    # endpoint 是官方、区域或自建 LangSmith API 地址。
    endpoint: str | None = None
    # available 表示 SDK、密钥和 Client 均已成功初始化。
    available: bool = False
    # warning 保存初始化阶段的安全降级原因。
    warning: str | None = None
    # client 是启用自动批处理的 LangSmith SDK Client。
    client: Any | None = field(default=None, repr=False)
    # log 始终指向本地结构化日志，远程故障时用于记录降级事件。
    log: StructuredLogger | None = field(default=None, repr=False)
    # run_tree_factory 默认使用 SDK RunTree；测试可以注入无网络替身。
    run_tree_factory: Callable[..., Any] | None = field(default=None, repr=False)
    # flush_timeout_seconds 限制应用关闭时等待后台批处理的最长时间。
    flush_timeout_seconds: float = 5.0
    # data_policy 决定只上传控制面，还是上传经凭据清理后的完整业务内容。
    data_policy: str = "control_plane_only"
    # max_field_chars 限制单个字符串字段体积，防止超长文档使远程请求失控。
    max_field_chars: int = 50_000
    # max_collection_items 限制单个列表或字典项目数，同时保留明确截断标记。
    max_collection_items: int = 500
    # thread_grouping_enabled 控制是否使用 thread_id 把多轮根 Run 聚合为 Threads/Turns。
    thread_grouping_enabled: bool = False
    # _contexts 按本地 trace_id 隔离并发请求的动态 Run Tree。
    _contexts: dict[str, _RemoteTraceContext] = field(default_factory=dict, init=False, repr=False)
    # _lock 保护共享 Engine 下并发请求对 _contexts 的读写与步骤切换。
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def captures_full_content(self) -> bool:
        """返回当前 Adapter 是否显式启用了完整业务内容追踪。"""

        # 只有精确策略值才启用正文上传，拼写错误不会意外扩大数据范围。
        return self.data_policy == FULL_CONTENT_POLICY

    @classmethod
    def from_env(cls, log: StructuredLogger | None = None) -> "LangSmithAdapter":
        """从环境变量创建启用自动批处理和安全错误回调的 LangSmith Client。"""

        # 只有显式 true 才开启远程 Trace，避免开发环境意外上传数据。
        tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        # API Key 只从 Secret 环境变量读取，不写入静态配置或日志。
        api_key = os.getenv("LANGSMITH_API_KEY")
        # 项目名用于隔离不同应用或环境的 Trace。
        project = os.getenv("LANGSMITH_PROJECT", "insurance-advisor-agent")
        # endpoint 支持官方、区域和企业自建地址。
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        # 应用退出时等待批处理的上限从环境读取，并限制为合理正数。
        flush_timeout = _bounded_float_env(
            "LANGSMITH_FLUSH_TIMEOUT_SECONDS",
            5.0,
            0.1,
            60.0,
        )
        # 数据策略默认只传控制面；完整模式必须通过明确字符串显式开启。
        requested_policy = os.getenv("LANGSMITH_DATA_POLICY", "control_plane_only").strip().lower()
        # 未知策略回退到最小数据范围，防止配置拼写错误造成业务正文意外上传。
        data_policy = (
            requested_policy
            if requested_policy in {"control_plane_only", FULL_CONTENT_POLICY}
            else "control_plane_only"
        )
        # 单字符串最多五十万字符，默认五万；非法数字由安全读取函数回退。
        max_field_chars = int(
            _bounded_float_env("LANGSMITH_MAX_FIELD_CHARS", 50_000, 1_000, 500_000)
        )
        # 单集合最多五千项，默认五百；足以覆盖大多数 KYC、检索和工具响应。
        max_collection_items = int(
            _bounded_float_env("LANGSMITH_MAX_COLLECTION_ITEMS", 500, 20, 5_000)
        )
        # Thread 分组默认关闭，保证项目列表点击新 Run 时直接显示 Agent Waterfall 子步骤。
        thread_grouping_enabled = (
            os.getenv("LANGSMITH_THREAD_GROUPING", "false").strip().lower() == "true"
        )
        # 未启用时返回明确的 disabled Adapter，所有运行方法都是安全 no-op。
        if not tracing:
            # 保留项目和 endpoint 便于健康诊断，但不创建 SDK Client。
            return cls(
                enabled=False,
                project=project,
                endpoint=endpoint,
                available=False,
                log=log,
                flush_timeout_seconds=flush_timeout,
                data_policy=data_policy,
                max_field_chars=max_field_chars,
                max_collection_items=max_collection_items,
                thread_grouping_enabled=thread_grouping_enabled,
            )
        # 用户开启远程追踪却没有 Key 时降级并写 warning，不阻断 Agent 启动。
        if not api_key:
            # 使用固定原因码，避免把任何环境变量值写进日志。
            warning = "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing"
            # 有本地 Logger 时输出一次初始化降级日志。
            if log is not None:
                # warning 只包含固定配置原因，不包含 Secret。
                log.warning("langsmith_degraded", reason=warning)
            # 返回不可用 Adapter，主链路继续使用本地 Trace。
            return cls(
                enabled=True,
                project=project,
                endpoint=endpoint,
                available=False,
                warning=warning,
                log=log,
                flush_timeout_seconds=flush_timeout,
                data_policy=data_policy,
                max_field_chars=max_field_chars,
                max_collection_items=max_collection_items,
                thread_grouping_enabled=thread_grouping_enabled,
            )
        # SDK 导入和 Client 构造都放在异常边界内，依赖故障不能阻断 Agent。
        try:
            # Client 启用批处理后台上传，避免每个节点同步等待远程网络。
            from langsmith import Client
            # RunTree 提供显式父子 Run 管理，适配本项目动态状态机分支。
            from langsmith.run_trees import RunTree

            # 异步批处理写失败时只记录异常类型，不记录可能含请求详情的异常文本。
            def tracing_error_callback(exc: Exception) -> None:
                """记录 LangSmith 后台写入失败，不抛回业务线程。"""

                # 只有配置了本地 Logger 才输出远程写入告警。
                if log is not None:
                    # 异常类型足以区分连接、认证和协议类别，避免异常正文泄露。
                    log.warning(
                        "langsmith_async_write_failed",
                        exception_type=type(exc).__name__,
                    )

            # 多 Workspace Key 可通过可选 Workspace ID 明确指定目标。
            workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID") or None
            # 采样率默认 1.0；生产可按成本与合规要求调低，但不能超出零到一范围。
            sampling_rate = _bounded_float_env(
                "LANGSMITH_SAMPLING_RATE",
                1.0,
                0.0,
                1.0,
            )
            # 构造启用自动批处理的 Client，并禁止上传本机运行时信息。
            client = Client(
                api_url=endpoint,
                api_key=api_key,
                auto_batch_tracing=True,
                omit_traced_runtime_info=True,
                tracing_sampling_rate=sampling_rate,
                workspace_id=workspace_id,
                tracing_error_callback=tracing_error_callback,
            )
        # 任意 SDK 初始化错误都转成本地可观测降级，不让远程平台成为业务硬依赖。
        except Exception as exc:
            # warning 只保留异常类型，避免 Client 错误信息中包含 endpoint 或认证细节。
            warning = f"langsmith initialization failed: {type(exc).__name__}"
            # 有本地 Logger 时记录固定降级摘要。
            if log is not None:
                # 输出初始化失败事件，供部署平台告警。
                log.warning("langsmith_degraded", reason=warning)
            # 返回不可用 Adapter，后续调用全部安全跳过。
            return cls(
                enabled=True,
                project=project,
                endpoint=endpoint,
                available=False,
                warning=warning,
                log=log,
                flush_timeout_seconds=flush_timeout,
                data_policy=data_policy,
                max_field_chars=max_field_chars,
                max_collection_items=max_collection_items,
                thread_grouping_enabled=thread_grouping_enabled,
            )
        # Client 与 RunTree 均就绪后返回可用 Adapter。
        return cls(
            enabled=True,
            project=project,
            endpoint=endpoint,
            available=True,
            client=client,
            log=log,
            run_tree_factory=RunTree,
            flush_timeout_seconds=flush_timeout,
            data_policy=data_policy,
            max_field_chars=max_field_chars,
            max_collection_items=max_collection_items,
            thread_grouping_enabled=thread_grouping_enabled,
        )

    def _project_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """按当前数据策略投影远程内容，并在完整模式强制清理凭据。"""

        # 完整模式保留正文、KYC、Prompt、模型和检索内容，但递归凭据保护始终生效。
        if self.captures_full_content:
            # 顶层输入确定为字典，递归结果在此分支仍保持相同结构。
            projected = _full_payload(
                payload,
                max_chars=self.max_field_chars,
                max_items=self.max_collection_items,
            )
            # 防御性确认返回对象类型，异常自定义 Mapping 不得绕过 Adapter 契约。
            return projected if isinstance(projected, dict) else {}
        # 控制面策略沿用默认拒绝白名单，不上传任何业务正文。
        return _safe_payload(payload)

    def start_run(
        self,
        *,
        trace_id: str,
        tenant_id: str,
        session_id: str,
        workflow_name: str,
        app_env: str,
        request_payload: dict[str, Any] | None = None,
    ) -> None:
        """为一次 Agent 请求创建根 Run，并按数据策略投影请求内容。"""

        # 未启用、不可用或缺少 SDK 依赖时直接保持本地日志路径。
        if not self.available or self.client is None or self.run_tree_factory is None:
            # no-op 不产生网络请求，也不改变 Agent 状态。
            return
        # 根 Run 创建和首次提交必须处于异常边界，远程平台不能中断客户请求。
        try:
            # workflow_name 来自公共请求，只允许枚举式 ASCII 标签，防止被用作正文外传通道。
            safe_workflow_name = _safe_label(workflow_name, "custom_workflow")
            # app_env 来自部署配置，同样规范成稳定标签供 Project 过滤。
            safe_app_env = _safe_label(app_env, "unknown_env")
            # 控制面根输入只记录本地 Trace 引用和流程标签，不包含用户原始问题。
            root_inputs: dict[str, Any] = {
                "local_trace_id": trace_id,
                "workflow_name": safe_workflow_name,
                "data_policy": self.data_policy,
            }
            # 完整模式把整个请求契约作为根输入，递归凭据清理后可在 LangSmith 直接回放。
            if self.captures_full_content and request_payload is not None:
                # _project_payload 保留客户原文、主体、metadata 等业务字段，同时强制移除凭据。
                projected_request = self._project_payload(request_payload)
                # input 使用 LangSmith 常见单轮字段名，Thread/Trace UI 会优先展示真正的用户问题。
                root_inputs["input"] = projected_request.get("input")
                # request 继续保存完整契约，避免为了 UI 兼容丢失 session、tenant 或公开 metadata。
                root_inputs["request"] = projected_request
            # 控制面 metadata 默认使用不可逆租户/Session 引用。
            metadata = {
                "local_trace_id": trace_id,
                "tenant_ref": _stable_reference(tenant_id),
                "session_ref": _stable_reference(session_id),
                "app_env": safe_app_env,
                "data_policy": self.data_policy,
                "ls_agent_type": "root",
            }
            # 完整模式保留原始业务标识，但避免用 session_id 这一 LangSmith 保留名意外触发 Thread 视图。
            if self.captures_full_content:
                # business_session_id 仍可搜索关联，但不会改变 Trace 默认展示模式。
                metadata.update(
                    self._project_payload(
                        {
                            "tenant_id": tenant_id,
                            "business_session_id": session_id,
                        }
                    )
                )
            # 只有显式开启 Thread 分组时才写 LangSmith 识别的 thread_id。
            if self.thread_grouping_enabled:
                # 完整模式用原始业务 Session；控制面模式使用稳定哈希，避免扩大标识数据范围。
                thread_value = (
                    session_id if self.captures_full_content else _stable_reference(session_id)
                )
                # thread_id 使相同会话的多个根 Run 聚合到同一个 Turns 页面。
                metadata["thread_id"] = _redact_secret_text(
                    thread_value,
                    max_chars=self.max_field_chars,
                )
            # 创建一次请求的顶层 chain Run，项目名决定 LangSmith 控制台归档位置。
            root = self.run_tree_factory(
                name="Insurance Advisor Agent",
                run_type="chain",
                inputs=root_inputs,
                extra={"metadata": metadata},
                tags=["agent", safe_app_env, safe_workflow_name],
                project_name=self.project or "insurance-advisor-agent",
                ls_client=self.client,
            )
            # post 使用 SDK 自动批处理提交根 Run，不同步等待远程响应。
            root.post()
        # 创建或提交失败时只降级当前 Trace，业务流程继续执行。
        except Exception as exc:
            # 记录不含远程异常文本的安全告警。
            self._warn("langsmith_run_start_failed", exc, trace_id=trace_id)
            # 当前 Trace 没有可用 Root，不写入 contexts。
            return
        # 上下文写入共享字典前取得锁，支持 FastAPI 线程池并发请求。
        with self._lock:
            # trace_id 来自本地 AgentState，作为进程内 Run Tree 查找键。
            self._contexts[trace_id] = _RemoteTraceContext(root=root)
        # 本地记录远程根 Run 已排队，run_id 可用于 LangSmith API 查询但不包含凭据。
        if self.log is not None:
            # 只输出 SDK 生成的随机 Run ID 与项目名。
            self.log.event(
                "langsmith_run_started",
                trace_id=trace_id,
                langsmith_run_id=str(root.id),
                project=self.project,
            )

    def trace_event(self, name: str, payload: dict[str, Any]) -> None:
        """按数据策略追加节点事件，并在状态迁移时切换子 Run。"""

        # Adapter 不可用时保持 no-op，调用方无需重复判断环境配置。
        if not self.available:
            # 不可用状态不创建任何远程对象。
            return
        # trace_id 只用于进程内查找 Root，不依赖远程策略是否保留该字段。
        trace_id = str(payload.get("trace_id") or "")
        # 没有 trace_id 的事件直接跳过，避免生成无法关联的顶层孤儿 Run。
        if not trace_id:
            # 返回给调用方，不抛出可观测性错误。
            return
        # 锁内完成上下文查找、步骤切换和事件追加，防止同一 Trace 并发乱序。
        with self._lock:
            # 查找 start_run 创建的当前请求上下文。
            context = self._contexts.get(trace_id)
            # 根 Run 尚未建立或已经结束时忽略迟到事件。
            if context is None:
                # 迟到事件不应重新创建独立 Trace。
                return
            # SDK RunTree 操作进入异常边界，当前 Trace 失败不能影响 Agent 主链路。
            try:
                # 按策略投影完整正文或最小控制面，两个模式都不允许认证凭据通过。
                remote_payload = self._project_payload(payload)
                # 完整状态快照单独保存为节点前后状态，避免在每条内部事件中重复上传同一大对象。
                state_snapshot = remote_payload.pop("state_snapshot", None)
                # 真实 Chat Completion 开始时创建嵌套 LLM Run，Tokens/Cost/TTFT 必须归属该 Run。
                if name == "model_call_started":
                    # 节点状态仍更新为当前快照，供外层状态 Run 最终写 state_after。
                    if isinstance(state_snapshot, dict):
                        # 保存模型调用前的最新 AgentState。
                        context.latest_state_snapshot = state_snapshot
                    # 创建带标准模型 metadata 的 LLM Run，LangSmith 才能匹配价格表。
                    self._start_model_run(context, remote_payload)
                    # 模型开始事件已经表现为子 Run，无需再作为普通 Event 重复保存。
                    context.total_event_count += 1
                    # 完成本事件处理并等待 finished/failed 配对事件。
                    return
                # 模型成功响应带有标准 usage、模型名和首 Token 时间，用于填充控制台三项指标。
                if name == "model_call_finished":
                    # 响应后的状态快照成为外层节点最近状态。
                    if isinstance(state_snapshot, dict):
                        # 保存模型调用完成时的最新 AgentState。
                        context.latest_state_snapshot = state_snapshot
                    # 正常结束嵌套 LLM Run，并写入 usage_metadata 与 new_token 事件。
                    self._finish_model_run(context, remote_payload, status="completed")
                    # 根事件计数包含模型完成事件。
                    context.total_event_count += 1
                    # 已由模型子 Run 消费，不再追加到外层状态事件。
                    return
                # 模型网络、协议或解析失败必须结束 LLM Run，否则控制台会长期显示 running。
                if name == "model_call_failed":
                    # 失败时仍保留最新 AgentState 供外层恢复节点查看。
                    if isinstance(state_snapshot, dict):
                        # 保存失败发生时的状态快照。
                        context.latest_state_snapshot = state_snapshot
                    # 失败 Run 上传完整脱敏异常详情，但错误字段只使用固定类型。
                    self._finish_model_run(context, remote_payload, status="failed")
                    # 根事件计数包含模型失败事件。
                    context.total_event_count += 1
                    # 已由模型子 Run 消费，结束本事件处理。
                    return
                # 状态迁移表示开始下一个显式节点，需要先完成旧节点并创建新子 Run。
                if name == "state_transition":
                    # 当前迁移发生时的状态代表上一个节点完成后的最新输出。
                    if isinstance(state_snapshot, dict):
                        # 保存快照供 _finish_active_step 写入 state_after。
                        context.latest_state_snapshot = state_snapshot
                    # 进入新节点前关闭当前节点，节点耗时自然覆盖两次状态迁移之间的执行区间。
                    self._finish_active_step(context, status="completed")
                    # 创建本次真实进入的节点子 Run。
                    self._start_step(context, remote_payload, state_snapshot=state_snapshot)
                # 非状态迁移事件挂到当前节点；尚无节点时挂到根 Run。
                else:
                    # 节点内部事件带来的新状态作为该节点最后状态，收尾时写入 state_after。
                    if isinstance(state_snapshot, dict):
                        # 后写覆盖前写，保证节点输出使用最近一次业务状态。
                        context.latest_state_snapshot = state_snapshot
                    # active_step 优先承载节点内部事件，根 Run 仅接收节点外事件。
                    target = context.active_step or context.root
                    # add_event 在内存中积累事件正文，节点 patch 时由 SDK 批量提交。
                    target.add_event(
                        {
                            "name": name,
                            "time": utc_now_iso(),
                            "kwargs": remote_payload,
                        }
                    )
                    # 当前步骤事件计数用于节点输出摘要。
                    context.active_step_event_count += 1
                # 根级总事件数包含状态迁移和节点内部事件。
                context.total_event_count += 1
            # 任意 RunTree 操作错误只关闭当前远程上下文，不影响业务请求。
            except Exception as exc:
                # 从字典移除损坏上下文，避免后续事件重复报错。
                self._contexts.pop(trace_id, None)
                # 记录远程事件写入失败类型。
                self._warn("langsmith_trace_event_failed", exc, trace_id=trace_id)

    def _start_step(
        self,
        context: _RemoteTraceContext,
        payload: dict[str, Any],
        *,
        state_snapshot: Any = None,
    ) -> None:
        """根据状态迁移创建一个动态节点子 Run。"""

        # step_code 优先使用日志层提供值，否则回退到状态迁移的 to_state。
        step_code = str(payload.get("step_code") or payload.get("to_state") or "UNKNOWN")
        # step_name 是终端中文名，缺失时保留内部码，避免远程 Run 无法识别。
        step_name = str(payload.get("step_name") or step_code)
        # step_index 是本轮真实执行顺序；缺失时使用问号而不猜固定编排位置。
        step_index = payload.get("step_index")
        # LangSmith Run 名称带实际序号与中文步骤，列表视图可直接顺序阅读。
        run_name = f"{step_index}. {step_name}" if step_index is not None else step_name
        # 子 Run 输入只包含状态码、原因码和序号，不包含节点业务输入。
        inputs = {
            "step_index": step_index,
            "step_code": step_code,
            "reason": payload.get("reason"),
        }
        # 完整模式把进入节点时的全部 AgentState 记录为 state_before，控制面模式不会传入该快照。
        if self.captures_full_content and isinstance(state_snapshot, dict):
            # 快照已经过 _project_payload 递归凭据清理，可直接交给 RunTree。
            inputs["state_before"] = state_snapshot
        # 根据节点职责选择 chain、llm、tool 或 retriever 类型。
        run_type = _run_type_for_step(step_code)
        # create_child 自动维护 parent_run_id、trace_id 和 dotted_order。
        child = context.root.create_child(
            name=run_name,
            run_type=run_type,
            inputs=inputs,
            tags=["agent-step", step_code.lower()],
            extra={
                "metadata": {
                    "step_index": step_index,
                    "step_code": step_code,
                    "step_name": step_name,
                    "data_policy": self.data_policy,
                }
            },
        )
        # 子 Run 开始事件交给 SDK 自动批处理队列。
        child.post()
        # 保存当前步骤，后续节点内部事件会归入这个子 Run。
        context.active_step = child
        # 保存内部码，失败收尾时可定位最后执行步骤。
        context.active_step_code = step_code
        # 新节点的内部事件计数从零开始。
        context.active_step_event_count = 0
        # 新节点初始状态同时作为当前最新状态，后续内部事件会逐步覆盖。
        context.latest_state_snapshot = state_snapshot if isinstance(state_snapshot, dict) else None

    def _start_model_run(
        self,
        context: _RemoteTraceContext,
        payload: dict[str, Any],
    ) -> None:
        """为一次真实 Chat Completion 创建嵌套 LLM Run。"""

        # 上一次模型 Run 未正常结束时先按失败收尾，避免并发/重入造成悬挂 Run。
        if context.active_model_run is not None:
            # 固定失败事件不含业务正文，仅说明新的模型调用打断了旧 Run。
            self._finish_model_run(
                context,
                {"exception_type": "OverlappingModelCall"},
                status="failed",
            )
        # 请求体来自实际 Chat Completion payload，完整模式保留 messages，控制面模式可能为空。
        model_request = payload.get("model_request")
        # model_name 优先从请求模型字段读取，缺失时使用稳定 unknown 标签。
        model_name = (
            str(model_request.get("model") or "unknown-model")
            if isinstance(model_request, dict)
            else "unknown-model"
        )
        # provider 是客户端协议标识；OpenAI-compatible GPT 模型映射为 openai 以匹配 LangSmith 价格表。
        raw_provider = str(payload.get("provider") or "openai_compatible")
        # GPT/text-embedding 等 OpenAI 模型通过企业兼容网关调用时仍使用 openai 价格目录标识。
        ls_provider = "openai" if model_name.startswith(("gpt-", "o1", "o3", "o4")) else raw_provider
        # 模型 Run 挂在当前状态节点下，控制台可以同时看到业务步骤和真实网络调用。
        parent = context.active_step or context.root
        # 创建标准 llm 类型子 Run，LangSmith 只会为该类型展示 Token/Cost/First Token 指标。
        model_messages = (
            model_request.get("messages", []) if isinstance(model_request, dict) else []
        )
        # LLM Run 输入同时保留便于 UI 渲染的 messages 和完整请求参数。
        model_run = parent.create_child(
            name=f"LLM · {model_name}",
            run_type="llm",
            inputs={"messages": model_messages, "request": model_request or {}},
            tags=["model-call", _safe_label(model_name, "custom-model")],
            extra={
                "metadata": {
                    "ls_provider": ls_provider,
                    "ls_model_type": "chat",
                    "ls_model_name": model_name,
                    "ls_temperature": (
                        model_request.get("temperature")
                        if isinstance(model_request, dict)
                        else None
                    ),
                    "data_policy": self.data_policy,
                }
            },
        )
        # 把模型调用开始提交到 SDK 批处理队列。
        model_run.post()
        # 保存活动模型 Run，finished/failed 事件按请求顺序与它配对。
        context.active_model_run = model_run
        # 保存模型名供结束输出和异常路径使用。
        context.active_model_name = model_name

    def _finish_model_run(
        self,
        context: _RemoteTraceContext,
        payload: dict[str, Any],
        *,
        status: str,
    ) -> None:
        """完成真实模型 Run，并写入 LangSmith 标准 usage 与首 Token事件。"""

        # 没有活动模型 Run 表示开始事件未到达或远程创建已经失败，安全跳过。
        if context.active_model_run is None:
            # 不创建孤立的结束 Run，保持父子树和实际调用一致。
            return
        # 保存局部引用，后续先清空上下文也能保证异常时不会重复结束同一 Run。
        model_run = context.active_model_run
        # normalized_result 是 ChatCompletionResult 的 JSON，含规范化 Token、延迟和模型名。
        normalized_result = payload.get("normalized_result")
        # 只有字典响应才读取数值；协议失败统一使用零 Token。
        result_data = normalized_result if isinstance(normalized_result, dict) else {}
        # 输入 Token 使用供应商 usage 规范化结果，确保 LangSmith Tokens 不再为空。
        input_tokens = int(result_data.get("token_input") or 0)
        # 输出 Token 使用供应商 usage 规范化结果。
        output_tokens = int(result_data.get("token_output") or 0)
        # 标准 usage_metadata 是 LangSmith 聚合 Token 与计算 Cost 的识别入口。
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        # usage 来源写进普通 metadata，避免把估算值误解成供应商精确计量。
        usage_source = str(result_data.get("token_usage_source") or "unavailable")
        # RunTree 与测试替身都维护 extra.metadata，可直接追加非标准说明字段。
        model_run.extra.setdefault("metadata", {})["token_usage_source"] = usage_source
        # RunTree.set 会把 usage 放入 metadata.usage_metadata，服务端据此回填 token/cost 列。
        if hasattr(model_run, "set"):
            # Fake/旧 SDK 之外的正式 RunTree 走标准 setter 并验证 usage 字段。
            model_run.set(usage_metadata=usage_metadata)
        # 非流式调用的 first_token_time 是完整响应到达时间，代表可观测 TTFT 上界。
        first_token_time = payload.get("first_token_time")
        # 有合法 ISO 时间时写入 new_token 事件，LangSmith 用首个该事件计算 First Token。
        if isinstance(first_token_time, str) and first_token_time:
            # kwargs 不保存 token 正文，避免为了 TTFT 再复制一份完整模型回答。
            model_run.add_event(
                {
                    "name": "new_token",
                    "time": first_token_time,
                    "kwargs": {},
                }
            )
        # 成功输出保留模型原始响应、规范化结果和标准 usage，失败输出保留脱敏异常详情。
        outputs = {
            "status": status,
            "model": context.active_model_name,
            "usage_metadata": usage_metadata,
            **payload,
        }
        # 失败错误字段使用固定异常类型，不把可能很长的供应商正文放进 LangSmith error 列。
        error = str(payload.get("exception_type") or "ModelCallError") if status == "failed" else None
        # 设置模型 Run 的输出、结束时间和错误状态。
        model_run.end(outputs=outputs, error=error)
        # Patch 把 usage、事件、输出和耗时一起交给 SDK 批处理上传。
        model_run.patch()
        # 清空活动模型引用，下一次模型调用可以创建新的兄弟 Run。
        context.active_model_run = None
        # 同步清空模型名，避免下一次异常错误引用旧模型。
        context.active_model_name = None

    def _finish_active_step(self, context: _RemoteTraceContext, *, status: str) -> None:
        """完成当前节点子 Run，并提交安全输出摘要。"""

        # 状态节点结束前确保嵌套模型 Run 已收尾；缺少 finished 事件时按失败处理。
        if context.active_model_run is not None:
            # 固定异常类型说明模型调用被状态切换或请求结束中断。
            self._finish_model_run(
                context,
                {"exception_type": "ModelRunInterrupted"},
                status="failed",
            )
        # 没有活动节点表示根 Run 尚未收到首个状态迁移，无需 patch。
        if context.active_step is None:
            # 直接返回保持上下文不变。
            return
        # 节点输出只记录状态、内部码和安全事件数量。
        outputs = {
            "status": status,
            "step_code": context.active_step_code,
            "event_count": context.active_step_event_count,
        }
        # 完整模式在节点结束时保存最近的全量 AgentState，和 state_before 组成可视化状态差异。
        if self.captures_full_content and context.latest_state_snapshot is not None:
            # latest_state_snapshot 已经在事件入口完成凭据清理和规模限制。
            outputs["state_after"] = context.latest_state_snapshot
        # failed 状态写固定错误类别，不能写底层异常或用户数据。
        error = "agent_step_failed" if status == "failed" else None
        # end 在本地 RunTree 上设置结束时间和输出摘要。
        context.active_step.end(outputs=outputs, error=error)
        # patch 通过 SDK 批处理提交节点结束、事件与耗时。
        context.active_step.patch()
        # 完成后清空当前子 Run，下一次迁移会创建新节点。
        context.active_step = None
        # 同步清空内部码，避免根 Run 收尾误引用旧节点。
        context.active_step_code = None
        # 节点事件计数归零，准备下一步。
        context.active_step_event_count = 0
        # 节点完成后清空快照引用，避免长会话不必要地保留上一节点大对象。
        context.latest_state_snapshot = None

    def finish_run(
        self,
        *,
        trace_id: str,
        status: str,
        final_state: str,
        intent: str | None,
        domain_skill: str | None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """完成节点和根 Run；完整模式附加最终状态和已脱敏异常正文。"""

        # 不可用 Adapter 无需查找上下文。
        if not self.available:
            # no-op 保持业务收尾不依赖远程平台。
            return
        # 取出上下文后立即从共享字典移除，迟到事件将被安全忽略。
        with self._lock:
            # pop 保证同一个 trace_id 只结束一次。
            context = self._contexts.pop(trace_id, None)
        # 没有上下文表示根 Run 创建失败、已结束或本次 Trace 未采样。
        if context is None:
            # 无远程对象可结束，直接返回。
            return
        # 节点结束和根 Patch 都在异常边界内执行。
        try:
            # 完整模式先投影最终详情，使最后一个 FINAL/ERROR 子节点也能获得真正终态快照。
            projected_details = (
                self._project_payload(details)
                if self.captures_full_content and details is not None
                else None
            )
            # Engine 约定 details.state 是最终 AgentState；存在时覆盖迁移前留下的旧快照。
            if isinstance(projected_details, dict) and isinstance(
                projected_details.get("state"),
                dict,
            ):
                # 最后子节点 state_after 与根 Run details 使用同一份已脱敏最终状态。
                context.latest_state_snapshot = projected_details["state"]
            # 请求失败时把当前节点标记 failed；正常路径标记 completed。
            self._finish_active_step(
                context,
                status="failed" if status == "failed" else "completed",
            )
            # 根输出只保留最终控制状态和统计信息。
            outputs = {
                "status": _safe_label(status, "unknown_status"),
                "final_state": _safe_label(final_state, "UNKNOWN"),
                "intent": _safe_label(intent, "unknown_intent"),
                "domain_skill": _safe_label(domain_skill, "none"),
                "event_count": context.total_event_count,
            }
            # 完整模式把最终 answer 提升为标准 output，避免 Thread UI 把控制字段当成助手回答。
            if isinstance(projected_details, dict) and isinstance(
                projected_details.get("state"),
                dict,
            ):
                # 读取最终状态中的 answer；空回答不额外创建误导性的 output 字段。
                final_answer = projected_details["state"].get("answer")
                # 只有字符串回答才适合作为 LangSmith Chat/Thread 的助手输出。
                if isinstance(final_answer, str) and final_answer:
                    # output 是 LangSmith 通用输出字段，Trace 详情和 Thread 视图都能正确渲染。
                    outputs["output"] = final_answer
            # 完整模式把最终 AgentState/响应详情写到根输出，便于从单条 Run 直接回放整个请求。
            if projected_details is not None:
                # 复用上方已完成凭据清理的投影，避免对大型最终状态重复递归扫描。
                outputs["details"] = projected_details
            # 完整失败 Trace 可保留异常正文，但必须先通过同一凭据清理边界。
            if self.captures_full_content and status == "failed" and exception_message:
                # 使用结构化投影而不是直接赋值，避免供应商异常回显 Authorization Header。
                outputs["exception_message"] = self._project_payload(
                    {"exception_message": exception_message}
                )["exception_message"]
            # 根错误只记录异常类型，不上传异常消息、堆栈或 SQL 参数。
            error = _safe_label(exception_type, "AgentRunError") if status == "failed" else None
            # end 设置根 Run 输出、错误和结束时间。
            context.root.end(outputs=outputs, error=error)
            # patch 交由自动批处理队列提交完整根 Run。
            context.root.patch()
        # 结束提交失败仍只影响远程观测。
        except Exception as exc:
            # 本地告警记录失败类型和本地 Trace ID。
            self._warn("langsmith_run_finish_failed", exc, trace_id=trace_id)
            # 返回后业务响应继续正常封装。
            return
        # 远程收尾已排队时记录本地确认事件。
        if self.log is not None:
            # 不记录输出正文，只记录本地 Trace、远程 Run ID 和最终状态。
            self.log.event(
                "langsmith_run_finished",
                trace_id=trace_id,
                langsmith_run_id=str(context.root.id),
                status=status,
                final_state=final_state,
            )

    def flush(self) -> None:
        """应用关闭时等待 LangSmith SDK 批处理队列完成。"""

        # 没有可用 Client 时无需等待。
        if not self.available or self.client is None:
            # no-op 保持关闭流程统一。
            return
        # flush 网络等待必须处于异常边界，不能阻塞其它资源最终关闭。
        try:
            # 使用配置的有限超时等待批处理队列。
            self.client.flush(timeout=self.flush_timeout_seconds)
        # 关闭阶段失败只写本地告警，Redis/PostgreSQL 仍继续释放。
        except Exception as exc:
            # 记录异常类型，不输出网络响应或认证信息。
            self._warn("langsmith_flush_failed", exc)

    def _warn(self, event: str, exc: Exception, **fields: Any) -> None:
        """输出不含异常正文的统一 LangSmith 降级日志。"""

        # 没有本地 Logger 时静默降级，观测组件不能反向制造业务依赖。
        if self.log is None:
            # 直接返回，不尝试全局 logging 兜底。
            return
        # warning 只记录事件、异常类型和调用方提供的安全控制字段。
        self.log.warning(
            event,
            exception_type=type(exc).__name__,
            **_safe_payload(fields),
        )
