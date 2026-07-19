"""Gateway middleware helpers."""

# 文件说明：
# - 本文件属于 Agent Gateway 适配层，负责 API 请求、路由或中间件边界。
# - 生产环境可在这一层补鉴权、限流、租户隔离和请求校验。
from __future__ import annotations

import json
import os
import re
import secrets
import time
from typing import Any

from agent_core.utils.ids import new_trace_id


# 预编译租户标识白名单：首字符必须是字母或数字，总长度最多 128，避免把任意文本带入 Redis Key。
_SAFE_TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def is_public_gateway_request(request: Any) -> bool:
    """判断请求是否只访问无需租户凭据的健康探针或本地调试页面。"""

    # 从 ASGI Request 读取方法和路径；测试替身只需提供这两个最小字段。
    method = str(getattr(request, "method", "GET")).upper()
    # URL.path 是已解析路径，不读取 query string，避免通过参数绕过鉴权范围。
    path = str(request.url.path)
    # 健康与就绪探针供编排平台调用，不读取业务数据也不占用租户限流预算。
    if path in {"/health", "/ready"}:
        # 探针允许任意 HTTP 方法进入现有 FastAPI 路由的标准处理与响应。
        return True
    # 两个本地控制台只提供静态 HTML；仅允许安全读取方法，不能借此放行任何管理 API。
    if path in {"/", "/registry"} and method in {"GET", "HEAD"}:
        # 页面不携带密钥、不读取业务数据，实际 API 请求仍需独立的租户和管理凭据。
        return True
    # 其它请求都必须继续经过租户身份、API Key 与限流校验。
    return False


def inject_trace_id(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Return headers with an x-trace-id value."""
    # 复制调用方 Header，避免 setdefault 原地修改调用方持有的字典。
    headers = dict(headers or {})
    # 仅在调用方没有追踪标识时生成新 ID，已有 ID 会原样贯穿请求链路。
    headers.setdefault("x-trace-id", new_trace_id())
    # 返回包含确定追踪标识的新字典，供直接函数调用和健康检查复用。
    return headers


async def enforce_gateway_request(request: Any, runtime: Any) -> tuple[str, str]:
    """执行请求体上限、租户绑定鉴权和 Redis 分布式限流。"""
    # 健康探针和只读调试页面不读取业务租户，也不占用 Redis 速率预算。
    if is_public_gateway_request(request):
        # 公共请求使用 system 伪租户，并确保仍有可写入响应头的 trace_id。
        return "system", inject_trace_id(dict(request.headers))["x-trace-id"]

    # API 中间件的限制参数统一来自已验证的 api.yaml 配置模型。
    settings = runtime.settings.api
    # 同时检查 Content-Length 和实际 Body，防止 chunked 请求绕过声明长度。
    content_length = request.headers.get("content-length")
    # 只有 Header 存在时才校验声明值；不存在时仍会在读取正文后检查实际大小。
    if content_length:
        # 捕获整数转换错误并改写成稳定的网关错误，避免暴露 Python 异常细节。
        try:
            # HTTP Header 是字符串，先转换为整数再与字节上限比较。
            declared_length = int(content_length)
        # 非法整数格式统一映射为稳定的网关请求错误。
        except ValueError as exc:
            # 非整数 Content-Length 属于畸形请求，包装成稳定的网关 400 错误。
            raise GatewayRequestError(400, "invalid content-length") from exc
        # 声明长度必须是非负数且不能超过 api.max_request_bytes。
        if declared_length < 0 or declared_length > settings.max_request_bytes:
            # 负数或超过配置上限都在读取正文前拒绝，减少无效内存占用。
            raise GatewayRequestError(413, "request body too large")
    # 实际读取正文以覆盖 chunked 传输或伪造 Content-Length 的情况。
    body = await request.body()
    # 实际正文大小是最终判据，不能只信任调用方 Header。
    if len(body) > settings.max_request_bytes:
        # 实际字节数超过上限时返回 413，不让超大请求进入 Pydantic 和工作流。
        raise GatewayRequestError(413, "request body too large")

    # tenant_id 必须由受控 Header 提供，后续路由还会校验它与请求体一致。
    tenant_id = str(request.headers.get("x-tenant-id") or "").strip()
    # 完整匹配租户白名单，拒绝空值、路径分隔符、空格和超长标识。
    if not _SAFE_TENANT_ID.fullmatch(tenant_id):
        # 缺失或包含白名单外字符的租户标识统一按未认证请求拒绝。
        raise GatewayRequestError(401, "missing or invalid tenant identity")

    # api.yaml 开启鉴权时才读取和比较密钥；本地测试可显式关闭。
    if settings.require_api_key:
        # 读取调用方凭据但不记录或回显其内容。
        supplied_key = str(request.headers.get("x-api-key") or "")
        # 按当前租户解析期望密钥，确保不同租户不能复用彼此凭据。
        expected_key = _tenant_api_key(settings, tenant_id)
        # 任一密钥为空或恒定时间比较失败，都按未认证请求处理。
        if not supplied_key or not expected_key or not secrets.compare_digest(supplied_key, expected_key):
            # compare_digest 使用恒定时间比较，降低通过响应耗时猜测密钥的风险。
            raise GatewayRequestError(401, "invalid API key")

    # 固定窗口长度来自 api.yaml；Redis INCR 保证所有 API 实例共享同一租户预算。
    window = int(time.time() // settings.rate_limit_window_seconds)
    # 将租户和自然窗口拼入 Key，使不同租户、不同窗口分别计数。
    rate_key = f"agent:{tenant_id}:gateway:rate:{window}"
    # 使用事务 Pipeline 将计数和过期设置作为一个 Redis 批次执行。
    pipeline = runtime.redis_client.pipeline(transaction=True)
    # 原子递增当前租户窗口内的请求计数。
    pipeline.incr(rate_key)
    # Key TTL 独立配置且由 Runtime 模型校验不短于窗口，防止窗口内计数提前丢失。
    pipeline.expire(rate_key, settings.rate_limit_key_ttl_seconds)
    # 一次往返执行两条命令；第二个返回值仅表示 EXPIRE 是否成功。
    count, _expiry = pipeline.execute()
    # 计数从 1 开始；只有严格大于预算时才拒绝，因此预算内最后一次请求仍可执行。
    if int(count) > settings.rate_limit_per_minute:
        # 当前计数超过配置预算时同步拒绝，不进入业务状态机。
        raise GatewayRequestError(429, "rate limit exceeded")

    # trace_id 允许调用方传入，但为空时由服务端生成；响应头会返回最终值。
    trace_id = str(request.headers.get("x-trace-id") or new_trace_id())
    # 返回已认证租户和追踪标识，server.py 会把它们写入 request.state。
    return tenant_id, trace_id


def _tenant_api_key(settings: Any, tenant_id: str) -> str:
    """从 Secret 环境变量读取租户绑定 API Key。"""
    # JSON 映射允许每个租户独立轮换密钥；解析失败必须拒绝启动请求而不是放行。
    raw_mapping = os.getenv(settings.tenant_api_keys_env, "")
    # 映射 Secret 非空时优先使用租户专属凭据，不再考虑全局 Key。
    if raw_mapping:
        # 解析错误会被转换成不含 Secret 原文的 GatewayRequestError。
        try:
            # 将 Secret 环境变量解析成 tenant_id -> API Key 的 JSON 映射。
            mapping = json.loads(raw_mapping)
        # Secret 不是合法 JSON 时禁止回退全局凭据，按服务端配置错误处理。
        except json.JSONDecodeError as exc:
            # Secret 内容不是合法 JSON 时按服务端配置错误拒绝，禁止退回全局 Key。
            raise GatewayRequestError(500, "tenant API key configuration is invalid") from exc
        # JSON 顶层必须是对象，保证可以按已经认证的 tenant_id 精确索引。
        if not isinstance(mapping, dict):
            # 顶层必须是对象，数组或标量不能表达安全的租户键值映射。
            raise GatewayRequestError(500, "tenant API key configuration is invalid")
        # 只读取当前已校验 tenant_id 对应的值，不接受调用方指定其它映射路径。
        expected = mapping.get(tenant_id)
        # 空值表示该租户未配置凭据；非空值规范化为字符串供恒定时间比较。
        return str(expected) if expected else ""
    # 全局 Key 仅在配置显式允许时生效，默认禁止跨租户共用凭据。
    if settings.allow_global_api_key:
        # 只有 api.yaml 显式开启时才读取兼容性的全局共享密钥。
        return os.getenv(settings.api_key_env, "")
    # 默认返回空字符串，使上层鉴权按缺少期望凭据拒绝请求。
    return ""


class GatewayRequestError(RuntimeError):
    """可转换为 HTTP 响应的网关拒绝。"""

    def __init__(self, status_code: int, detail: str) -> None:
        """保存 HTTP 状态码和可安全公开的拒绝原因。"""

        # 状态码和安全摘要供 server middleware 返回，不包含凭据原文。
        super().__init__(detail)
        # 保存 HTTP 状态码，供 FastAPI 中间件构造对应拒绝响应。
        self.status_code = status_code
        # 保存不含敏感原文的错误摘要，作为响应 detail 返回。
        self.detail = detail
