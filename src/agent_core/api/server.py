"""FastAPI application adapter."""

# 文件说明：
# - 本文件属于 Agent Gateway 适配层，负责 API 请求、路由或中间件边界。
# - 生产环境可在这一层补鉴权、限流、租户隔离和请求校验。
from __future__ import annotations

import ipaddress
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Request

from agent_core.api.middleware import GatewayRequestError, enforce_gateway_request
from agent_core.api.routes import build_router
from agent_core.api.registry_routes import build_registry_router
from agent_core.api.runtime import get_runtime, initialize_runtime, shutdown_runtime
from agent_core.config.runtime import RuntimeSettings, load_runtime_settings


def _registry_console_defaults(settings: RuntimeSettings, client_host: str | None) -> dict[str, str]:
    """仅为本机控制台解析已授权的凭据默认值。"""

    # 严格环境默认关闭注入；本机控制台必须通过环境变量显式授权，避免部署时意外开启。
    strict_environment = settings.app_env.casefold() in {"prod", "production", "staging", "preprod"}
    # 只有明确的布尔真值才视为授权，空值或拼写错误一律保持关闭。
    explicit_opt_in = os.getenv("REGISTRY_CONSOLE_DEFAULTS_ENABLED", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # local/test 环境保持开发便利；严格环境则要求运维侧显式开启。
    if strict_environment and not explicit_opt_in:
        return {}
    # Request.client 缺失时按非本机请求处理，避免测试替身或代理场景意外放宽。
    if not client_host:
        return {}
    # 只接受标准回环 IP；不信任可伪造的 Host 或 Forwarded Header。
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    # 非 IP 客户端地址不能证明来自本机，因此禁止注入开发凭据。
    except ValueError:
        return {}
    # 非回环请求即使处于 local 环境也必须手动输入凭据。
    if not is_loopback:
        return {}
    # 租户密钥映射使用网关配置声明的 Secret 环境变量名。
    raw_mapping = os.getenv(settings.api.tenant_api_keys_env, "")
    # 缺少租户密钥时仍可填充非敏感身份字段，但不能伪造可用凭据。
    try:
        parsed_mapping = json.loads(raw_mapping) if raw_mapping else {}
    # 非法 Secret 配置不应阻止静态控制台加载，数据 API 会继续按网关规则拒绝。
    except json.JSONDecodeError:
        parsed_mapping = {}
    # 只有字符串到字符串的映射才可作为浏览器请求 Header 使用。
    tenant_keys = {
        str(tenant): str(api_key)
        for tenant, api_key in parsed_mapping.items()
        if isinstance(parsed_mapping, dict) and tenant and api_key
    } if isinstance(parsed_mapping, dict) else {}
    # 显式本地控制台租户优先，否则使用配置中的第一个租户。
    requested_tenant = os.getenv("REGISTRY_CONSOLE_TENANT_ID", "").strip()
    # 未显式指定时按配置插入顺序选择本地开发租户。
    tenant_id = requested_tenant or next(iter(tenant_keys), "")
    # 返回值只进入本次 no-store HTML 响应，不写日志、磁盘或浏览器持久存储。
    return {
        "tenantId": tenant_id,
        "apiKey": tenant_keys.get(tenant_id, ""),
        "adminToken": os.getenv("REGISTRY_ADMIN_TOKEN", ""),
        "actorId": os.getenv("REGISTRY_CONSOLE_ACTOR_ID", "local-admin"),
        "actorRoles": os.getenv("REGISTRY_CONSOLE_ACTOR_ROLES", "administrator"),
    }


def _resolve_runtime_settings(settings: RuntimeSettings | None = None) -> RuntimeSettings:
    """解析应用构建配置；显式注入优先，否则统一遵守 CONFIG_DIR。"""
    # 单独封装解析逻辑，确保 CORS 注册与 lifespan 初始化不会分别读取不同目录。
    if settings is not None:
        # 测试或嵌入式调用显式注入时原样复用，避免再次读取环境配置。
        return settings
    # 默认从 CONFIG_DIR 加载整套配置，未设置时使用仓库 configs 目录。
    return load_runtime_settings(os.getenv("CONFIG_DIR", "configs"))


def create_app(settings: RuntimeSettings | None = None):
    """Create the FastAPI app when FastAPI is installed."""
    # FastAPI 属于可选 API 依赖，延迟导入使纯 CLI/测试环境仍可使用核心工作流。
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    # 可选 API 依赖导入失败时转成包含安装提示的稳定运行时错误。
    except Exception as exc:
        # API 依赖缺失时抛出带安装方式的明确错误，并保留原始异常链用于排障。
        raise RuntimeError(
            "FastAPI is not installed. Install with `pip install -e \".[api]\"`."
        ) from exc
    # CORS 和 lifespan 必须使用同一份配置；CONFIG_DIR 支持容器挂载自定义配置目录。
    runtime_settings = _resolve_runtime_settings(settings)

    @asynccontextmanager
    async def lifespan(_app):
        """在 ASGI 应用生命周期内创建并关闭一次共享生产 Runtime。"""
        # 启动失败会阻止服务接收流量，生产后端不可用时不回退内存模式。
        initialize_runtime(runtime_settings)
        # yield 前后分别对应服务启动完成和服务退出阶段，finally 保证异常退出也会清理资源。
        try:
            # 将控制权交给 FastAPI；整个服务生命周期共享上面创建的 Runtime。
            yield
        # 无论服务正常停止还是异常退出，都必须进入统一资源清理阶段。
        finally:
            # 服务停止时释放 Redis/PostgreSQL 连接池。
            shutdown_runtime()

    # 创建 ASGI 应用并绑定上述 lifespan，确保资源初始化/释放有唯一入口。
    app = FastAPI(
        title="Production Agent Framework",
        version="0.1.0",
        lifespan=lifespan,
    )
    # 调试页面与 server.py 一起发布；使用绝对路径避免 uvicorn 从不同工作目录启动时找不到资源。
    playground_path = Path(__file__).with_name("static") / "agent-playground.html"
    # Artifact Registry reuses the same zero-build static delivery path as the Agent Playground.
    registry_path = Path(__file__).with_name("static") / "artifact-registry.html"
    # 中间件在 Pydantic 解析前执行请求大小、鉴权和分布式限流。
    @app.middleware("http")
    async def production_gateway_middleware(request: Request, call_next):
        """在进入业务路由前执行生产网关校验，并把追踪标识写回响应头。"""

        # 将网关错误统一转换成安全 HTTP 响应，其它业务异常继续交给 FastAPI 异常处理器。
        try:
            # 在请求体模型解析前完成大小限制、租户鉴权、限流和 trace_id 解析。
            tenant_id, trace_id = await enforce_gateway_request(request, get_runtime())
        # 已知网关拒绝转换成安全 HTTP 响应，不把内部异常细节暴露给客户。
        except GatewayRequestError as exc:
            # 拒绝响应只返回安全摘要，不回显 API Key、请求正文或 Redis Key。
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        # 把已认证租户写入 request.state，业务路由不得再信任请求体选择租户。
        request.state.tenant_id = tenant_id
        # 保存最终追踪标识，供后续路由或日志适配器读取。
        request.state.trace_id = trace_id
        # 鉴权通过后才调用下一层中间件和实际 FastAPI 路由。
        response = await call_next(request)
        # 无论业务响应内容如何，都在响应头回传本次实际使用的 trace_id。
        response.headers["x-trace-id"] = trace_id
        # 返回业务响应，至此网关中间件处理结束。
        return response

    # 根路径仅提供不含密钥的本地调试台；实际 Agent 请求仍在 /agent/run 走网关鉴权。
    @app.get("/", include_in_schema=False)
    def agent_playground():
        """返回用于本地联调的单页 Agent Playground。"""

        # FileResponse 流式读取静态 HTML，不把页面内容复制进 Python 字符串或 API 响应模型。
        return FileResponse(playground_path, media_type="text/html; charset=utf-8")

    @app.get("/registry", include_in_schema=False)
    def artifact_registry_console(request: Request):
        """返回管理台，并仅对本机授权访问注入环境中的默认凭据。"""

        # 每次读取静态资源可让本地开发修改立即生效，同时避免把 Secret 写入源码文件。
        html = registry_path.read_text(encoding="utf-8")
        # 只使用 ASGI 解析的直连客户端地址判断回环访问，不读取代理 Header。
        client_host = request.client.host if request.client else None
        # 默认值由环境和安全边界共同决定，生产或非本机访问会得到空对象。
        defaults = _registry_console_defaults(runtime_settings, client_host)
        # 转义左尖括号可阻断值中构造 </script> 的 HTML 注入路径。
        payload = json.dumps(defaults, ensure_ascii=False).replace("<", "\\u003c")
        # 用不可执行的 application/json 数据块承载默认值，页面脚本再按字段白名单读取。
        injection = f'<script id="registryConsoleDefaults" type="application/json">{payload}</script>'
        # 标记只存在一次；替换失败时页面仍正常加载，只是不自动填充凭据。
        html = html.replace("<!-- REGISTRY_CONSOLE_DEFAULTS -->", injection, 1)
        # 凭据页面必须禁止浏览器、代理或共享缓存持久化。
        response = HTMLResponse(html, media_type="text/html; charset=utf-8")
        # no-store 覆盖默认缓存行为，确保每次加载都重新读取当前环境配置。
        response.headers["Cache-Control"] = "no-store"
        # 返回已经完成安全注入的本地控制台页面。
        return response

    # CORS 仅在配置了精确白名单时启用，不使用通配 Origin + Credential 组合。
    if runtime_settings.api.allowed_origins:
        # 仅将配置模型已校验的精确白名单传给 Starlette CORS 中间件。
        app.add_middleware(
            CORSMiddleware,
            allow_origins=runtime_settings.api.allowed_origins,
            allow_credentials=runtime_settings.api.allow_credentials,
            allow_methods=runtime_settings.api.allowed_methods,
            allow_headers=runtime_settings.api.allowed_headers,
        )
    # 构建可选 API Router；缺少 FastAPI 的直接函数场景会返回 None。
    router = build_router()
    # Router 存在时才注册；可选依赖不可用的兼容返回值为 None。
    if router:
        # 将 Agent、评估、记忆隐私和 Consent 端点注册到同一应用。
        app.include_router(router)
    # Registry endpoints are isolated in their own OpenAPI tag and require a dedicated admin credential.
    registry_router = build_registry_router()
    if registry_router:
        app.include_router(registry_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """进程存活探针，不查询外部依赖。"""
        # 固定返回 ok，仅说明 ASGI 进程能处理请求。
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        """返回共享 Runtime 与可选 LangSmith 观测层的就绪状态。"""
        # 读取全局 Runtime；尚未初始化时会抛错并使就绪探针失败。
        runtime = get_runtime()
        # tracing 未打开时明确返回 disabled，避免把未启用误判成远端故障。
        if not runtime.engine.langsmith.enabled:
            # disabled 只描述开关状态，不影响 Redis/PostgreSQL 就绪结论。
            langsmith_status = "disabled"
        # 已请求启用但 Client 未成功初始化时报告 degraded，业务仍可继续使用本地日志。
        elif not runtime.engine.langsmith.available:
            # degraded 提醒运维检查 Key、网络或 SDK，同时保持 readiness 为 ready。
            langsmith_status = "degraded"
        # Client 已建立时表示运行时请求会创建远程 Run Tree。
        else:
            # enabled 是观测层健康状态，不承诺远端 Experiment 自动执行。
            langsmith_status = "enabled"
        # 返回非敏感项目名和状态，绝不通过探针公开 Key、Endpoint 或 Workspace ID。
        return {
            "status": "ready",
            "langsmith": langsmith_status,
            "langsmith_project": runtime.engine.langsmith.project or "",
            "langsmith_data_policy": runtime.engine.langsmith.data_policy,
            "langsmith_thread_grouping": (
                "enabled" if runtime.engine.langsmith.thread_grouping_enabled else "disabled"
            ),
        }
    # 返回配置、路由和中间件均已注册完成的 FastAPI 应用。
    return app


# 模块级 app 供 uvicorn 直接导入；依赖或生产配置不可用时保持可导入。
try:
    # 立即构建默认应用对象，使 `uvicorn agent_core.api.server:app` 可以直接启动。
    app = create_app()
# 默认应用构建失败时保留模块可导入性，CLI 与核心工作流仍可单独运行。
except RuntimeError:
    # CLI/测试未安装 API 依赖时使用 None，直接 WorkflowEngine 调用不受影响。
    app = None
