"""Agent Gateway routes.

The route factory returns a FastAPI router when FastAPI is installed. Local
tests can call ``run_agent`` directly without the optional API dependency.
"""

# 文件说明：
# - 本文件属于 Agent Gateway 适配层，负责 API 请求、路由或中间件边界。
# - 生产环境可在这一层补鉴权、限流、租户隔离和请求校验。
from __future__ import annotations

from typing import Any

from agent_core.api.runtime import get_runtime
from agent_core.memory.privacy import ConsentRequest, MemoryPrivacyRequest, MemoryPrivacyService
from agent_core.workflow.contracts import AgentRunRequest, AgentRunResponse, PublicAgentRunResponse
from agent_core.workflow.engine import WorkflowEngine

# FastAPI Request 是可选类型依赖，延迟捕获导入错误以支持纯核心环境。
try:
    # FastAPI 安装时使用真实 Request 类型；未安装时直接函数调用仍可导入本模块。
    from fastapi import Request as FastAPIRequest
# 可选 FastAPI 依赖不可用时进入纯核心兼容分支。
except Exception:
    # 可选依赖不可用时用 Any 占位，保证非 HTTP 场景仍能导入并调用 run_agent。
    FastAPIRequest = Any  # 可选依赖占位保证核心模块可导入；type: ignore[misc,assignment]


# 直接函数调用（单元测试、SDK Adapter）复用一个进程级 Engine；ASGI 路由始终优先使用 lifespan Runtime。
_adapter_engine = WorkflowEngine()


def run_agent(
    request: AgentRunRequest,
    engine: WorkflowEngine | None = None,
) -> AgentRunResponse:
    """通过共享生产 Engine 或测试显式注入的 Engine 执行请求。"""
    # API 层不直接调用 graph 节点，而是统一走 WorkflowEngine，保证 CLI/API/Dify 主链路一致。
    # 补充说明：生产路由从 lifespan Runtime 取共享实例，不能每次请求创建 MemoryManager。
    if engine is not None:
        # 测试或 SDK 显式传入的 Engine 优先，便于隔离后端和构造确定性用例。
        selected_engine = engine
    # 未显式注入 Engine 时优先复用生产 lifespan Runtime。
    else:
        # 优先尝试取得 FastAPI lifespan Runtime；直接函数调用环境可能尚未初始化。
        try:
            # 正常 HTTP 请求从 lifespan 获取 Redis/PostgreSQL 生产 Engine。
            selected_engine = get_runtime().engine
        # 非 ASGI 直接调用尚无 Runtime 时才使用进程级兼容 Engine。
        except RuntimeError:
            # 仅非 ASGI 直接调用使用共享 Adapter Engine，仍不会每次请求新建实例。
            selected_engine = _adapter_engine
    # 所有入口最终只调用一次统一 Engine，返回内部完整诊断响应。
    return selected_engine.run(request)


def run_agent_stream(
    request: AgentRunRequest,
    engine: WorkflowEngine | None = None,
) -> dict[str, object]:
    """返回流式事件骨架；未来可替换为真正 SSE StreamingResponse。"""
    # 第一版仍同步执行完整 workflow，只把 state.stream_events 作为 adapter-ready 输出。
    response = run_agent(request, engine)
    # HTTP 流式适配器只附带客户安全 DTO；完整 Trace 和知识正文留在服务端日志/内部响应。
    public_response = PublicAgentRunResponse.from_internal(response)
    # 事件骨架保留进度事件，但最终正文必须使用客户安全 DTO 的字段投影。
    return {
        "trace_id": response.trace_id,
        "final_state": response.final_state,
        "stream_events": response.stream_events,
        "final_response": public_response.model_dump(),
    }


def build_router():
    """Return an APIRouter or None when FastAPI is not installed."""
    # FastAPI 是可选依赖；本地测试没有安装时，返回 None 而不是让导入失败。
    try:
        from fastapi import APIRouter, HTTPException
    # FastAPI 不可用时保留直接函数调用能力，不构建 HTTP Router。
    except Exception:
        # FastAPI 未安装时通知 server/测试跳过路由注册，而不是在模块导入阶段崩溃。
        return None
    # 创建 API router，server.py 会把它 include 到 FastAPI app。
    router = APIRouter()

    # /agent/run 是正式执行入口，输入输出都使用结构化 Pydantic 契约。
    @router.post("/agent/run", response_model=PublicAgentRunResponse)
    def agent_run(request: AgentRunRequest, http_request: FastAPIRequest) -> PublicAgentRunResponse:
        """校验认证租户并返回 default-deny 的客户安全 Agent 响应。"""
        # 每个 API 请求交给 run_agent，避免路由函数里重复 workflow 逻辑。
        # 请求体 tenant_id 必须与网关认证出的租户一致，防止调用方横向切换租户。
        if request.tenant_id != getattr(http_request.state, "tenant_id", None):
            # Header 认证租户与请求体租户不一致时拒绝，防止横向读取其它租户数据。
            raise HTTPException(status_code=403, detail="tenant identity mismatch")
        # Engine 的完整响应仅用于进程内诊断；客户 HTTP 契约执行 default-deny 字段投影。
        return PublicAgentRunResponse.from_internal(run_agent(request))

    # /agent/stream 当前返回事件骨架，未来可无破坏升级为 SSE。
    @router.post("/agent/stream")
    def agent_stream(request: AgentRunRequest, http_request: FastAPIRequest) -> dict[str, object]:
        """校验租户后同步执行工作流并返回可升级为 SSE 的安全事件包。"""
        # 不直接在路由里写状态机逻辑，仍复用 WorkflowEngine 和 run_agent_stream。
        if request.tenant_id != getattr(http_request.state, "tenant_id", None):
            # 流式适配入口执行与同步入口相同的租户一致性校验。
            raise HTTPException(status_code=403, detail="tenant identity mismatch")
        # 当前版本同步完成工作流后返回事件骨架，后续可在该边界替换为 SSE。
        return run_agent_stream(request)

    # /agent/eval 是轻量评估入口，只返回 trace_id 和 final_state，方便自动化评测。
    @router.post("/agent/eval")
    def agent_eval(request: AgentRunRequest, http_request: FastAPIRequest) -> dict[str, str]:
        """运行完整评估请求，但只公开 trace_id 和最终状态。"""
        # 先执行完整 Agent，再抽取评估需要的核心字段。
        if request.tenant_id != getattr(http_request.state, "tenant_id", None):
            # 评测入口也必须绑定认证租户，不能因返回字段少而绕过隔离。
            raise HTTPException(status_code=403, detail="tenant identity mismatch")
        # 执行统一工作流以确保评测与正式请求经过完全相同的节点。
        response = run_agent(request)
        # 返回最小评估结果，避免 eval endpoint 暴露完整 trace。
        return {"trace_id": response.trace_id, "final_state": response.final_state}

    # 隐私导出默认只返回脱敏消息和结构化记忆，不解密 PostgreSQL 原始证据。
    @router.post("/memory/export")
    def memory_export(request: MemoryPrivacyRequest, http_request: FastAPIRequest) -> dict[str, object]:
        """按认证租户导出指定主体的脱敏记忆视图。"""
        # 复用 lifespan 中已经初始化并健康检查通过的生产资源。
        runtime = get_runtime()
        # 隐私服务同时需要通用记忆和业务记忆后端，确保一次导出覆盖两类数据。
        service = MemoryPrivacyService(runtime.engine.memory_manager, runtime.business_store)  # 共享两类记忆后端执行完整导出；type: ignore[arg-type]
        # tenant_id 只取认证中间件写入的值，不接受请求体选择租户。
        return service.export_subject(http_request.state.tenant_id, request)

    # 删除接口同时处理 Redis Session、通用 Preference/Embedding 和可选业务客户血缘。
    @router.post("/memory/delete")
    def memory_delete(request: MemoryPrivacyRequest, http_request: FastAPIRequest) -> dict[str, int]:
        """按认证租户协调删除 Redis、Preference 和业务记忆。"""
        # 获取共享后端，避免删除操作落到进程内兼容存储。
        runtime = get_runtime()
        # 用同一个隐私服务协调 Redis、Preference 和业务记录的删除范围。
        service = MemoryPrivacyService(runtime.engine.memory_manager, runtime.business_store)  # 共享两类记忆后端协调删除；type: ignore[arg-type]
        # 删除范围由认证租户和结构化主体请求共同确定。
        return service.delete_subject(http_request.state.tenant_id, request)

    # Consent Grant 按主体和用途写入版本化授权记录。
    @router.post("/memory/consent/grant")
    def memory_consent_grant(request: ConsentRequest, http_request: FastAPIRequest) -> dict[str, str]:
        """为受支持主体和用途写入 granted Consent 版本。"""
        # 仅支持系统已经实现读写语义的主体/用途组合。
        if not _is_supported_consent_scope(request):
            # 未建模的主体/用途组合没有授权语义，因此直接返回参数错误。
            raise HTTPException(status_code=422, detail="unsupported consent scope")
        # 使用生产 Runtime，保证授权记录与后续事实写入读取同一数据库。
        runtime = get_runtime()
        # 构造统一隐私服务以封装版本化 Consent 写入逻辑。
        service = MemoryPrivacyService(runtime.engine.memory_manager, runtime.business_store)  # 共享后端写入可追踪授权版本；type: ignore[arg-type]
        # 认证租户是授权记录的强制隔离维度，状态显式写为 granted。
        service.set_consent(http_request.state.tenant_id, request, status="granted")
        # 仅回传最终状态，不暴露内部授权记录或数据库主键。
        return {"status": "granted"}

    # Consent Revoke 会立即禁用 Preference 召回；客户事实查询也要求 granted 状态。
    @router.post("/memory/consent/revoke")
    def memory_consent_revoke(request: ConsentRequest, http_request: FastAPIRequest) -> dict[str, str]:
        """为受支持主体和用途写入 revoked Consent 版本。"""
        # 撤销也使用同一白名单，避免写入无法被读取路径解释的 Consent 记录。
        if not _is_supported_consent_scope(request):
            # 与 grant 保持同一用途白名单，防止构造系统不理解的撤销记录。
            raise HTTPException(status_code=422, detail="unsupported consent scope")
        # 获取与在线记忆读写共享的生产 Runtime。
        runtime = get_runtime()
        # 隐私服务负责同时影响后续通用偏好和业务事实的可访问性。
        service = MemoryPrivacyService(runtime.engine.memory_manager, runtime.business_store)  # 共享后端立即写入撤销状态；type: ignore[arg-type]
        # 写入 revoked 状态后，后续读取路径必须按默认拒绝语义停止召回。
        service.set_consent(http_request.state.tenant_id, request, status="revoked")
        # 响应只确认状态变化，不返回被撤销主体的其它数据。
        return {"status": "revoked"}

    # 返回构建好的 router；FastAPI 不可用时前面已经返回 None。
    return router


def _is_supported_consent_scope(request: ConsentRequest) -> bool:
    """限制主体类型与用途组合，防止伪造不支持的授权语义。"""
    # 白名单只允许用户偏好记忆和客户业务记忆两种已经实现完整读写语义的组合。
    allowed = {
        ("user", "preference_memory"),
        ("customer", "memory_processing"),
    }
    # 返回布尔值，由 FastAPI endpoint 转成稳定的 422 响应。
    return (request.subject_type, request.purpose) in allowed
