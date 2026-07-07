"""Agent Gateway routes.

The route factory returns a FastAPI router when FastAPI is installed. Local
tests can call ``run_agent`` directly without the optional API dependency.
"""

# 文件说明：
# - 本文件属于 Agent Gateway 适配层，负责 API 请求、路由或中间件边界。
# - 生产环境可在这一层补鉴权、限流、租户隔离和请求校验。
from __future__ import annotations

from agent_core.workflow.contracts import AgentRunRequest, AgentRunResponse
from agent_core.workflow.engine import WorkflowEngine


def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    """Run a single agent request."""
    # API 层不直接调用 graph 节点，而是统一走 WorkflowEngine，保证 CLI/API/Dify 主链路一致。
    return WorkflowEngine().run(request)


def build_router():
    """Return an APIRouter or None when FastAPI is not installed."""
    # FastAPI 是可选依赖；本地测试没有安装时，返回 None 而不是让导入失败。
    try:
        from fastapi import APIRouter
    except Exception:
        return None
    # 创建 API router，server.py 会把它 include 到 FastAPI app。
    router = APIRouter()

    # /agent/run 是正式执行入口，输入输出都使用结构化 Pydantic 契约。
    @router.post("/agent/run", response_model=AgentRunResponse)
    def agent_run(request: AgentRunRequest) -> AgentRunResponse:
        # 每个 API 请求交给 run_agent，避免路由函数里重复 workflow 逻辑。
        return run_agent(request)

    # /agent/eval 是轻量评估入口，只返回 trace_id 和 final_state，方便自动化评测。
    @router.post("/agent/eval")
    def agent_eval(request: AgentRunRequest) -> dict[str, str]:
        # 先执行完整 Agent，再抽取评估需要的核心字段。
        response = run_agent(request)
        # 返回最小评估结果，避免 eval endpoint 暴露完整 trace。
        return {"trace_id": response.trace_id, "final_state": response.final_state}

    # 返回构建好的 router；FastAPI 不可用时前面已经返回 None。
    return router
