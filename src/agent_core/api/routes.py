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
    return WorkflowEngine().run(request)


def build_router():
    """Return an APIRouter or None when FastAPI is not installed."""
    try:
        from fastapi import APIRouter
    except Exception:
        return None
    router = APIRouter()

    @router.post("/agent/run", response_model=AgentRunResponse)
    def agent_run(request: AgentRunRequest) -> AgentRunResponse:
        return run_agent(request)

    @router.post("/agent/eval")
    def agent_eval(request: AgentRunRequest) -> dict[str, str]:
        response = run_agent(request)
        return {"trace_id": response.trace_id, "final_state": response.final_state}

    return router

