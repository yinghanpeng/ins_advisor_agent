"""API-facing schemas."""

# 文件说明：
# - 本文件属于 Agent Gateway 适配层，负责 API 请求、路由或中间件边界。
# - 生产环境可在这一层补鉴权、限流、租户隔离和请求校验。
from agent_core.workflow.contracts import AgentRunRequest, AgentRunResponse, PublicAgentRunResponse

# 限定 API schema 模块的公开导出，避免内部 Workflow 契约被通配导入意外暴露。
__all__ = ["AgentRunRequest", "AgentRunResponse", "PublicAgentRunResponse"]
