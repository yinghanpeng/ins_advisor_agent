"""State transition policy."""

# 文件说明：
# - 本文件属于显式状态机层，负责状态对象、节点函数、边或 checkpoint。
# - 所有复杂任务都应通过状态流转表达，避免把流程藏在大 Prompt 中。
from __future__ import annotations

from agent_core.graph.state import AgentNode, AgentState


def next_after_route(state: AgentState) -> AgentNode:
    """Route based on the capability chosen by the router."""
    if state.capability_route == "general":
        return AgentNode.GENERAL_TOOL_ROUTING
    if state.capability_route == "domain":
        return AgentNode.DOMAIN_WORKFLOW_ROUTING
    return AgentNode.GENERAL_RESPONSE_GENERATION


def next_after_domain_route(state: AgentState) -> AgentNode:
    """Route domain workflows that need sales intelligence."""
    if state.domain_skill == "insurance_advisor":
        return AgentNode.SALES_INTELLIGENCE_ROUTING
    return AgentNode.BUILD_CONTEXT

