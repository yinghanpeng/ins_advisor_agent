"""Agentic 工具循环 planner。

# 文件说明：
# - RuleBasedToolLoopPlanner 复用现有 ToolRouter，保证本地无模型也能运行。
# - ModelToolLoopPlanner 是可选接口壳；模型不可用时只返回 None，绝不编造外部事实或工具结果。
"""

from __future__ import annotations

from typing import Any, Protocol

from agent_core.agentic_loop.schemas import ToolLoopDecision
from agent_core.graph.state import AgentState
from agent_core.tools.router import ToolRouter
from agent_core.tools.schemas import ToolCall


class ToolLoopPlanner(Protocol):
    """工具循环 planner 协议，便于测试注入模型或规则实现。"""

    def decide(self, state: AgentState, *, iteration_index: int) -> ToolLoopDecision:
        """根据当前 AgentState 和轮次输出下一步工具决策。"""


class RuleBasedToolLoopPlanner:
    """本地可运行的规则 planner，内部复用现有 ToolRouter。"""

    def __init__(self, router: ToolRouter | None = None) -> None:
        """允许测试注入自定义 ToolRouter，默认使用本地注册表。"""
        # router 是现有一次性工具路由器，规则 planner 不重新发明工具选择逻辑。
        self.router = router or ToolRouter()

    def decide(self, state: AgentState, *, iteration_index: int) -> ToolLoopDecision:
        """输出下一步工具决策；已有成功工具结果时结束循环。"""
        # 如果上下文规划已经声明不需要工具，直接结束，避免误调 summarizer 兜底工具。
        if not state.context_needs.get("tool"):
            return ToolLoopDecision(
                action="finish",
                finish_reason="context_needs.tool is false",
                rationale_summary="上下文规划未要求工具，跳过工具循环。",
                confidence=1.0,
            )
        # 如果有成功工具结果，说明现有证据已足够进入知识融合和生成阶段。
        if any(item.get("status") == "success" for item in state.tool_results):
            return ToolLoopDecision(
                action="finish",
                finish_reason="successful_tool_observation_available",
                rationale_summary="已有成功工具 observation，结束工具循环。",
                confidence=0.9,
            )
        # 如果工具已失败且当前请求不允许继续重试，由 loop 外层根据预算决定停止或降级。
        if state.tool_results and all(item.get("status") != "success" for item in state.tool_results):
            return ToolLoopDecision(
                action="finish",
                finish_reason="only_failed_tool_observations_available",
                rationale_summary="已有失败 observation，交给校验与降级链路处理，不编造事实。",
                confidence=0.75,
            )
        # 复用现有 ToolRouter 选择白名单工具；route 返回 None 时结束并走保守回答。
        spec = self.router.route(state.input_text)
        if spec is None:
            return ToolLoopDecision(
                action="finish",
                finish_reason="no_registered_tool_matched",
                rationale_summary="规则路由没有匹配到注册工具，结束工具循环。",
                confidence=0.8,
            )
        # 工具参数仍由 graph.nodes.general_tool_routing 构造；planner 这里只保存最小工具名占位。
        tool_call = ToolCall(name=spec.name, arguments={}, trace_id=state.trace_id)
        # 返回结构化决策；不包含外部事实，只包含下一步调用哪个注册工具。
        return ToolLoopDecision(
            action="call_tool",
            tool_calls=[tool_call],
            rationale_summary=f"规则 planner 选择工具 {spec.name}，具体参数由现有 routing 节点生成。",
            confidence=0.8,
        )


class ModelToolLoopPlanner:
    """模型 planner 接口壳；未注入模型客户端时安全降级。"""

    def __init__(self, model_client: Any | None = None) -> None:
        """保存可选模型客户端；None 表示当前环境没有模型 planner。"""
        # model_client 由生产环境注入；本地测试默认 None。
        self.model_client = model_client

    def try_decide(self, state: AgentState, *, iteration_index: int) -> ToolLoopDecision | None:
        """尝试用模型输出工具决策；不可用或非法时返回 None。"""
        # 没有模型客户端时明确返回 None，让调用方走 RuleBasedToolLoopPlanner。
        if self.model_client is None:
            state.add_trace_event(
                "tool_loop_model_planner_unavailable",
                iteration_index=iteration_index,
                reason="model_client_not_configured",
            )
            return None
        # 第一版不强行接入真实模型 planner，避免引入网络依赖和隐藏推理链。
        state.add_trace_event(
            "tool_loop_model_planner_skipped",
            iteration_index=iteration_index,
            reason="model_planner_adapter_not_enabled",
        )
        return None


def build_tool_loop_planner(state: AgentState) -> ToolLoopPlanner:
    """按配置选择工具循环 planner，并在模型不可用时回退规则 planner。"""
    # 读取 state.tool_loop_config 中的开关；不存在时使用默认开启模型、允许规则兜底。
    enable_model = bool(state.tool_loop_config.get("enable_model_planner", True))
    # fallback_to_rule_router 控制模型不可用时是否允许回到现有 ToolRouter。
    fallback_to_rule = bool(state.tool_loop_config.get("fallback_to_rule_router", True))
    # 当前本地环境不注入模型客户端，因此只记录模型不可用并返回规则 planner。
    if enable_model:
        model_decision = ModelToolLoopPlanner().try_decide(state, iteration_index=0)
        if model_decision is not None:
            return _StaticDecisionPlanner(model_decision)
    # 不允许规则兜底时返回一个只会 finish 的 planner，避免伪造工具计划。
    if not fallback_to_rule:
        return _FinishOnlyPlanner()
    # 默认使用规则 planner，保证本地测试和 main.py 可运行。
    return RuleBasedToolLoopPlanner()


class _StaticDecisionPlanner:
    """把单次模型决策包装成 planner；主要用于未来扩展和测试。"""

    def __init__(self, decision: ToolLoopDecision) -> None:
        """保存已校验的模型决策。"""
        self.decision = decision

    def decide(self, state: AgentState, *, iteration_index: int) -> ToolLoopDecision:
        """返回已保存决策；调用方仍会做预算和工具 guardrail。"""
        return self.decision


class _FinishOnlyPlanner:
    """不允许工具兜底时的安全 planner。"""

    def decide(self, state: AgentState, *, iteration_index: int) -> ToolLoopDecision:
        """直接结束工具循环，避免模型缺失时编造工具结果。"""
        return ToolLoopDecision(
            action="finish",
            finish_reason="planner_unavailable_and_rule_fallback_disabled",
            rationale_summary="模型 planner 不可用且禁用规则兜底，安全结束工具循环。",
            confidence=1.0,
        )
