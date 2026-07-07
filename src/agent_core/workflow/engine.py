"""Workflow engine facade."""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

from agent_core.graph.builder import build_agent_graph
from agent_core.graph.state import AgentState
from agent_core.memory.manager import MemoryManager
from agent_core.observability.langsmith_client import LangSmithAdapter
from agent_core.observability.logger import StructuredLogger
from agent_core.workflow.contracts import AgentRunRequest, AgentRunResponse


class WorkflowEngine:
    """统一执行 Agent workflow，是 main、API、Dify 调用的共同入口。"""

    def __init__(
        self,
        log: StructuredLogger | None = None,
        langsmith: LangSmithAdapter | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        """初始化日志、LangSmith adapter 和本地兼容 graph。"""
        # 创建结构化日志器；如果调用方没有传入，就使用本地 stdout JSON logger。
        self.log = log or StructuredLogger()
        # 从环境变量初始化 LangSmith；没有配置时 adapter 会自动降级，不影响本地运行。
        self.langsmith = langsmith or LangSmithAdapter.from_env(self.log)
        # 创建共享 MemoryManager；同一个 WorkflowEngine 实例内的多轮对话会复用这份内存存储。
        self.memory_manager = memory_manager or MemoryManager()
        # 构建 Agent 执行图；LangGraph 可用时返回 StateGraph，不可用时返回等价 LocalGraph。
        self.graph = build_agent_graph(self.memory_manager)

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        """执行一次 Agent 请求，并返回包含状态链路和 trace 的结构化响应。"""
        # 把 API/CLI/Dify 传入的请求契约转换成 AgentState；后续所有节点都只读写这个显式状态对象。
        state = AgentState(
            # 会话 ID 用于读取和更新短期记忆，也是多轮对话能接上的关键索引。
            session_id=request.session_id,
            # 用户 ID 用于读取偏好记忆；匿名用户可以为空，此时会退回使用 session_id。
            user_id=request.user_id,
            # 租户 ID 用于隔离不同团队、机构或渠道的记忆和知识库。
            tenant_id=request.tenant_id,
            # 用户原始输入会进入意图识别、Query Understanding、工具规划和回答生成。
            input_text=request.input,
            # workflow_name 标识当前跑哪条工作流，便于 Dify 映射、日志过滤和评估统计。
            workflow_name=request.workflow_name,
            # domain_skill 可由调用方指定；为空时由 classify_intent / route_domain_workflow 自动判断。
            domain_skill=request.domain_skill,
            # metadata 保存调用端、调试开关、实验分组等扩展信息，不污染核心字段。
            metadata=request.metadata,
        )
        # 记录请求开始事件；trace_id 从这里开始贯穿状态迁移、工具调用、检索和最终响应。
        self.log.event(
            "agent_run_started",
            trace_id=state.trace_id,
            session_id=state.session_id,
            workflow_name=state.workflow_name,
        )
        # graph.invoke 是唯一运行入口；main、FastAPI、Dify webhook 都复用同一条主链路，避免多套流程分叉。
        result = self.graph.invoke(state)
        # 兼容 LangGraph 返回 dict 的情况；统一恢复成 AgentState 后，响应封装逻辑就不用关心底层图实现。
        if isinstance(result, dict):
            result = AgentState(**result)
        # 将 state_transitions 单独写日志；这类日志只描述状态从哪里跳到哪里，适合排查链路卡点。
        for transition in result.state_transitions:
            self.log.event("state_transition", **transition)
        # 将 trace_events 写入结构化日志；这里包含工具、RAG、风控、错误、成本等更细粒度事件。
        for event in result.trace_events:
            # trace event 内部也有 event 字段，写 logger 前改名，避免和 StructuredLogger.event 的参数名冲突。
            event_payload = dict(event)
            # 保留原始事件名到 trace_event_name，方便日志平台按事件类型检索。
            event_payload["trace_event_name"] = event_payload.pop("event", "unknown")
            # 输出统一的 trace_event 日志，具体事件名放在 payload 中。
            self.log.event("trace_event", **event_payload)
        # 记录请求结束事件，给日志平台一个快速统计 final_state / intent 的入口。
        self.log.event(
            "agent_run_finished",
            trace_id=result.trace_id,
            final_state=result.final_state.value if result.final_state else result.current_state.value,
            intent=result.intent,
        )
        # 把内部 AgentState 封装成外部响应契约；API、CLI、Dify 都只依赖这个稳定结构。
        return AgentRunResponse(
            # trace_id 返回给调用方，用于在日志、LangSmith 或 eval 报告中定位同一次运行。
            trace_id=result.trace_id,
            # session_id 回传给前端，方便下一轮对话继续使用同一个 session。
            session_id=result.session_id,
            # final_state 说明本轮是正常 FINAL、ERROR，还是停在 HUMAN_APPROVAL。
            final_state=result.final_state.value if result.final_state else result.current_state.value,
            # answer 是最终可展示文本；如果中途被阻断，也会放入阻断说明。
            answer=result.answer or "",
            # intent 让调用方知道本轮被识别成天气、搜索、保险顾问还是普通对话。
            intent=result.intent,
            # domain_skill 说明是否命中了 insurance_advisor 等业务 Skill。
            domain_skill=result.domain_skill,
            # guardrails 暴露输入/输出/工具风控结果，方便审计和前端提示。
            guardrails=result.guardrail_results,
            # retrieved_context 返回本轮用到的检索证据摘要，便于解释回答来源。
            retrieved_context=result.retrieved_context,
            # trace_events 返回完整事件流，本地 main.py 可以直接打印成链路回放。
            trace_events=result.trace_events,
            # state_transitions 只返回状态迁移路径，不混入工具或检索细节。
            state_transitions=result.state_transitions,
            # tool_calls 返回工具调用审计记录，例如工具名、入参、状态和错误。
            tool_calls=result.tool_calls,
            # tool_results 返回工具输出的结构化结果，前端可展示成 tool card。
            tool_results=result.tool_results,
            # query_understanding 展示指代消解、时间解析、改写 query 和 filters。
            query_understanding=result.query_understanding,
            # context_needs 展示本轮为什么需要或不需要 memory、RAG、tool、human。
            context_needs=result.context_needs,
            # response_package 是更接近前端组件的数据包，包含引用、工具卡片和下一步建议。
            response_package=result.response_package,
            # grounding_result 展示事实校验结论，说明回答是否有证据支撑。
            grounding_result=result.grounding_result,
            # cost 汇总 token budget、工具次数、上下文长度等成本信息。
            cost=result.cost,
        )
