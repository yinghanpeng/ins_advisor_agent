"""Workflow engine facade."""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

from agent_core.graph.builder import build_agent_graph
from agent_core.graph import nodes
from agent_core.graph.checkpoints import InMemoryCheckpointStore
from agent_core.graph.state import AgentState
from agent_core.guardrails.human_approval import ApprovalDecision, InMemoryApprovalStore
from agent_core.memory.business_store import BusinessMemoryStore, InMemoryBusinessMemoryStore
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
        business_store: BusinessMemoryStore | None = None,
        approval_store: InMemoryApprovalStore | None = None,
        checkpoint_store: InMemoryCheckpointStore | None = None,
    ) -> None:
        """初始化日志、LangSmith adapter 和本地兼容 graph。"""
        # 创建结构化日志器；如果调用方没有传入，就使用本地 stdout JSON logger。
        self.log = log or StructuredLogger()
        # 从环境变量初始化 LangSmith；没有配置时 adapter 会自动降级，不影响本地运行。
        self.langsmith = langsmith or LangSmithAdapter.from_env(self.log)
        # 创建共享 MemoryManager；同一个 WorkflowEngine 实例内的多轮对话会复用这份内存存储。
        self.memory_manager = memory_manager or MemoryManager()
        # 业务记忆 store 独立于 session/task/preference 记忆；默认内存实现保证本地 demo 和测试无需数据库。
        self.business_store = business_store or InMemoryBusinessMemoryStore()
        # 人工审批 store 保存待审批事项；生产环境应注入 PostgreSQL-backed store。
        self.approval_store = approval_store or InMemoryApprovalStore()
        # checkpoint store 保存审批前 AgentState，审批恢复必须从 checkpoint 继续，而不是重跑用户请求。
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
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
        # 默认走通用 Agent 图；显式指定 KYC 教练 workflow 时，执行业务记忆链路。
        if request.workflow_name == "insurance_kyc_coach_workflow":
            result = self._run_insurance_kyc_coach(state)
        else:
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
        if result.current_state == nodes.AgentNode.HUMAN_APPROVAL:
            self.checkpoint_store.save(result)
        return self._response_from_state(result)

    def resume_from_approval(self, approval_id: str, decision: ApprovalDecision) -> AgentRunResponse:
        """从人工审批恢复 workflow。

        Human Approval 需要 checkpoint 的原因是：审批期间外部世界可能已经变化，直接重跑用户原始
        请求会重复调用工具、重复写记忆或生成不一致结果。checkpoint 让恢复点精确落在审批触发处。
        """
        request = self.approval_store.get_request(approval_id)
        if decision.approval_id != approval_id:
            raise ValueError("decision.approval_id 与 approval_id 不一致")
        saved_decision = self.approval_store.decide(decision)
        state = self.checkpoint_store.get(request.checkpoint_id) or self.checkpoint_store.get(request.trace_id)
        if state is None:
            raise RuntimeError(f"未找到审批 checkpoint：{request.checkpoint_id}")

        state.add_trace_event(
            "human_approval_decision",
            approval_id=approval_id,
            decision=saved_decision.decision,
            reviewer=saved_decision.reviewer,
            pending_action=request.pending_action,
        )
        if saved_decision.decision == "rejected":
            state.answer = "该高风险动作未通过人工审批，已为你停止执行。"
            state.move_to(nodes.AgentNode.FINAL, reason="人工审批拒绝，进入安全结束。")
            return self._response_from_state(state)

        if saved_decision.modified_payload is not None:
            state.metadata["approval_modified_payload"] = saved_decision.modified_payload
        state.metadata["approval_id"] = approval_id
        state.metadata["approval_status"] = saved_decision.decision
        state.move_to(nodes.AgentNode.GENERATE_RESPONSE, reason="人工审批通过或修改后恢复执行。")

        if request.pending_action == "final_response":
            nodes.response_packaging(state)
            nodes.trace_finalize(state)
        else:
            state.answer = state.answer or "人工审批已完成，当前动作已恢复到安全执行点。"
            state.move_to(nodes.AgentNode.FINAL, reason="审批恢复动作完成。")
        return self._response_from_state(state)

    def _response_from_state(self, state: AgentState) -> AgentRunResponse:
        """把内部 AgentState 封装成外部响应契约。"""
        return AgentRunResponse(
            trace_id=state.trace_id,
            session_id=state.session_id,
            final_state=state.final_state.value if state.final_state else state.current_state.value,
            answer=state.answer or "",
            intent=state.intent,
            domain_skill=state.domain_skill,
            guardrails=state.guardrail_results,
            retrieved_context=state.retrieved_context,
            trace_events=state.trace_events,
            state_transitions=state.state_transitions,
            tool_calls=state.tool_calls,
            tool_results=state.tool_results,
            query_understanding=state.query_understanding,
            context_needs=state.context_needs,
            response_package=state.response_package,
            grounding_result=state.grounding_result,
            cost=state.cost,
        )

    def _run_insurance_kyc_coach(self, state: AgentState) -> AgentState:
        """执行保险 KYC 教练业务记忆链路。

        这条链路对齐 Dify 4 轮 KYC workflow，并通过显式节点把分析、写入提案、
        模式检索、compact_context 和策略生成拆开，便于后续逐节点接入配置化模型。
        """
        state.workflow_name = "insurance_kyc_coach_workflow"
        state.domain_skill = state.domain_skill or "insurance_advisor"
        state.metadata.setdefault("workflow_version", "local-kyc-v1")

        nodes.initialize_context(state)
        nodes.input_guardrail(state)
        if state.final_state:
            return state

        nodes.load_business_memory(state, self.business_store)
        nodes.analyze_kyc_and_route(state)
        nodes.propose_memory_writes(state)
        nodes.validate_memory_writes(state)
        nodes.persist_memory_snapshot(state, self.business_store)
        nodes.build_compact_context_node(state, self.business_store)
        nodes.status_router(state)

        if state.current_state == nodes.AgentNode.GENERATE_KYC_QUESTIONS:
            nodes.generate_kyc_questions(state)
        elif state.current_state == nodes.AgentNode.RETRIEVE_DIALOGUE_PATTERNS:
            nodes.retrieve_dialogue_patterns_node(state)
            nodes.retrieve_external_context_if_needed_node(state)
            nodes.build_compact_context_node(state, self.business_store)
            nodes.generate_strategy_node(state)
        else:
            nodes.generate_strategy_node(state)

        nodes.compliance_review(state)
        if state.current_state != nodes.AgentNode.HUMAN_APPROVAL:
            nodes.response_packaging(state)
            nodes.post_response_logger_node(state, self.business_store)
            nodes.trace_finalize(state)
        return state
