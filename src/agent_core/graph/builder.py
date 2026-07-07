"""LangGraph builder adapter.

# 文件说明：
# - 本文件属于显式状态机层，负责把节点函数编排成 Agent 主链路。
# - LangGraph 可用时构建 StateGraph；不可用时使用等价 LocalGraph，保证本地测试稳定。
"""

from __future__ import annotations

from typing import Any

from agent_core.graph import nodes
from agent_core.graph.state import AgentNode, AgentState
from agent_core.memory.manager import MemoryManager


class LocalGraph:
    """本地状态机 runner，用于测试、CLI 和 LangGraph 不可用时的降级执行。"""

    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        """注入 MemoryManager，让短期记忆能跨同一个 WorkflowEngine 实例延续。"""
        # 保存同一个 WorkflowEngine 共享的 MemoryManager；这样 main.py 连续跑多条消息时能复用会话记忆。
        self.memory_manager = memory_manager

    def invoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        """按生产级主链路顺序执行节点，并在条件点做分支。"""
        # LangGraph 某些调用方式可能传入 dict；本地 runner 先统一恢复成 AgentState，避免下游节点处理两种类型。
        if isinstance(state, dict):
            state = AgentState(**state)

        # 初始化本轮请求的 trace、成本预算和第一条用户消息，这是整条 Agent 链路的起点。
        state = nodes.initialize_context(state)
        # 在读取记忆、做 RAG 或调用工具前先做输入安全检查，命中 Prompt Injection 时会直接进入 ERROR。
        state = nodes.input_guardrail(state)
        # 如果输入风控已经阻断，请求不再继续读取记忆或调用工具，避免越权内容污染后续上下文。
        if state.final_state == AgentNode.ERROR:
            return state

        # 读取 session/task/preference 三层记忆，把历史消息、任务状态和用户偏好写入 state.memory_context。
        state = nodes.restore_memory(state, self.memory_manager)
        # 把历史记忆里的消息与本轮输入合并成 normalized_messages，供后续上下文构建和模型生成使用。
        state = nodes.normalize_messages(state)
        # 识别用户本轮意图，并设置 intent 与 capability_route，决定后面走通用工具还是保险顾问 Skill。
        state = nodes.classify_intent(state)
        # 给请求打 low/medium/high 语义风险等级，后续工具权限、人审和输出风控都会复用该结果。
        state = nodes.semantic_risk_classification(state)
        # 从用户输入中抽取客户画像、公司实体、语言、融资主题等结构化槽位。
        state = nodes.extract_slots(state)
        # 校验关键槽位是否缺失；生产环境可在这里决定是否向用户追问。
        state = nodes.validate_slots(state)
        # 完成指代消解、时间解析、实体抽取、query rewrite 和检索 filters 生成。
        state = nodes.query_understanding(state)
        # 判断本轮到底需要 memory、RAG、tool、human、reject 还是 clarify，作为后续分支依据。
        state = nodes.context_need_planning(state)

        # 需要外部事实、计算、天气等能力时进入工具路径，而不是直接让模型凭空回答。
        if state.context_needs.get("tool"):
            # 根据 intent 和 query_understanding 选择具体工具，并生成 ToolCall 计划。
            state = nodes.general_tool_routing(state)
            # 执行工具计划；工具权限、人审和结果摘要会写入 state.tool_calls / state.tool_results。
            state = nodes.general_tool_call(state)
            # 如果工具调用涉及高风险副作用并触发人工审批，就在 HUMAN_APPROVAL 状态停住等待用户确认。
            if state.current_state == AgentNode.HUMAN_APPROVAL:
                return state
            # 校验工具结果是否成功、是否需要重试或降级，防止坏工具结果直接进入回答生成。
            state = nodes.verify_tool_result(state)
        # 保险销售、客户沟通、KYC、异议处理等请求进入领域 Skill 路径，检索销售实战语料。
        elif state.capability_route == "domain":
            # 明确当前领域工作流，例如 insurance_advisor，后续检索和提示词按 Skill 组织。
            state = nodes.route_domain_workflow(state)
            # 从销售实战知识库中检索破冰、KYC、异议处理等经验卡片，并生成销售洞察摘要。
            state = nodes.retrieve_sales_intelligence(state)
            # 把客户画像、销售洞察、历史上下文和当前问题组装成生成前的业务上下文。
            state = nodes.build_context(state)

        # 融合 memory、RAG、tool result 和 conversation，形成统一可信知识上下文。
        state = nodes.knowledge_fusion(state)
        # 按 token budget 压缩上下文，保留证据、槽位和关键对话，丢弃重复或低价值内容。
        state = nodes.compress_context(state)
        # 组装最终发送给模型的 prompt，包括系统约束、历史、检索证据、工具结果和用户问题。
        state = nodes.prompt_assembly(state)
        # 根据风险、预算和任务复杂度选择模型；本地 demo 使用可测试的占位模型名。
        state = nodes.model_routing(state)
        # 基于已组装上下文生成初版回答；工具类问题优先使用工具结果，销售类问题使用洞察摘要。
        state = nodes.generate_response(state)
        # 校验回答是否有工具、检索或上下文证据支撑，降低幻觉和不一致风险。
        state = nodes.grounding_verification(state)
        # 输出前最后做合规审查；触发高风险承诺、违规销售话术时会进入人工审批。
        state = nodes.compliance_review(state)
        # 如果输出风控要求人工确认，就停止在 HUMAN_APPROVAL，避免自动返回高风险内容。
        if state.current_state == AgentNode.HUMAN_APPROVAL:
            return state

        # 封装前端可消费的响应包：answer、citations、tool_cards、next_actions 和 trace_id。
        state = nodes.response_packaging(state)
        # 把本轮用户问题、助手回答和槽位更新写入短期 session/task memory。
        state = nodes.update_short_term_memory(state, self.memory_manager)
        # 判断哪些用户偏好或画像值得长期保存，并写入 preference memory 候选。
        state = nodes.long_term_memory_candidate(state, self.memory_manager)
        # 完成 trace、成本和最终状态收尾，正常情况下将状态推进到 FINAL。
        state = nodes.trace_finalize(state)
        # 返回完整 AgentState，调用方可以读取 answer、state_transitions、trace_events 等审计信息。
        return state


def build_agent_graph(memory_manager: MemoryManager | None = None) -> Any:
    """构建 LangGraph 图；如果运行时不支持则返回 LocalGraph。

    这条图对应项目的真实主链路：
    初始化 → 输入风控 → 记忆恢复 → 消息标准化 → 意图/风险/槽位/Query Understanding →
    Context Need → Tool 或 Domain RAG → 融合 → 压缩 → Prompt → 模型路由 → 生成 →
    Grounding → 输出风控 → 响应封装 → 记忆更新 → Trace 收尾。
    """
    try:
        # 优先尝试使用真实 LangGraph；如果依赖未安装或运行时不兼容，下面会降级到 LocalGraph。
        from langgraph.graph import END, StateGraph

        # StateGraph 的状态类型就是 AgentState，保证每个 LangGraph node 都读写同一个显式状态对象。
        graph = StateGraph(AgentState)
        # 注册上下文初始化节点：负责 trace、预算和用户消息初始化。
        graph.add_node("initialize_context", nodes.initialize_context)
        # 注册输入风控节点：在任何记忆/检索/工具之前阻断 Prompt Injection。
        graph.add_node("input_guardrail", nodes.input_guardrail)
        # 注册记忆恢复节点：通过闭包注入 memory_manager，读取三层 memory。
        graph.add_node("restore_memory", lambda state: nodes.restore_memory(state, memory_manager))
        # 注册消息标准化节点：合并历史消息和本轮输入。
        graph.add_node("normalize_messages", nodes.normalize_messages)
        # 注册意图识别节点：确定 intent 和 capability_route。
        graph.add_node("classify_intent", nodes.classify_intent)
        # 注册语义风险分级节点：给工具、人审和输出风控提供风险等级。
        graph.add_node("semantic_risk_classification", nodes.semantic_risk_classification)
        # 注册槽位抽取节点：抽出客户、公司、语言、时间等结构化变量。
        graph.add_node("extract_slots", nodes.extract_slots)
        # 注册槽位校验节点：判断是否需要补问或澄清。
        graph.add_node("validate_slots", nodes.validate_slots)
        # 注册 Query Understanding 节点：做指代消解、时间解析、改写 query 和 filters。
        graph.add_node("query_understanding", nodes.query_understanding)
        # 注册 Context Need 节点：判断是否需要工具、RAG、人审、拒答或澄清。
        graph.add_node("context_need_planning", nodes.context_need_planning)
        # 注册通用工具路由节点：选择天气、新闻、搜索、计算器等工具。
        graph.add_node("general_tool_routing", nodes.general_tool_routing)
        # 注册工具执行节点：真正调用工具并写入 tool_calls/tool_results。
        graph.add_node("general_tool_call", nodes.general_tool_call)
        # 注册工具结果校验节点：检查工具是否成功以及是否需要恢复。
        graph.add_node("verify_tool_result", nodes.verify_tool_result)
        # 注册领域工作流路由节点：进入 insurance_advisor 等业务 Skill。
        graph.add_node("route_domain_workflow", nodes.route_domain_workflow)
        # 注册销售洞察检索节点：从销售实战库检索经验卡片。
        graph.add_node("retrieve_sales_intelligence", nodes.retrieve_sales_intelligence)
        # 注册上下文构建节点：把画像、洞察、记忆和当前问题组织成生成上下文。
        graph.add_node("build_context", nodes.build_context)
        # 注册知识融合节点：统一 memory、RAG、工具和对话来源。
        graph.add_node("knowledge_fusion", nodes.knowledge_fusion)
        # 注册上下文压缩节点：按 token budget 控制上下文长度。
        graph.add_node("compress_context", nodes.compress_context)
        # 注册 Prompt 组装节点：生成最终给模型的 assembled_prompt。
        graph.add_node("prompt_assembly", nodes.prompt_assembly)
        # 注册模型路由节点：选择本轮推理使用的模型档位。
        graph.add_node("model_routing", nodes.model_routing)
        # 注册回答生成节点：生成初版 answer。
        graph.add_node("generate_response", nodes.generate_response)
        # 注册事实校验节点：检查回答是否有证据支撑。
        graph.add_node("grounding_verification", nodes.grounding_verification)
        # 注册输出合规节点：检查销售承诺、敏感信息和 prompt 泄露风险。
        graph.add_node("compliance_review", nodes.compliance_review)
        # 注册响应封装节点：形成前端/API 可直接展示的 response_package。
        graph.add_node("response_packaging", nodes.response_packaging)
        # 注册短期记忆更新节点：通过闭包注入 memory_manager，把本轮对话写入 session/task memory。
        graph.add_node("update_short_term_memory", lambda state: nodes.update_short_term_memory(state, memory_manager))
        # 注册长期记忆候选节点：通过闭包注入 memory_manager，保存稳定画像或偏好。
        graph.add_node(
            "long_term_memory_candidate",
            lambda state: nodes.long_term_memory_candidate(state, memory_manager),
        )
        # 注册 Trace 收尾节点：汇总成本、状态和最终审计事件。
        graph.add_node("trace_finalize", nodes.trace_finalize)

        # LangGraph 入口固定为 initialize_context，确保每次请求都先建立 trace 与预算。
        graph.set_entry_point("initialize_context")
        # 初始化完成后必须先做输入风控，避免恶意输入进入记忆、检索或工具层。
        graph.add_edge("initialize_context", "input_guardrail")
        # 输入风控后根据是否被阻断分流：被阻断则结束，否则继续恢复记忆。
        graph.add_conditional_edges(
            "input_guardrail",
            lambda state: "error" if state.final_state == AgentNode.ERROR else "continue",
            {"error": END, "continue": "restore_memory"},
        )
        # 记忆恢复后标准化消息，保证历史上下文以统一格式进入后续节点。
        graph.add_edge("restore_memory", "normalize_messages")
        # 消息标准化后识别意图，因为 intent 需要结合当前输入和历史上下文。
        graph.add_edge("normalize_messages", "classify_intent")
        # 意图识别后立即做风险分级，为后续工具权限与输出策略提供依据。
        graph.add_edge("classify_intent", "semantic_risk_classification")
        # 风险分级后抽取槽位，让 Query Understanding 和工具规划可以使用结构化参数。
        graph.add_edge("semantic_risk_classification", "extract_slots")
        # 槽位抽取后校验必要信息是否齐全。
        graph.add_edge("extract_slots", "validate_slots")
        # 槽位校验后进入 Query Understanding，生成可用于检索和工具调用的 query / filters。
        graph.add_edge("validate_slots", "query_understanding")
        # Query Understanding 完成后判断本轮到底需要哪些上下文来源或执行动作。
        graph.add_edge("query_understanding", "context_need_planning")
        # Context Need 是主分叉点：需要工具走工具链，需要业务能力走 Domain Skill，否则直接融合。
        graph.add_conditional_edges(
            "context_need_planning",
            _route_after_context_need,
            {
                "tool": "general_tool_routing",
                "domain": "route_domain_workflow",
                "fusion": "knowledge_fusion",
            },
        )
        # 工具路由后进入工具执行，严格按 ToolCall 计划调用白名单工具。
        graph.add_edge("general_tool_routing", "general_tool_call")
        # 工具执行后如果触发人工审批则停在 END，否则进入工具结果校验。
        graph.add_conditional_edges(
            "general_tool_call",
            lambda state: "human" if state.current_state == AgentNode.HUMAN_APPROVAL else "verify",
            {"human": END, "verify": "verify_tool_result"},
        )
        # 工具结果校验通过后进入知识融合，把工具事实并入最终上下文。
        graph.add_edge("verify_tool_result", "knowledge_fusion")
        # Domain 路径先明确业务 Skill，再检索销售洞察。
        graph.add_edge("route_domain_workflow", "retrieve_sales_intelligence")
        # 销售洞察检索后构建业务上下文，把洞察卡片压缩成可生成材料。
        graph.add_edge("retrieve_sales_intelligence", "build_context")
        # 业务上下文构建后进入知识融合，与 memory/conversation 一起统一排序。
        graph.add_edge("build_context", "knowledge_fusion")
        # 知识融合后压缩上下文，避免 prompt 超预算。
        graph.add_edge("knowledge_fusion", "compress_context")
        # 压缩后的上下文用于组装最终 prompt。
        graph.add_edge("compress_context", "prompt_assembly")
        # Prompt 组装后选择模型，生产环境可按成本、风险和时延动态路由。
        graph.add_edge("prompt_assembly", "model_routing")
        # 模型确定后生成回答。
        graph.add_edge("model_routing", "generate_response")
        # 初版回答生成后做事实校验，减少幻觉。
        graph.add_edge("generate_response", "grounding_verification")
        # 事实校验后做输出合规审查。
        graph.add_edge("grounding_verification", "compliance_review")
        # 合规审查可选择人工审批或继续响应封装。
        graph.add_conditional_edges(
            "compliance_review",
            lambda state: "human" if state.current_state == AgentNode.HUMAN_APPROVAL else "package",
            {"human": END, "package": "response_packaging"},
        )
        # 响应封装后更新短期记忆，保证下一轮对话能承接当前状态。
        graph.add_edge("response_packaging", "update_short_term_memory")
        # 短期记忆更新后判断是否有长期记忆候选，例如稳定偏好或客户画像。
        graph.add_edge("update_short_term_memory", "long_term_memory_candidate")
        # 长期记忆处理后做 trace 收尾。
        graph.add_edge("long_term_memory_candidate", "trace_finalize")
        # trace_finalize 是正常执行的最后一个节点。
        graph.add_edge("trace_finalize", END)
        # 返回可执行图对象，供 WorkflowEngine 调用 invoke。
        return graph.compile()
    except Exception:
        # 本地没有安装 LangGraph 时走 LocalGraph；它执行同样的节点顺序，方便离线测试和面试演示。
        return LocalGraph(memory_manager)


def _route_after_context_need(state: AgentState) -> str:
    """根据 Context Need 结果选择工具、业务 Skill 或直接融合。"""
    # context_needs.tool 为真时走通用工具路径，例如天气、搜索、新闻、计算器。
    if state.context_needs.get("tool"):
        return "tool"
    # 领域请求走 Domain Skill 路径，例如保险顾问的 KYC、破冰和异议处理。
    if state.capability_route == "domain":
        return "domain"
    # 不需要工具或领域检索时，直接进入知识融合并生成回答。
    return "fusion"
