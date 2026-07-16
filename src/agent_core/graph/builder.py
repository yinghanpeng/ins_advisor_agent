"""基于 LangGraph Graph API 的 Agent 总控工作流。

本模块只迁移原 ``AgentGraph`` 的编排方式：业务函数、调用顺序、同步入口、条件判断和
终止语义保持不变。``WorkflowEngine`` 仍调用 ``AgentGraph.invoke``，因此 API、CLI、Dify
适配器及测试不需要改变输入输出契约。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_core.agents.bootstrap import build_default_domain_agent_registry
from agent_core.agents.insurance_proposal.port import ProposalAgentPort
from agent_core.agents.registry import DomainAgentRegistry
from agent_core.config.runtime import load_runtime_settings
from agent_core.graph import nodes
from agent_core.graph.state import AgentGraphState, AgentNode, AgentState
from agent_core.intents.router import IntentRouter, build_intent_router
from agent_core.memory.business_store import BusinessMemoryStore
from agent_core.memory.manager import MemoryBackend
from agent_core.skills.insurance_advisor.kyc import InsuranceKycExtractor
from agent_core.skills.insurance_advisor.knowledge import (
    InsuranceKnowledgeProvider,
    LocalInsuranceKnowledgeProvider,
)


# 通用主流程实际节点数超过 LangGraph 默认递归步数；100 只放宽框架执行上限，不改变任何业务重试次数。
GRAPH_RECURSION_LIMIT = 100

# 适配器只接收并返回现有 AgentState；依赖仍由 AgentGraph 构造阶段注入闭包，不进入共享状态。
StateHandler = Callable[[AgentState], AgentState]


class AgentGraph:
    """保留原同步 ``invoke`` 接口、内部使用已编译 ``StateGraph`` 的总控执行器。"""

    def __init__(
        self,
        memory_manager: MemoryBackend | None = None,
        business_store: BusinessMemoryStore | None = None,
        intent_router: IntentRouter | None = None,
        kyc_extractor: InsuranceKycExtractor | None = None,
        insurance_knowledge_provider: InsuranceKnowledgeProvider | None = None,
        insurance_news_enabled: bool | None = None,
        domain_agent_registry: DomainAgentRegistry | None = None,
        proposal_agent: ProposalAgentPort | None = None,
    ) -> None:
        """注入现有运行时依赖，并在不发起业务请求的前提下编译总控图。"""

        # Session/Preference 记忆管理器继续由 WorkflowEngine 持有并注入，避免改变连接生命周期。
        self.memory_manager = memory_manager
        # 保险业务 Store 仍由领域 Agent 使用，Graph 构建阶段不会读取或写入数据库。
        self.business_store = business_store
        # 意图 Router 继续在 Engine 生命周期内复用模型客户端和配置。
        self.intent_router = intent_router or build_intent_router()
        # KYC 抽取器仍只负责抽取事实，不参与 Graph 路由判断。
        self.kyc_extractor = kyc_extractor or InsuranceKycExtractor()
        # 未注入生产知识 Provider 时继续使用原本地空实现，不新增虚假知识降级。
        self.insurance_knowledge_provider = (
            insurance_knowledge_provider or LocalInsuranceKnowledgeProvider()
        )
        # 新闻权限开关继续来自现有配置，Graph 不改变外部工具授权边界。
        self.insurance_news_enabled = (
            load_runtime_settings(
                os.getenv("CONFIG_DIR", "configs")
            ).insurance_knowledge.news_enabled
            if insurance_news_enabled is None
            else insurance_news_enabled
        )
        # 完整 Registry 与单独计划书 Agent 仍互斥，保留原启动期错误类型和错误文本。
        if domain_agent_registry is not None and proposal_agent is not None:
            # 同时传入两种装配入口会造成替换语义不明确，因此继续使用原 ValueError 阻断启动。
            raise ValueError("domain_agent_registry 与 proposal_agent 不能同时传入")
        # 调用方显式提供 Registry 时不追加或覆盖任何专业 Agent。
        if domain_agent_registry is not None:
            # 保存调用方完整注册表，禁止 Graph 构建阶段偷偷增加默认 Agent。
            self.domain_agent_registry = domain_agent_registry
        # 未提供完整注册表时才按原 Runtime 依赖创建默认 Agent 集合。
        else:
            # 默认装配继续共享原有 Memory、Store、Router、KYC 和知识 Provider 实例。
            self.domain_agent_registry = build_default_domain_agent_registry(
                memory_manager=self.memory_manager,
                business_store=self.business_store,
                intent_router=self.intent_router,
                kyc_extractor=self.kyc_extractor,
                insurance_knowledge_provider=self.insurance_knowledge_provider,
                insurance_news_enabled=self.insurance_news_enabled,
                proposal_agent=proposal_agent,
            )
        # Graph 构建只注册 Python 节点与边；真实业务副作用只会在 invoke 后发生。
        self.compiled_graph = self._build_graph()

    @staticmethod
    def _apply_node(graph_state: AgentGraphState, handler: StateHandler) -> AgentGraphState:
        """调用一个原业务节点，并仅更新 Graph 的 ``agent_state`` 状态通道。"""

        # 原节点继续接收请求专属 Pydantic 状态，保证 PrivateAttr Trace Sink 与原地更新顺序不丢失。
        updated_state = handler(graph_state["agent_state"])
        # Graph 只有一个显式通道；返回该通道是最小更新，不复制或覆盖任何其它 Graph 字段。
        return {"agent_state": updated_state}

    def _initialize_context_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """初始化节点；读请求身份与输入，写 Trace/预算/消息；异常原样抛出，随后进入输入风控。

        本节点处理客户输入及会话标识等敏感信息，但不新增日志或外部调用。
        """

        # 仅返回初始化函数产生的请求级状态更新，下一节点由 Graph 普通边控制。
        return self._apply_node(graph_state, nodes.initialize_context)

    def _input_guardrail_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """输入风控节点；读原始输入，写风控与终态；异常语义不变，后续由条件边继续或结束。

        本节点可能识别客户 PII 和高风险保险请求，仍仅调用现有 Guardrail 实现。
        """

        # 仅返回风控节点更新，是否终止由后续纯路由函数判断。
        return self._apply_node(graph_state, nodes.input_guardrail)

    def _restore_memory_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """记忆恢复节点；读租户/会话/用户标识，写 memory_context；后续固定进入消息标准化。

        原 MemoryBackend 异常继续上抛；节点可能读取客户对话和偏好等敏感信息。
        """

        # 注入原 MemoryBackend 后执行恢复，不在 Graph State 中存放连接对象。
        return self._apply_node(
            graph_state,
            lambda state: nodes.restore_memory(state, self.memory_manager),
        )

    def _normalize_messages_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """消息标准化节点；读历史与本轮输入，写 normalized_messages；随后进入意图识别。

        节点处理客户对话正文，完全复用现有函数并保留其异常行为。
        """

        # 仅返回标准化消息所在的请求级状态，不复制客户对话正文。
        return self._apply_node(graph_state, nodes.normalize_messages)

    def _classify_intent_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """意图识别节点；读消息和活跃意图，写路由结果；条件边选择澄清或继续。

        节点可能读取保险沟通上下文；模型、规则、异常与降级均沿用原 Router。
        """

        # 注入原 IntentRouter 后执行分类，保持模型与规则降级路径不变。
        return self._apply_node(
            graph_state,
            lambda state: nodes.classify_intent(state, self.intent_router),
        )

    def _clarification_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """澄清节点；读意图或工具缺口，写澄清问题；随后按原分支封装并结束。

        节点可能使用客户场景信息，不改变现有问题文本、异常或隐私处理。
        """

        # 复用原澄清函数，具体分支的记忆写入差异由不同 Graph Edge 保留。
        return self._apply_node(graph_state, nodes.generate_clarification_response)

    def _semantic_risk_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """语义风险节点；读用户输入，写 risk_level；随后选择专业 Agent 或通用链路。

        节点处理保险风险表达，但继续使用原代码规则且不在路由函数中产生副作用。
        """

        # 只返回原风险分级更新，专业能力选择留给无副作用条件路由。
        return self._apply_node(graph_state, nodes.semantic_risk_classification)

    def _invoke_domain_agent_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """专业 Agent 调用节点；读 intent/domain_skill，写领域处理结果；完成后直接结束总控图。

        专业 Agent 自己负责保险 KYC、记忆和合规敏感数据；未找到已选 Agent 时明确抛错。
        """

        # 读取现有路由字段，不把专业 Agent 实例写入共享业务状态。
        state = graph_state["agent_state"]
        # 条件边和执行节点使用同一稳定 Registry；这里再次精确解析，不把 Agent 对象放进共享状态。
        selected_agent = self.domain_agent_registry.resolve(
            intent=state.intent,
            domain_skill=state.domain_skill,
        )
        # 正常情况下条件边已保证命中；若 Registry 被并发修改，显式失败比绕过领域链路更安全。
        if selected_agent is None:
            # Registry 结果失效属于运行时装配错误，继续抛出明确异常而不是改走通用回答。
            raise RuntimeError("Domain Agent 路由结果在执行前失效")
        # 专业 Agent 保留同步 invoke 语义，并返回同一 AgentState 契约。
        return {"agent_state": selected_agent.invoke(state)}

    def _domain_agent_unavailable_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """占位能力降级节点；读已声明 Agent，写原错误码与安全答复；随后封装、记忆并结束。

        本节点不访问客户业务数据；异常只可能来自 Registry 不一致，并保持明确失败。
        """

        # 读取已裁定领域与意图，只用于再次确认禁用占位能力的稳定描述。
        state = graph_state["agent_state"]
        # 只查声明而不检查 enabled，保持原“能力已占位但尚未接入”的识别语义。
        declared_agent = self.domain_agent_registry.find_declared(
            intent=state.intent,
            domain_skill=state.domain_skill,
        )
        # 条件边已确认存在声明；这里防御 Registry 被运行期并发修改的极端情况。
        if declared_agent is None:
            # 声明失效时继续显式失败，禁止静默落入通用路径并伪造专业能力结果。
            raise RuntimeError("已声明 Domain Agent 路由结果在执行前失效")
        # Trace 只记录稳定控制面字段，不记录客户输入或保险事实。
        state.add_trace_event(
            "domain_agent_unavailable",
            agent_id=declared_agent.descriptor.agent_id,
            available=False,
            execution_mode=declared_agent.descriptor.execution_mode,
        )
        # 保留原错误码和固定安全答复，不让禁用占位能力伪装成业务成功。
        state.errors.append(f"domain_agent_unavailable:{declared_agent.descriptor.agent_id}")
        # 固定文案沿用改造前实现，确保外部响应和测试断言不发生变化。
        state.answer = "该专业能力尚未接入，当前请求没有执行。"
        # 仅返回更新后的请求状态，后续封装与记忆节点由普通边显式执行。
        return {"agent_state": state}

    def _query_understanding_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """查询理解节点；读输入与记忆，写实体、时间和 filters；随后固定进入需求规划。

        节点可能处理客户实体与历史上下文；原规则、日期解析和异常语义保持不变。
        """

        # 只返回原查询理解结果，不在适配层调整实体、日期或过滤器。
        return self._apply_node(graph_state, nodes.query_understanding)

    def _context_need_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """上下文需求节点；读意图和查询理解，写 context_needs；条件边选择原四类路径。

        节点不新增敏感数据处理，仍由现有函数决定是否需要工具、领域检索或澄清。
        """

        # 只返回原需求计划，分支优先级由后续有限路由值显式表达。
        return self._apply_node(graph_state, nodes.context_need_planning)

    def _general_tool_routing_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """工具规划节点；读输入与查询理解，写 tool_plan/缺参状态；条件边选择澄清或执行。

        工具名、参数、Prompt 和异常均保持原样；节点可能处理查询中的客户业务信息。
        """

        # 只返回原工具计划和缺参标记，不在适配层推测或补齐参数。
        return self._apply_node(graph_state, nodes.general_tool_routing)

    def _general_tool_call_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """工具执行节点；读白名单计划，写调用与结果；条件边保留执行后澄清兼容分支。

        节点可能调用现有第三方只读接口；权限、超时、异常与脱敏逻辑均未改变。
        """

        # 调用原工具执行函数并返回其状态更新，权限和第三方逻辑保持封装边界。
        return self._apply_node(graph_state, nodes.general_tool_call)

    def _verify_tool_result_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """工具校验节点；读 tool_results，写错误、重试计数和降级答案；随后进入统一生成尾链。

        失败仍由原函数进入 RECOVERY 且不自动重试外部接口，不记录客户敏感结果。
        """

        # 只返回原校验、RECOVERY 和 retry_count 更新，不增加 Graph 级重试。
        return self._apply_node(graph_state, nodes.verify_tool_result)

    def _route_domain_workflow_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """兼容领域路由节点；读 domain_skill，写 sales_route；随后固定检索销售智能。

        节点不执行外部写操作，保持未来非保险领域路径的原有同步行为。
        """

        # 只返回兼容领域路由标签，后续检索顺序由普通边固定。
        return self._apply_node(graph_state, nodes.route_domain_workflow)

    def _retrieve_sales_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """销售智能检索节点；读领域查询，写已审核检索结果；随后进入上下文构建。

        节点可能处理脱敏客户场景；检索器、过滤规则、异常和数据源均未修改。
        """

        # 只返回原审核检索结果，不改变语料准入或排序规则。
        return self._apply_node(graph_state, nodes.retrieve_sales_intelligence)

    def _build_context_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """上下文构建节点；读检索结果，写销售摘要；随后进入知识融合。

        节点可能整理保险沟通材料，继续使用原实现且不改变 Prompt 内容。
        """

        # 只返回原上下文摘要，避免 Graph 适配层触碰 Prompt 或客户事实。
        return self._apply_node(graph_state, nodes.build_context)

    def _knowledge_fusion_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """知识融合节点；读记忆/检索/工具结果，写 knowledge_context；随后进入压缩。

        节点可能汇总客户敏感上下文，但不会新增日志或改变来源边界。
        """

        # 只返回原可信上下文融合结果，来源边界和冲突规则保持不变。
        return self._apply_node(graph_state, nodes.knowledge_fusion)

    def _compress_context_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """上下文压缩节点；读融合上下文，写 compressed_context；随后进入 Prompt 组装。

        节点可能处理客户与保险内容；压缩预算和截断规则保持原样。
        """

        # 只返回原压缩结果，字符和 token 预算不在迁移层重算。
        return self._apply_node(graph_state, nodes.compress_context)

    def _prompt_assembly_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """Prompt 组装节点；读压缩上下文，写 assembled_prompt；随后进入模型路由。

        Prompt 文本、消息结构和敏感上下文边界完全复用原函数，异常原样抛出。
        """

        # 只返回原 Prompt 结构，禁止 Graph 迁移改变任何提示文本。
        return self._apply_node(graph_state, nodes.prompt_assembly)

    def _model_routing_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """模型路由节点；读风险与预算，写 model_name；随后进入回答生成。

        模型名称和参数选择逻辑保持不变，本节点不读取额外客户数据。
        """

        # 只返回原模型选择结果，不覆盖模型名称或调用参数。
        return self._apply_node(graph_state, nodes.model_routing)

    def _generate_response_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """回答生成节点；读现有 Prompt/工具结果，写 answer；随后进入事实校验。

        模型调用、Prompt、工具优先规则和异常降级保持不变；可能处理客户保险上下文。
        """

        # 只返回原生成结果，模型、工具优先和降级逻辑继续由业务函数负责。
        return self._apply_node(graph_state, nodes.generate_response)

    def _grounding_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """事实校验节点；读答案与证据，写 grounding_result；后续顺序由显式普通边决定。

        节点处理保险事实与工具证据，但继续使用原校验规则和异常语义。
        """

        # 只返回原事实校验结果，初次与重生成后复核通过不同 Graph 节点复用本适配器。
        return self._apply_node(graph_state, nodes.grounding_verification)

    def _compliance_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """合规审查节点；读答案，写风控结果或安全答复；后续按原顺序继续。

        节点处理保险承诺、核保或理赔等敏感表达，规则与固定降级文案未修改。
        """

        # 只返回原合规审查或安全降级答案，不在 Graph 层增加新规则。
        return self._apply_node(graph_state, nodes.compliance_review)

    def _output_pii_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """输出 PII 节点；读候选答案，写脱敏答案和扫描摘要；后续按原顺序继续。

        节点可能识别身份、联系方式或银行卡信息，仍不记录原始敏感值。
        """

        # 只返回原 PII 脱敏更新，扫描类型与替换格式不变。
        return self._apply_node(graph_state, nodes.output_pii_scan)

    def _evaluate_response_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """质量评估节点；读答案/证据/风控，写 evaluation_result；随后固定进入有界重生成。

        本节点无外部副作用，不改变原评估条件、次数或客户信息处理边界。
        """

        # 只返回原确定性评估结果，节点本身不触发模型或外部接口。
        return self._apply_node(graph_state, nodes.evaluate_response_quality)

    def _regenerate_response_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """有界重生成节点；读评估结果，按原规则写答案和次数；随后再次执行三重审查。

        最多一次的预算、模型/工具调用语义和保险敏感信息处理保持不变。
        """

        # 只返回原有界重生成更新，最大次数仍取原状态字段。
        return self._apply_node(graph_state, nodes.regenerate_response_if_needed)

    def _response_packaging_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """响应封装节点；读最终状态字段，写原 response_package；随后进入对应分支收尾。

        客户安全 DTO、响应字段和异常格式保持不变，不额外暴露内部完整状态。
        """

        # 只返回原响应包更新，外部 DTO 仍由 WorkflowEngine 按既有字段投影。
        return self._apply_node(graph_state, nodes.response_packaging)

    def _short_term_memory_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """短期记忆节点；读本轮问答与会话标识，写原 Session 记忆；随后进入既定收尾节点。

        节点处理客户对话敏感数据；存储、CAS 冲突与异常逻辑完全复用原实现。
        """

        # 注入原 MemoryBackend 后写 Session，不在 Graph 层新增 checkpoint。
        return self._apply_node(
            graph_state,
            lambda state: nodes.update_short_term_memory(state, self.memory_manager),
        )

    def _long_term_memory_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """长期记忆候选节点；读回答与偏好信号，写候选/存储结果；随后进入 Trace 收尾。

        节点可能处理客户偏好；原同意、敏感性、去重和异常策略均保持不变。
        """

        # 注入原 MemoryBackend 后处理长期候选，保留 Consent 与去重逻辑。
        return self._apply_node(
            graph_state,
            lambda state: nodes.long_term_memory_candidate(state, self.memory_manager),
        )

    def _trace_finalize_node(self, graph_state: AgentGraphState) -> AgentGraphState:
        """Trace 收尾节点；读执行摘要，写成本与 FINAL 终态；随后通过普通边进入 END。

        节点不新增客户数据处理，保留原唯一正常终态和日志字段语义。
        """

        # 只返回原 FINAL 与成本摘要更新，随后由 Graph 普通边进入 END。
        return self._apply_node(graph_state, nodes.trace_finalize)

    @staticmethod
    def _route_after_input_guardrail(
        graph_state: AgentGraphState,
    ) -> Literal["continue", "finish"]:
        """输入风控纯路由：FINAL/ERROR 结束；其余状态继续恢复记忆，不修改 State。"""

        # finish：输入被阻断或同步安全降级已经形成终态，直接进入 END。
        if graph_state["agent_state"].final_state in {AgentNode.ERROR, AgentNode.FINAL}:
            # 返回有限路由值供 Conditional Edge 映射到 END，不写任何状态字段。
            return "finish"
        # continue：输入通过风控，下一节点为 restore_memory。
        return "continue"

    @staticmethod
    def _route_after_intent(
        graph_state: AgentGraphState,
    ) -> Literal["clarify", "continue"]:
        """意图纯路由：低置信/换题不明进入澄清；否则继续风险分类，不修改 State。"""

        # clarify：原 Router 明确要求补充信息，下一节点生成澄清问题并最终结束。
        if graph_state["agent_state"].context_needs.get("clarify"):
            # 返回澄清标识，不在路由函数中生成问题或写会话记忆。
            return "clarify"
        # continue：意图足够明确，下一节点为 semantic_risk_classification。
        return "continue"

    def _route_after_semantic_risk(
        self,
        graph_state: AgentGraphState,
    ) -> Literal["domain_agent", "domain_unavailable", "general"]:
        """专业能力纯路由：已启用 Agent、禁用占位或通用链路；只查询 Registry，不修改 State。"""

        # 读取稳定领域与意图字段，路由判断不保存 Registry 查询结果。
        state = graph_state["agent_state"]
        # domain_agent：领域与意图精确命中已启用 Agent，下一节点同步调用该 Agent 并结束总控图。
        if self.domain_agent_registry.resolve(
            intent=state.intent,
            domain_skill=state.domain_skill,
        ) is not None:
            # 返回已启用专业 Agent 标识，下一 Graph 节点才执行真实领域业务。
            return "domain_agent"
        # domain_unavailable：能力有声明但未启用，下一节点返回原固定安全说明后正常结束。
        if self.domain_agent_registry.find_declared(
            intent=state.intent,
            domain_skill=state.domain_skill,
        ) is not None:
            # 返回禁用占位标识，下一 Graph 节点生成原固定安全响应。
            return "domain_unavailable"
        # general：未命中专业能力，下一节点为通用 Query Understanding。
        return "general"

    @staticmethod
    def _route_after_context_need(
        graph_state: AgentGraphState,
    ) -> Literal["clarify", "tool", "domain", "direct"]:
        """上下文纯路由：按原优先级选择澄清、工具、兼容领域或直达生成，不修改 State。"""

        # 只读取原 planner 输出和 capability_route，保持 if/elif 的既有优先级。
        state = graph_state["agent_state"]
        # clarify：planner 明确缺信息，下一节点生成澄清并结束，优先级高于其它能力。
        if state.context_needs.get("clarify"):
            # 返回澄清标识，不调用问题生成、工具或数据库。
            return "clarify"
        # tool：需要外部或本地工具，下一节点只负责工具规划。
        if state.context_needs.get("tool"):
            # 返回工具标识，真实工具调用至少在两个后续节点之后发生。
            return "tool"
        # domain：未来非保险领域 Skill 的兼容路径，下一节点执行现有领域路由。
        if state.capability_route == "domain":
            # 返回兼容领域标识，不在纯路由中检索销售语料。
            return "domain"
        # direct：无需工具或领域检索，直接进入知识融合，不提前结束流程。
        return "direct"

    @staticmethod
    def _route_after_tool_stage(
        graph_state: AgentGraphState,
    ) -> Literal["clarify", "continue"]:
        """工具阶段纯路由：缺参进入澄清；否则继续调用或校验，不修改 State。"""

        # clarify：工具 Schema 或兼容 planner 要求补参，下一节点生成澄清并结束。
        if graph_state["agent_state"].context_needs.get("clarify"):
            # 返回澄清标识，确保缺参时不会执行真实工具。
            return "clarify"
        # continue：工具计划或执行结果可继续，按所在条件边进入调用或校验节点。
        return "continue"

    def _build_graph(self) -> CompiledStateGraph:
        """注册总控 Node/Edge/Conditional Edge，并返回未配置持久化的编译图。"""

        # AgentGraphState 只承载现有显式 AgentState，避免 Graph 迁移产生第二套业务字段。
        builder = StateGraph(AgentGraphState)

        # 节点规格按业务阶段分组；同一适配器可在不同分支位置注册，以保留原收尾差异。
        node_specs = (
            # 公共入口严格保持初始化、输入风控、记忆、消息、意图和风险的原顺序。
            ("initialize_context", self._initialize_context_node),
            ("input_guardrail", self._input_guardrail_node),
            ("restore_memory", self._restore_memory_node),
            ("normalize_messages", self._normalize_messages_node),
            ("classify_intent", self._classify_intent_node),
            ("semantic_risk_classification", self._semantic_risk_node),
            # 四处澄清分别注册，避免把是否写短期记忆变成新的隐藏状态字段。
            ("clarify_after_intent", self._clarification_node),
            ("package_after_intent_clarify", self._response_packaging_node),
            ("memory_after_intent_clarify", self._short_term_memory_node),
            ("finalize_after_intent_clarify", self._trace_finalize_node),
            ("clarify_after_context", self._clarification_node),
            ("package_after_context_clarify", self._response_packaging_node),
            ("finalize_after_context_clarify", self._trace_finalize_node),
            ("clarify_after_tool_routing", self._clarification_node),
            ("package_after_tool_routing_clarify", self._response_packaging_node),
            ("finalize_after_tool_routing_clarify", self._trace_finalize_node),
            ("clarify_after_tool_call", self._clarification_node),
            ("package_after_tool_call_clarify", self._response_packaging_node),
            ("finalize_after_tool_call_clarify", self._trace_finalize_node),
            # 专业 Agent 与禁用占位能力继续在语义风险节点之后分流。
            ("invoke_domain_agent", self._invoke_domain_agent_node),
            ("domain_agent_unavailable", self._domain_agent_unavailable_node),
            ("package_domain_unavailable", self._response_packaging_node),
            ("memory_domain_unavailable", self._short_term_memory_node),
            ("finalize_domain_unavailable", self._trace_finalize_node),
            # 通用查询、工具和兼容领域路径仍调用原独立业务函数。
            ("query_understanding", self._query_understanding_node),
            ("context_need_planning", self._context_need_node),
            ("general_tool_routing", self._general_tool_routing_node),
            ("general_tool_call", self._general_tool_call_node),
            ("verify_tool_result", self._verify_tool_result_node),
            ("route_domain_workflow", self._route_domain_workflow_node),
            ("retrieve_sales_intelligence", self._retrieve_sales_node),
            ("build_context", self._build_context_node),
            # 固定生成尾链保留初次审查、一次有界优化及优化后三次复核。
            ("knowledge_fusion", self._knowledge_fusion_node),
            ("compress_context", self._compress_context_node),
            ("prompt_assembly", self._prompt_assembly_node),
            ("model_routing", self._model_routing_node),
            ("generate_response", self._generate_response_node),
            ("grounding_verification_initial", self._grounding_node),
            ("compliance_review_initial", self._compliance_node),
            ("output_pii_scan_initial", self._output_pii_node),
            ("evaluate_response_quality", self._evaluate_response_node),
            ("regenerate_response_if_needed", self._regenerate_response_node),
            ("output_pii_scan_final", self._output_pii_node),
            ("grounding_verification_final", self._grounding_node),
            ("compliance_review_final", self._compliance_node),
            ("response_packaging", self._response_packaging_node),
            ("update_short_term_memory", self._short_term_memory_node),
            ("long_term_memory_candidate", self._long_term_memory_node),
            ("trace_finalize", self._trace_finalize_node),
        )
        # 逐项显式注册命名节点，规格表只消除机械重复，不改变 Graph 拓扑。
        for node_name, node_handler in node_specs:
            # 每个名称与原业务步骤或分支位置一一对应，便于 Trace 和拓扑测试精确定位。
            builder.add_node(node_name, node_handler)

        # finish 代表输入阻断或同步安全降级，保持原实现不调用 trace_finalize 的提前返回语义。
        builder.add_conditional_edges(
            "input_guardrail",
            self._route_after_input_guardrail,
            {"continue": "restore_memory", "finish": END},
        )
        # 意图澄清分支仍会写短期记忆；这是它与后续三个澄清分支的原有差异。
        builder.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {"clarify": "clarify_after_intent", "continue": "semantic_risk_classification"},
        )
        # Registry 查询只决定下一节点，不写 State、不调用数据库或外部接口。
        builder.add_conditional_edges(
            "semantic_risk_classification",
            self._route_after_semantic_risk,
            {
                "domain_agent": "invoke_domain_agent",
                "domain_unavailable": "domain_agent_unavailable",
                "general": "query_understanding",
            },
        )
        # 原 if/elif 优先级被显式编码为四个有限路由值。
        builder.add_conditional_edges(
            "context_need_planning",
            self._route_after_context_need,
            {
                "clarify": "clarify_after_context",
                "tool": "general_tool_routing",
                "domain": "route_domain_workflow",
                "direct": "knowledge_fusion",
            },
        )
        # 工具规划前后分别保留原澄清短路，缺参时绝不执行工具。
        builder.add_conditional_edges(
            "general_tool_routing",
            self._route_after_tool_stage,
            {"clarify": "clarify_after_tool_routing", "continue": "general_tool_call"},
        )
        # 工具执行后的兼容澄清判断沿用同一纯路由，但 continue 映射到结果校验。
        builder.add_conditional_edges(
            "general_tool_call",
            self._route_after_tool_stage,
            {"clarify": "clarify_after_tool_call", "continue": "verify_tool_result"},
        )

        # 普通边规格逐项对应改造前赋值调用顺序，包含所有分支收尾和 END 终止路径。
        fixed_edges = (
            # 公共入口与输入通过后的固定顺序。
            (START, "initialize_context"),
            ("initialize_context", "input_guardrail"),
            ("restore_memory", "normalize_messages"),
            ("normalize_messages", "classify_intent"),
            # 意图澄清保留原额外短期记忆写入。
            ("clarify_after_intent", "package_after_intent_clarify"),
            ("package_after_intent_clarify", "memory_after_intent_clarify"),
            ("memory_after_intent_clarify", "finalize_after_intent_clarify"),
            ("finalize_after_intent_clarify", END),
            # 专业 Agent 完成后结束；禁用占位能力仍封装、写 Session 再结束。
            ("invoke_domain_agent", END),
            ("domain_agent_unavailable", "package_domain_unavailable"),
            ("package_domain_unavailable", "memory_domain_unavailable"),
            ("memory_domain_unavailable", "finalize_domain_unavailable"),
            ("finalize_domain_unavailable", END),
            # 通用查询规划及 planner 澄清收尾。
            ("query_understanding", "context_need_planning"),
            ("clarify_after_context", "package_after_context_clarify"),
            ("package_after_context_clarify", "finalize_after_context_clarify"),
            ("finalize_after_context_clarify", END),
            # 工具规划和执行后的两条澄清收尾都不写短期记忆。
            ("clarify_after_tool_routing", "package_after_tool_routing_clarify"),
            ("package_after_tool_routing_clarify", "finalize_after_tool_routing_clarify"),
            ("finalize_after_tool_routing_clarify", END),
            ("clarify_after_tool_call", "package_after_tool_call_clarify"),
            ("package_after_tool_call_clarify", "finalize_after_tool_call_clarify"),
            ("finalize_after_tool_call_clarify", END),
            # 工具失败仍由校验节点写 RECOVERY，之后与兼容领域路径汇入统一尾链。
            ("verify_tool_result", "knowledge_fusion"),
            ("route_domain_workflow", "retrieve_sales_intelligence"),
            ("retrieve_sales_intelligence", "build_context"),
            ("build_context", "knowledge_fusion"),
            # 生成、初审、优化、复审、记忆和正常终止顺序保持不变。
            ("knowledge_fusion", "compress_context"),
            ("compress_context", "prompt_assembly"),
            ("prompt_assembly", "model_routing"),
            ("model_routing", "generate_response"),
            ("generate_response", "grounding_verification_initial"),
            ("grounding_verification_initial", "compliance_review_initial"),
            ("compliance_review_initial", "output_pii_scan_initial"),
            ("output_pii_scan_initial", "evaluate_response_quality"),
            ("evaluate_response_quality", "regenerate_response_if_needed"),
            ("regenerate_response_if_needed", "output_pii_scan_final"),
            ("output_pii_scan_final", "grounding_verification_final"),
            ("grounding_verification_final", "compliance_review_final"),
            ("compliance_review_final", "response_packaging"),
            ("response_packaging", "update_short_term_memory"),
            ("update_short_term_memory", "long_term_memory_candidate"),
            ("long_term_memory_candidate", "trace_finalize"),
            ("trace_finalize", END),
        )
        # 逐项注册普通边；规格表中的顺序仅供审阅，执行顺序由边关系本身决定。
        for source_node, target_node in fixed_edges:
            # 每条边都只表达固定控制流，不调用业务函数或修改共享 State。
            builder.add_edge(source_node, target_node)

        # 当前同步客户链路原本没有 checkpoint/store/interrupt；显式不配置以保持无断点恢复语义。
        return builder.compile()

    def invoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        """用编译后的 Graph 同步执行一次请求，并继续返回原 ``AgentState`` 对象。"""

        # 兼容原调用方直接传字典；Pydantic 继续负责字段、默认值和可选语义校验。
        if isinstance(state, dict):
            # 将字典恢复为原 Pydantic 状态，拒绝无约束通用 dict 在业务节点间传播。
            state = AgentState(**state)
        # 只给 LangGraph 设置框架步数上限；未传 thread_id/checkpointer，不改变会话恢复或重试语义。
        result = self.compiled_graph.invoke(
            {"agent_state": state},
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        # Graph API 的输出是 TypedDict 映射；内部对象仍是带 Trace Sink 的原 AgentState 实例。
        graph_result = cast(AgentGraphState, result)
        # 返回原外部调用方预期的 AgentState，而不是暴露内部 Graph TypedDict 信封。
        return graph_result["agent_state"]

    def _run_insurance_conversation(self, state: AgentState) -> AgentState:
        """保留旧私有入口，并委托已注册保险 Agent 的 Graph API 实现。"""

        # 旧测试、脚本或外部代码可能直接调用该私有方法，因此继续保留相同签名。
        advisor_agent = self.domain_agent_registry.get("advisor_coach_agent")
        # 自定义 Registry 漏配现有保险 Agent 时继续使用原明确失败语义。
        if advisor_agent is None:
            # 不允许兼容入口静默绕过保险合规链路，因此抛出原 RuntimeError。
            raise RuntimeError("DomainAgentRegistry 未注册 advisor_coach_agent")
        # 同步委托领域 Agent 的编译图，返回结构仍是原 AgentState。
        return advisor_agent.invoke(state)


def build_agent_graph(
    memory_manager: MemoryBackend | None = None,
    business_store: BusinessMemoryStore | None = None,
    intent_router: IntentRouter | None = None,
    kyc_extractor: InsuranceKycExtractor | None = None,
    insurance_knowledge_provider: InsuranceKnowledgeProvider | None = None,
    insurance_news_enabled: bool | None = None,
    domain_agent_registry: DomainAgentRegistry | None = None,
    proposal_agent: ProposalAgentPort | None = None,
) -> AgentGraph:
    """构建并编译总控 Graph；函数名和参数顺序保持原调用方兼容。"""

    # 构造阶段只注入依赖、注册节点和 compile，不执行模型、工具、数据库或第三方请求。
    return AgentGraph(
        memory_manager,
        business_store,
        intent_router,
        kyc_extractor,
        insurance_knowledge_provider,
        insurance_news_enabled,
        domain_agent_registry,
        proposal_agent,
    )
