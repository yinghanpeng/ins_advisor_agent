"""Workflow engine facade."""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

import os
from typing import Any

from agent_core.config.runtime import RuntimeSettings, load_runtime_settings
from agent_core.graph.builder import build_agent_graph
from agent_core.graph.state import AgentNode, AgentState
from agent_core.intents.router import IntentRouter, build_intent_router
from agent_core.memory.business_store import BusinessMemoryStore, InMemoryBusinessMemoryStore
from agent_core.memory.manager import MemoryBackend, MemoryManager
from agent_core.memory.recall import ProductionMemoryRetriever
from agent_core.models.client import OpenAICompatibleChatClient
from agent_core.models.client import bind_model_trace_sink
from agent_core.observability.langsmith_client import LangSmithAdapter
from agent_core.observability.logger import StructuredLogger, configure_logging
from agent_core.registry.builtins import bootstrap_repository_artifacts
from agent_core.registry.models import ArtifactKind, RiskLevel
from agent_core.registry.prompts import PromptRenderer
from agent_core.registry.runtime import ArtifactResolver, RuntimeRequest
from agent_core.registry.service import ArtifactRegistry
from agent_core.skills.insurance_advisor.kyc import InsuranceKycExtractor
from agent_core.skills.insurance_advisor.knowledge import (
    InsuranceKnowledgeProvider,
    LocalInsuranceKnowledgeProvider,
)
from agent_core.workflow.contracts import AgentRunExecutionContext, AgentRunRequest, AgentRunResponse


# TRACE_LOG_SAFE_FIELDS 只允许控制面与计数字段进入实时日志，避免记录客户原文、KYC 值或模型正文。
TRACE_LOG_SAFE_FIELDS = frozenset(
    {
        "ts",
        "trace_id",
        "session_id",
        "workflow_name",
        "domain_skill",
        "node_name",
        "from_state",
        "to_state",
        "reason",
        "intent",
        "route",
        "risk_level",
        "decision_action",
        "status",
        "action",
        "tool_name",
        "attempt",
        "confidence",
        "count",
        "error_count",
        "final_state",
        "response_ready",
        "fallback",
        "fields",
        "keys",
        "step_index",
    }
)

# AGENT_STEP_LABELS 把内部状态码翻译成终端可直接阅读的中文流程名称，覆盖当前全部显式节点。
AGENT_STEP_LABELS: dict[str, str] = {
    AgentNode.IDLE.value: "等待请求",
    AgentNode.GREETING.value: "问候处理",
    AgentNode.INIT_CONTEXT.value: "初始化",
    AgentNode.INPUT_GUARDRAIL.value: "输入安全拦截",
    AgentNode.RESTORE_MEMORY.value: "恢复记忆",
    AgentNode.LOAD_BUSINESS_MEMORY.value: "加载业务记忆",
    AgentNode.NORMALIZE_MESSAGES.value: "消息标准化",
    AgentNode.CLASSIFY_INTENT.value: "意图识别",
    AgentNode.SEMANTIC_RISK_CLASSIFICATION.value: "语义风险分类",
    AgentNode.QUERY_UNDERSTANDING.value: "Query Understanding",
    AgentNode.CONTEXT_NEED_PLANNING.value: "Context Need Planning",
    AgentNode.ROUTE_CAPABILITY.value: "执行路由",
    AgentNode.GENERAL_TOOL_ROUTING.value: "通用工具路由",
    AgentNode.AGENTIC_TOOL_LOOP.value: "Agentic 工具循环",
    AgentNode.GENERAL_TOOL_CALL.value: "通用工具执行",
    AgentNode.VERIFY_TOOL_RESULT.value: "工具结果校验",
    AgentNode.GENERATE_CLARIFICATION_RESPONSE.value: "生成澄清问题",
    AgentNode.GENERAL_RESPONSE_GENERATION.value: "通用回答生成",
    AgentNode.DOMAIN_WORKFLOW_ROUTING.value: "领域工作流路由",
    AgentNode.SALES_INTELLIGENCE_ROUTING.value: "销售智能路由",
    AgentNode.SALES_CORPUS_INGESTION.value: "销售语料导入",
    AgentNode.SALES_INSIGHT_EXTRACTION.value: "销售洞察抽取",
    AgentNode.SALES_INSIGHT_RETRIEVAL.value: "销售洞察检索",
    AgentNode.COLLECT_REQUIREMENTS.value: "需求采集",
    AgentNode.UPDATE_STATE.value: "状态更新",
    AgentNode.EXTRACT_INSURANCE_KYC.value: "保险 KYC 抽取",
    AgentNode.ANALYZE_KYC_AND_ROUTE.value: "KYC 分析与路由",
    AgentNode.MEMORY_WRITE_PROPOSAL.value: "记忆写入提案",
    AgentNode.VALIDATE_MEMORY_WRITE.value: "记忆写入校验",
    AgentNode.PERSIST_MEMORY_SNAPSHOT.value: "业务记忆持久化",
    AgentNode.BUILD_COMPACT_CONTEXT.value: "构建业务紧凑上下文",
    AgentNode.STATUS_ROUTER.value: "KYC 状态路由",
    AgentNode.GENERATE_KYC_QUESTIONS.value: "生成 KYC 补问",
    AgentNode.RETRIEVE_DIALOGUE_PATTERNS.value: "检索沟通模式",
    AgentNode.RETRIEVE_INSURANCE_KNOWLEDGE.value: "检索保险双知识库",
    AgentNode.RETRIEVE_EXTERNAL_CONTEXT_IF_NEEDED.value: "按需检索外部上下文",
    AgentNode.GENERATE_STRATEGY.value: "生成保险沟通策略",
    AgentNode.POST_RESPONSE_LOGGER.value: "记录生成结果",
    AgentNode.PLAN_TASK.value: "任务规划",
    AgentNode.RETRIEVE_CONTEXT.value: "上下文检索",
    AgentNode.BUILD_CONTEXT.value: "上下文组装",
    AgentNode.KNOWLEDGE_FUSION.value: "知识融合",
    AgentNode.CONTEXT_COMPRESSION.value: "上下文压缩",
    AgentNode.PROMPT_ASSEMBLY.value: "Prompt 组装",
    AgentNode.MODEL_ROUTING.value: "模型路由",
    AgentNode.GENERATE_RESPONSE.value: "模型生成",
    AgentNode.GROUNDING_VERIFICATION.value: "Grounding 校验",
    AgentNode.COMPLIANCE_REVIEW.value: "合规检查",
    AgentNode.OUTPUT_PII_SCAN.value: "输出 PII 扫描",
    AgentNode.EVALUATE_RESPONSE_QUALITY.value: "回答质量评估",
    AgentNode.REGENERATE_RESPONSE.value: "回答重生成",
    AgentNode.RESPONSE_PACKAGING.value: "响应封装",
    AgentNode.SHORT_TERM_MEMORY_UPDATE.value: "写入短期记忆",
    AgentNode.LONG_TERM_MEMORY_CANDIDATE.value: "长期记忆候选判断",
    AgentNode.TRACE_FINALIZE.value: "Trace 收尾",
    AgentNode.RECOVERY.value: "异常恢复",
    AgentNode.FINAL.value: "完成",
    AgentNode.ERROR.value: "错误终止",
}


class WorkflowEngine:
    """统一执行 Agent workflow，是 main、API、Dify 调用的共同入口。"""

    def __init__(
        self,
        log: StructuredLogger | None = None,
        langsmith: LangSmithAdapter | None = None,
        memory_manager: MemoryBackend | None = None,
        business_store: BusinessMemoryStore | None = None,
        intent_router: IntentRouter | None = None,
        kyc_extractor: InsuranceKycExtractor | None = None,
        insurance_knowledge_provider: InsuranceKnowledgeProvider | None = None,
        insurance_news_enabled: bool | None = None,
        memory_retriever: ProductionMemoryRetriever | None = None,
        memory_decision_client: OpenAICompatibleChatClient | None = None,
        artifact_registry: ArtifactRegistry | None = None,
        artifact_resolver: ArtifactResolver | None = None,
        settings: RuntimeSettings | None = None,
    ) -> None:
        """初始化日志、LangSmith adapter 和本地兼容 graph。"""
        # 确保 CLI 与 Uvicorn 都启用本地 INFO 结构化日志；LangSmith 不能替代这条基础排障链路。
        configure_logging(os.getenv("LOG_LEVEL", "INFO"))
        # 直接构造 Engine 的 CLI/SDK 也读取 CONFIG_DIR，保证新闻开关和模型配置不是生产 Runtime 专属。
        self.settings = settings or load_runtime_settings(os.getenv("CONFIG_DIR", "configs"))
        # 创建结构化日志器；如果调用方没有传入，就使用本地 stdout JSON logger。
        self.log = log or StructuredLogger()
        # 从环境变量初始化 LangSmith；没有配置时 adapter 会自动降级，不影响本地运行。
        self.langsmith = langsmith or LangSmithAdapter.from_env(self.log)
        # 创建共享 MemoryManager；同一个 WorkflowEngine 实例内的多轮对话会复用这份内存存储。
        self.memory_manager = memory_manager or MemoryManager()
        # 业务记忆 store 独立于 session/task/preference 记忆；默认内存实现保证本地 demo 和测试无需数据库。
        self.business_store = business_store or InMemoryBusinessMemoryStore()
        # 双层意图 Router 在 Engine 生命周期内复用，避免每个请求重复读取意图目录和创建模型客户端。
        self.intent_router = intent_router or build_intent_router(settings=self.settings)
        # 保险 KYC Extractor 只抽事实；代码负责合并、完整度、轮次和路由。
        self.kyc_extractor = kyc_extractor or InsuranceKycExtractor(self.settings)
        # 本地不伪造知识内容；生产通过 pgvector Provider 注入两个真实知识库。
        self.insurance_knowledge_provider = (
            insurance_knowledge_provider or LocalInsuranceKnowledgeProvider()
        )
        # 是否允许保险代码路径调用只读新闻工具由运行时配置注入。
        self.insurance_news_enabled = (
            self.settings.insurance_knowledge.news_enabled
            if insurance_news_enabled is None
            else insurance_news_enabled
        )
        # Tool、Skill 和 Prompt 在一个 Engine 内共享 Registry，Artifact 身份不会被拆成无类型配置。
        self.artifact_registry = artifact_registry or ArtifactRegistry()
        # Registry 缓存参数来自强类型运行时配置，不接受单次请求动态覆盖。
        registry_settings = self.settings.artifact_registry
        # Resolver 与 Registry 同生命周期；具体 Alias 仍在每次 Run 开始时解析并固定快照。
        self.artifact_resolver = artifact_resolver or ArtifactResolver(
            self.artifact_registry,
            cache_ttl_seconds=registry_settings.cache_ttl_seconds,
            negative_ttl_seconds=registry_settings.negative_cache_ttl_seconds,
            maximum_cache_entries=registry_settings.maximum_cache_entries,
        )
        # 构建统一代码执行器；保险请求由意图自动进入 Handler，不再按 workflow_name 分叉。
        self.graph = build_agent_graph(
            self.memory_manager,
            self.business_store,
            self.intent_router,
            self.kyc_extractor,
            self.insurance_knowledge_provider,
            self.insurance_news_enabled,
            memory_retriever,
            memory_decision_client,
            self.settings.memory,
            self.artifact_resolver,
        )

    def _langsmith_state_snapshot(self, state: AgentState) -> dict[str, Any]:
        """构造完整 AgentState 快照，排除会递归复制全部历史事件的字段。"""

        # trace_events 已作为独立 Run Event 上传；嵌入每个状态快照会造成指数级重复和请求膨胀。
        return state.model_dump(mode="json", exclude={"trace_events"})

    def _log_trace_event(self, event: dict[str, Any], state: AgentState | None = None) -> None:
        """本地记录安全摘要，并按 LangSmith 数据策略上传远程事件。"""

        # 只复制控制面字段，完整 memory_context、工具输出、Prompt 和回答不会进入本地日志。
        safe_payload = {
            key: value
            for key, value in event.items()
            if key in TRACE_LOG_SAFE_FIELDS
        }
        # 原始 event 字段改名后写入，避免与 StructuredLogger.event 的位置参数冲突。
        safe_payload["trace_event_name"] = str(event.get("event") or "unknown")
        # 每次 add_trace_event 都立即输出一条 JSON 日志，便于按 trace_id 实时观察节点进度。
        self.log.event("trace_event", **safe_payload)
        # 远程事件从原始结构复制；Adapter 再按 control/full 策略投影并强制清除认证凭据。
        remote_payload = dict(event)
        # 完整模式在每次事件旁保存最新状态，Adapter 会把它放进节点 state_before/state_after。
        if self.langsmith.captures_full_content and state is not None:
            # model_dump 保留 KYC、Prompt、工具、知识、模型结果和最终回答等全部业务字段。
            remote_payload["state_snapshot"] = self._langsmith_state_snapshot(state)
        # 状态迁移代表真实执行了一步；额外输出中文步骤日志，终端无需翻译内部枚举。
        if event.get("event") == "state_transition":
            # to_state 是本步实际进入的状态码；未知扩展节点回退显示原状态码，避免日志丢失。
            step_code = str(event.get("to_state") or "UNKNOWN")
            # 中文步骤名优先来自完整映射，未来插件节点未登记时仍显示其原始状态码。
            step_name = AGENT_STEP_LABELS.get(step_code, step_code)
            # 远程 Run Tree 使用同一中文名和内部码，保证本地与 LangSmith 节点一一对应。
            remote_payload.update({"step_name": step_name, "step_code": step_code})
            # agent_flow_step 是面向人工排障的主流程日志，不包含用户输入、模型正文或业务事实。
            self.log.event(
                "agent_flow_step",
                trace_id=event.get("trace_id"),
                session_id=event.get("session_id"),
                step_index=event.get("step_index"),
                step_name=step_name,
                step_code=step_code,
                status="entered",
                reason=event.get("reason"),
            )
        # LangSmith Adapter 按配置选择控制面或完整正文，并始终执行不可关闭的凭据清理。
        self.langsmith.trace_event(
            str(event.get("event") or "unknown"),
            remote_payload,
        )

    def _log_flow_summary(self, state: AgentState, *, status: str) -> None:
        """输出本次请求真实经过的中文箭头流程摘要。"""

        # 只读取已审计的状态迁移目标；未执行的条件分支不会出现在摘要中。
        step_codes = [str(item.get("to_state") or "UNKNOWN") for item in state.state_transitions]
        # 将状态码翻译成中文名称，未知扩展节点保留原码以便排障而不是静默丢弃。
        step_names = [AGENT_STEP_LABELS.get(code, code) for code in step_codes]
        # 单条 agent_flow_summary 便于直接看到“初始化 → 风控 → 路由 → 生成 → 收尾”的完整路径。
        self.log.event(
            "agent_flow_summary",
            trace_id=state.trace_id,
            session_id=state.session_id,
            status=status,
            step_count=len(step_names),
            flow=" → ".join(step_names),
        )

    def run(
        self,
        request: AgentRunRequest,
        *,
        execution_context: AgentRunExecutionContext | None = None,
    ) -> AgentRunResponse:
        """执行一次 Agent 请求，并返回包含状态链路和 trace 的结构化响应。"""
        # execution_context 只来自代码侧受信调用者；公开 API 仍只能构造受 default-deny 保护的请求 DTO。
        trusted_context = execution_context or AgentRunExecutionContext()
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
            # workflow_name 仅保留 API 向后兼容和日志标签；保险路由不再读取该字段。
            workflow_name=request.workflow_name,
            # domain_skill 可由调用方指定；为空时由 classify_intent / route_domain_workflow 自动判断。
            domain_skill=request.domain_skill,
            # metadata 只保存契约允许的调用端、渠道和实验观测标签，不参与知识注入或业务选行。
            metadata=request.metadata,
            # cost 只接受建模后的请求预算；其它内部状态不能通过 execution_context 任意覆盖。
            cost=(
                {"request_token_budget": trusted_context.request_token_budget}
                if trusted_context.request_token_budget is not None
                else {}
            ),
        )
        # 仓库内置 Artifact 已经过代码评审，可按租户幂等初始化，不接受请求 Metadata 动态创建实现。
        if (
            self.settings.artifact_registry.enabled
            and self.settings.artifact_registry.bootstrap_repository_artifacts
        ):
            # 初始化只写入仓库签名/允许的内置定义，并由 Registry 服务保证租户内幂等。
            bootstrap_repository_artifacts(self.artifact_registry, tenant_id=state.tenant_id)
        # 记录请求开始事件；trace_id 从这里开始贯穿状态迁移、工具调用、检索和最终响应。
        self.log.event(
            "agent_run_started",
            trace_id=state.trace_id,
            session_id=state.session_id,
            workflow_name=state.workflow_name,
        )
        # 创建 LangSmith 根 Run；完整模式会附请求正文，控制面模式只保留流程标签与主体引用。
        self.langsmith.start_run(
            trace_id=state.trace_id,
            tenant_id=state.tenant_id,
            session_id=state.session_id,
            workflow_name=state.workflow_name,
            app_env=self.settings.app_env,
            request_payload=request.model_dump(mode="json"),
        )
        # 请求级事件回调捕获当前 AgentState，使远程节点可以记录完整前后状态。
        def trace_event_sink(event: dict[str, Any]) -> None:
            """把当前事件和最新状态交给统一的本地/远程日志边界。"""

            # state 是当前请求独占对象，闭包不会跨请求共享或串联客户信息。
            self._log_trace_event(event, state)

        # 绑定事件 Sink 后，每个状态迁移、模型、工具和检索事件都会立即进入观测链路。
        state.bind_trace_event_sink(trace_event_sink)

        # 模型客户端通过 ContextVar 上报实际 messages、供应商响应和 token/latency，不携带鉴权 Header。
        def model_trace_sink(event: str, payload: dict[str, Any]) -> None:
            """把真实模型请求与响应转换成当前 AgentState 的结构化事件。"""

            # add_trace_event 会自动补 trace/session/workflow，并触发上面绑定的实时 Sink。
            state.add_trace_event(event, **payload)
        # 统一代码入口先执行安全、记忆和双层意图识别，再自动选择通用或保险领域处理器。
        try:
            # 模型 Trace 上下文只覆盖本次图执行，退出后自动恢复，保证并发请求隔离。
            with bind_model_trace_sink(model_trace_sink):
                # 图执行前只解析一次 Alias/Binding；序列化快照随后随子任务传递，运行中不会重新漂移版本。
                if self.settings.artifact_registry.enabled:
                    # RuntimeRequest 固定本次运行的租户、环境、风险和权限解析边界。
                    runtime_request = RuntimeRequest(
                        run_id=state.trace_id,
                        tenant_id=state.tenant_id,
                        agent_id=state.domain_skill or state.workflow_name,
                        environment=self.settings.app_env,
                        scenario=state.workflow_name,
                        # 网关认证租户级服务 Agent；终端用户的资源权限仍由 ToolGuardrail 做确定性校验。
                        roles=["operator"],
                        scopes=["*"],
                        maximum_risk=RiskLevel.MEDIUM,
                    )
                    # Snapshot 在图执行前一次性解析，后续节点只消费精确版本与 Hash。
                    snapshot = self.artifact_resolver.create_snapshot(runtime_request)
                    # 保存受信解析请求供子任务复用，不允许子任务重新拼装更高权限请求。
                    state.metadata["_artifact_runtime_request"] = runtime_request.model_dump(mode="json")
                    # 保存不可变运行快照，确保同一 Agent Run 内 Alias 变更不会影响已经开始的任务。
                    state.metadata["_artifact_runtime_snapshot"] = snapshot.model_dump(mode="json")
                    # 系统 Prompt 只从当前快照按 Artifact 身份查找，不执行第二次 Registry 解析。
                    system_ref = snapshot.by_name(
                        ArtifactKind.PROMPT,
                        "agent_core",
                        "agent_system_prompt",
                    )
                    # 只有快照明确包含系统 Prompt 时才渲染；请求不能通过名称临时解析另一个版本。
                    if system_ref:
                        # Renderer 使用同一 Request/Snapshot 渲染，保持依赖版本和变量边界可审计。
                        state.metadata["_resolved_system_prompt"] = PromptRenderer(
                            self.artifact_resolver
                        ).render(
                            runtime_request,
                            snapshot,
                            system_ref.artifact_id,
                            {},
                        )
                    # Trace 只记录 Artifact ID、版本和 Hash，不写入完整 Prompt 或 Skill 正文。
                    state.add_trace_event(
                        "artifact_runtime_snapshot_created",
                        artifact_count=len(snapshot.artifacts),
                        registry_generation=snapshot.registry_generation,
                        artifact_versions=[
                            {
                                "artifact_id": item.artifact_id,
                                "version_id": item.version_id,
                                "version": item.version,
                                "content_hash": item.content_hash,
                            }
                            for item in snapshot.artifacts.values()
                        ],
                    )
                # 执行完整 Agent 图；节点和模型客户端通过 add_trace_event 实时报告完整进度。
                result = self.graph.invoke(state)
        # 任一未处理异常都必须留下失败节点和异常类型，然后继续抛给 API 统一处理。
        except Exception as exc:
            # 异常时先输出已经完成的流程摘要，最后一个步骤就是最接近故障的位置。
            self._log_flow_summary(state, status="failed")
            # 远程根 Run 与当前节点以失败状态收尾；完整模式额外上传脱敏异常正文与最终状态。
            self.langsmith.finish_run(
                trace_id=state.trace_id,
                status="failed",
                final_state=state.current_state.value,
                intent=state.intent,
                domain_skill=state.domain_skill,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                details={"state": self._langsmith_state_snapshot(state)},
            )
            # 失败日志不包含异常文本，避免数据库参数、用户原文或供应商响应意外进入日志。
            self.log.warning(
                "agent_run_failed",
                trace_id=state.trace_id,
                session_id=state.session_id,
                current_state=state.current_state.value,
                exception_type=type(exc).__name__,
            )
            # 保留原始异常栈供 Uvicorn/错误监控记录，HTTP 层仍只返回安全错误响应。
            raise
        # invoke 返回 AgentState；这里做一次防御性兼容，万一传入 dict 也能恢复成 AgentState。
        if isinstance(result, dict):
            # 使用 AgentState 契约重新验证字典字段，避免后续属性访问依赖未校验 Mapping。
            result = AgentState(**result)
        # 正常完成后输出一次中文箭头流程摘要，直观看到本轮实际走了哪些条件分支。
        self._log_flow_summary(result, status="completed")
        # 正常完成 LangSmith 根 Run；完整模式把最终 AgentState 作为可回放详情写入根输出。
        self.langsmith.finish_run(
            trace_id=result.trace_id,
            status="completed",
            final_state=result.final_state.value if result.final_state else result.current_state.value,
            intent=result.intent,
            domain_skill=result.domain_skill,
            details={"state": self._langsmith_state_snapshot(result)},
        )
        # 记录请求结束事件，给日志平台一个快速统计 final_state / intent 的入口。
        self.log.event(
            "agent_run_finished",
            trace_id=result.trace_id,
            final_state=result.final_state.value if result.final_state else result.current_state.value,
            intent=result.intent,
        )
        # 将内部可变状态投影成稳定的 Pydantic 响应契约后返回调用方。
        return self._response_from_state(result)

    def _response_from_state(self, state: AgentState) -> AgentRunResponse:
        """把内部 AgentState 封装成外部响应契约。"""
        # 多意图父状态的顶层 domain_skill 镜像最高优先级步骤，可能不是保险；
        # 因此还要检查完整执行计划，确保实际执行过保险步骤时返回 KYC 控制摘要。
        has_insurance_step = state.domain_skill == "insurance_advisor" or any(
            step.get("domain_skill") == "insurance_advisor"
            for step in state.intent_execution_plan
            if isinstance(step, dict)
        )
        # 显式逐字段映射，避免 AgentState 后续新增内部字段时被响应自动暴露。
        return AgentRunResponse(
            # 回传贯穿整次运行的追踪 ID，供日志和评估关联。
            trace_id=state.trace_id,
            # 回传会话 ID，调用方下一轮可继续相同短期记忆。
            session_id=state.session_id,
            # 优先返回显式终态；尚无终态时使用当前节点作为防御性兼容。
            final_state=state.final_state.value if state.final_state else state.current_state.value,
            # answer 为空时规范化为空字符串，满足响应契约的非空类型要求。
            answer=state.answer or "",
            # 返回本轮最终意图标签。
            intent=state.intent,
            # 返回实际执行的领域 Skill 标签。
            domain_skill=state.domain_skill,
            # 保留完整意图路由诊断信息给内部响应消费者。
            intent_routing_result=state.intent_routing_result,
            # 返回 Redis 活跃意图控制信封，不在此处读取 KYC 事实值。
            active_intent=state.active_intent_state,
            # 只有保险领域响应才组装 KYC 控制摘要，通用请求固定返回空字典。
            insurance_kyc_status=(
                {
                    # information_status 表示信息是否足够继续策略生成。
                    "information_status": state.information_status,
                    # missing_fields 仅列出仍需补充的字段名称。
                    "missing_fields": state.missing_fields,
                    # asked_focuses 记录当前任务已经向客户展示过的补问焦点。
                    "asked_focuses": state.asked_focuses,
                    # kyc_question_round_count 控制最多连续追问轮数。
                    "kyc_question_round_count": state.kyc_question_round_count,
                    # kyc_completeness_score 供内部评估信息完整度。
                    "kyc_completeness_score": state.kyc_completeness_score,
                    # opportunity_score 供内部评估沟通机会，不进入客户安全 DTO。
                    "opportunity_score": state.opportunity_score,
                }
                if has_insurance_step
                else {}
            ),
            # 汇总输入、工具和输出侧风控结果。
            guardrails=state.guardrail_results,
            # 返回本轮实际使用的检索证据供内部诊断。
            retrieved_context=state.retrieved_context,
            # 返回完整结构化 Trace 事件。
            trace_events=state.trace_events,
            # 返回可被 SSE Adapter 消费的进度事件。
            stream_events=state.stream_events,
            # 返回纯状态节点迁移审计链。
            state_transitions=state.state_transitions,
            # 返回工具调用参数、状态、耗时和错误审计。
            tool_calls=state.tool_calls,
            # 返回结构化工具执行结果。
            tool_results=state.tool_results,
            # 返回实体、时间、改写 Query 等理解结果。
            query_understanding=state.query_understanding,
            # 返回 Memory、RAG、Tool 等上下文需求规划标记。
            context_needs=state.context_needs,
            # 返回已经过输出检查的前端响应包。
            response_package=state.response_package,
            # 返回 Grounding 事实校验结果。
            grounding_result=state.grounding_result,
            # 返回质量评估和重生成信息。
            evaluation_result=state.evaluation_result,
            # 返回输出侧 PII 扫描摘要。
            output_pii_scan_result=state.output_pii_scan_result,
            # 返回本轮预算和资源消耗汇总。
            cost=state.cost,
        )
