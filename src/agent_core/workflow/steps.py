"""Workflow step contracts for Agent Core and Dify mapping."""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

from agent_core.graph.state import AgentNode
from agent_core.workflow.contracts import WorkflowContract, WorkflowStepContract


# 保险破冰助手的逻辑步骤清单，用于文档、Dify 映射和面试讲解。
BREAK_ICE_ASSISTANT_STEPS = [
    # 识别用户是不是保险沟通/破冰/异议处理需求。
    "classify_intent",
    # 抽取客户 KYC，例如企业主、两个孩子、资产偏好。
    "extract_customer_kyc",
    # 抽取销售当前卡点，例如不会破冰、客户不信任、客户只看银行理财。
    "extract_sales_pain",
    # 判断具体沟通场景，例如饭局破冰、老客维护、计划书讲解。
    "classify_scene",
    # 检索已审核的销售实战洞察卡片。
    "retrieve_sales_intelligence",
    # 必要时检索外部上下文，例如热点新闻或宏观背景。
    "retrieve_external_context_if_needed",
    # 构建包含证据边界的上下文。
    "build_context",
    # 生成合规、低压、可执行的回答。
    "generate_response",
    # 输出前做保险合规审查。
    "compliance_review",
    # 返回最终响应。
    "final_response",
]


# BREAK_ICE_ASSISTANT_CONTRACT 是“破冰助手 workflow”的显式 step contract。
# 它不执行代码，而是声明每个 step 的输入、输出、允许下一状态、风控和 trace 字段。
BREAK_ICE_ASSISTANT_CONTRACT = WorkflowContract(
    # Dify、文档和测试都会引用这个稳定工作流名。
    name="break_ice_assistant_workflow",
    # 该工作流从意图识别开始，因为用户请求进入时还不知道是否命中保险顾问。
    entry_state=AgentNode.CLASSIFY_INTENT,
    # 允许正常结束、人工审批停住或错误终止。
    final_states=[AgentNode.FINAL, AgentNode.HUMAN_APPROVAL, AgentNode.ERROR],
    # steps 明确每个节点的契约，避免流程只藏在 prompt 或代码 if/else 中。
    steps=[
        # classify_intent step：识别意图并产出路由结果。
        WorkflowStepContract(
            name="classify_intent",
            state=AgentNode.CLASSIFY_INTENT,
            description="Classify user intent and choose general/domain capability route.",
            required_inputs=["input_text"],
            produced_outputs=["intent", "capability_route", "domain_skill"],
            allowed_next_states=[AgentNode.ROUTE_CAPABILITY],
            guardrails=["input_prompt_injection"],
            trace_fields=["trace_id", "intent", "capability_route"],
        ),
        # route_domain_workflow step：把保险顾问请求转入销售智能层。
        WorkflowStepContract(
            name="route_domain_workflow",
            state=AgentNode.DOMAIN_WORKFLOW_ROUTING,
            description="Route insurance advisor requests to the proper domain workflow.",
            required_inputs=["intent", "domain_skill"],
            produced_outputs=["sales_route"],
            allowed_next_states=[AgentNode.SALES_INTELLIGENCE_ROUTING, AgentNode.BUILD_CONTEXT],
            trace_fields=["trace_id", "domain_skill", "sales_route"],
        ),
        # retrieve_sales_intelligence step：只检索已审核销售卡片，不直接使用原始访谈。
        WorkflowStepContract(
            name="retrieve_sales_intelligence",
            state=AgentNode.SALES_INSIGHT_RETRIEVAL,
            description="Retrieve approved sales insight cards instead of raw interviews.",
            required_inputs=["input_text", "sales_route"],
            produced_outputs=["rewritten_queries", "retrieved_context"],
            allowed_next_states=[AgentNode.BUILD_CONTEXT, AgentNode.RECOVERY],
            guardrails=["sales_corpus_guardrail"],
            tools_allowed=["knowledge_search"],
            trace_fields=["trace_id", "rewritten_queries", "retrieved_context"],
        ),
        # build_context step：把检索证据压缩成带来源边界的 digest。
        WorkflowStepContract(
            name="build_context",
            state=AgentNode.BUILD_CONTEXT,
            description="Build compact context with source boundaries and evidence digest.",
            required_inputs=["retrieved_context"],
            produced_outputs=["sales_insight_digest"],
            allowed_next_states=[AgentNode.GENERATE_RESPONSE],
            trace_fields=["trace_id", "sales_insight_digest"],
        ),
        # generate_response step：基于压缩上下文生成候选回答。
        WorkflowStepContract(
            name="generate_response",
            state=AgentNode.GENERATE_RESPONSE,
            description="Generate a domain answer from compact context.",
            required_inputs=["input_text", "sales_insight_digest"],
            produced_outputs=["answer"],
            allowed_next_states=[AgentNode.COMPLIANCE_REVIEW],
            trace_fields=["trace_id", "answer"],
        ),
        # compliance_review step：输出前审查，必要时进入人工审批。
        WorkflowStepContract(
            name="compliance_review",
            state=AgentNode.COMPLIANCE_REVIEW,
            description="Review output and route unsafe responses to human approval.",
            required_inputs=["answer"],
            produced_outputs=["guardrail_results"],
            allowed_next_states=[AgentNode.FINAL, AgentNode.HUMAN_APPROVAL],
            guardrails=["insurance_output_compliance"],
            trace_fields=["trace_id", "guardrail_results", "final_state"],
        ),
    ],
)


# INSURANCE_KYC_COACH_STEPS 对齐 Dify “4 轮 KYC + 热点新闻工具”的业务语义。
INSURANCE_KYC_COACH_STEPS = [
    "initialize_context",
    "input_guardrail",
    "load_business_memory",
    "analyze_kyc_and_route",
    "memory_write_proposal",
    "validate_memory_write",
    "persist_memory_snapshot",
    "build_compact_context",
    "status_router",
    "generate_kyc_questions",
    "retrieve_dialogue_patterns",
    "retrieve_external_context_if_needed",
    "generate_strategy",
    "compliance_review",
    "post_response_logger",
    "final_response",
]


INSURANCE_KYC_COACH_CONTRACT = WorkflowContract(
    name="insurance_kyc_coach_workflow",
    entry_state=AgentNode.INIT_CONTEXT,
    final_states=[AgentNode.FINAL, AgentNode.HUMAN_APPROVAL, AgentNode.ERROR],
    steps=[
        WorkflowStepContract(
            name="initialize_context",
            state=AgentNode.INIT_CONTEXT,
            description="初始化 trace、租户、会话和本轮用户输入。",
            required_inputs=["input_text", "tenant_id", "session_id"],
            produced_outputs=["trace_id", "messages", "cost"],
            allowed_next_states=[AgentNode.INPUT_GUARDRAIL],
            trace_fields=["trace_id", "session_id", "tenant_id"],
        ),
        WorkflowStepContract(
            name="input_guardrail",
            state=AgentNode.INPUT_GUARDRAIL,
            description="在读取业务记忆之前检查 prompt injection 和高风险输入。",
            required_inputs=["input_text"],
            produced_outputs=["guardrail_results", "risk_level"],
            allowed_next_states=[AgentNode.LOAD_BUSINESS_MEMORY, AgentNode.ERROR],
            guardrails=["input_prompt_injection", "unsafe_instruction"],
            trace_fields=["trace_id", "guardrail_results"],
        ),
        WorkflowStepContract(
            name="load_business_memory",
            state=AgentNode.LOAD_BUSINESS_MEMORY,
            description="读取客户事实、从业者事实、active case、已问 KYC 焦点和最近会话快照。",
            required_inputs=["tenant_id", "metadata.advisor_id", "metadata.customer_id"],
            produced_outputs=["profile_state", "practitioner_state", "asked_focuses"],
            allowed_next_states=[AgentNode.ANALYZE_KYC_AND_ROUTE],
            trace_fields=["trace_id", "opportunity_case_id", "asked_focuses"],
        ),
        WorkflowStepContract(
            name="analyze_kyc_and_route",
            state=AgentNode.ANALYZE_KYC_AND_ROUTE,
            description="产出 Dify KYC 分析节点的 18 个顶层字段，并执行 4 轮补问上限规则。",
            required_inputs=["input_text", "profile_state", "practitioner_state", "asked_focuses"],
            produced_outputs=[
                "information_status",
                "subject_type",
                "target_persona",
                "profile_state",
                "practitioner_state",
                "advisor_stage",
                "missing_fields",
                "match_evidence",
                "route_reason",
                "kyc_completeness_score",
                "opportunity_score",
                "external_grade",
                "trigger_module",
                "current_stage",
                "objective_material_need",
                "support_note",
                "kyc_question_round_count",
                "asked_focuses",
            ],
            allowed_next_states=[AgentNode.MEMORY_WRITE_PROPOSAL],
            trace_fields=["trace_id", "information_status", "missing_fields", "kyc_completeness_score"],
        ),
        WorkflowStepContract(
            name="memory_write_proposal",
            state=AgentNode.MEMORY_WRITE_PROPOSAL,
            description="只把明确事实、事件、问题、快照和分析运行整理成写入提案。",
            required_inputs=["profile_state", "practitioner_state", "match_evidence"],
            produced_outputs=["memory_write_proposal"],
            allowed_next_states=[AgentNode.VALIDATE_MEMORY_WRITE],
            trace_fields=["trace_id", "memory_write_proposal"],
        ),
        WorkflowStepContract(
            name="validate_memory_write",
            state=AgentNode.VALIDATE_MEMORY_WRITE,
            description="校验证据、PII、生成建议误写和不确定事实标记。",
            required_inputs=["memory_write_proposal"],
            produced_outputs=["memory_write_validation"],
            allowed_next_states=[AgentNode.PERSIST_MEMORY_SNAPSHOT],
            guardrails=["memory_write_policy", "pii_default_block"],
            trace_fields=["trace_id", "memory_write_validation"],
        ),
        WorkflowStepContract(
            name="persist_memory_snapshot",
            state=AgentNode.PERSIST_MEMORY_SNAPSHOT,
            description="把通过校验的业务事实、事件、问题、快照和分析运行写入业务记忆 store。",
            required_inputs=["memory_write_proposal", "memory_write_validation"],
            produced_outputs=["memory_context"],
            allowed_next_states=[AgentNode.BUILD_COMPACT_CONTEXT],
            trace_fields=["trace_id", "memory_context"],
        ),
        WorkflowStepContract(
            name="build_compact_context",
            state=AgentNode.BUILD_COMPACT_CONTEXT,
            description="构建策略生成唯一优先上下文，区分 confirmed/uncertain，过滤 PII 和原始对话全文。",
            required_inputs=["profile_state", "practitioner_state", "asked_focuses", "missing_fields"],
            produced_outputs=["compact_context"],
            allowed_next_states=[AgentNode.STATUS_ROUTER],
            trace_fields=["trace_id", "compact_context"],
        ),
        WorkflowStepContract(
            name="status_router",
            state=AgentNode.STATUS_ROUTER,
            description=(
                "按 information_status 路由：insufficient 进入补问，matched 进入模式检索和策略，"
                "unmatched 进入低压维护；第 5 轮后必须转 matched，不继续卡住。"
            ),
            required_inputs=["information_status", "kyc_question_round_count"],
            produced_outputs=["current_state"],
            allowed_next_states=[
                AgentNode.GENERATE_KYC_QUESTIONS,
                AgentNode.RETRIEVE_DIALOGUE_PATTERNS,
                AgentNode.GENERATE_STRATEGY,
            ],
            trace_fields=["trace_id", "information_status", "kyc_question_round_count"],
        ),
        WorkflowStepContract(
            name="generate_kyc_questions",
            state=AgentNode.GENERATE_KYC_QUESTIONS,
            description="根据缺失字段和 KYCQuestion 已问焦点生成一条低压补问，避免重复追问。",
            required_inputs=["missing_fields", "asked_focuses"],
            produced_outputs=["answer", "asked_focuses", "kyc_question_round_count"],
            allowed_next_states=[AgentNode.COMPLIANCE_REVIEW, AgentNode.POST_RESPONSE_LOGGER],
            trace_fields=["trace_id", "asked_focuses", "answer"],
        ),
        WorkflowStepContract(
            name="retrieve_dialogue_patterns",
            state=AgentNode.RETRIEVE_DIALOGUE_PATTERNS,
            description="检索 approved_for_generation=True 且非 high 风险的 DialoguePattern 摘要。",
            required_inputs=["compact_context", "target_persona", "trigger_module"],
            produced_outputs=["retrieved_dialogue_patterns"],
            allowed_next_states=[AgentNode.RETRIEVE_EXTERNAL_CONTEXT_IF_NEEDED, AgentNode.GENERATE_STRATEGY],
            guardrails=["sales_corpus_generation_boundary"],
            trace_fields=["trace_id", "retrieved_dialogue_patterns"],
        ),
        WorkflowStepContract(
            name="retrieve_external_context_if_needed",
            state=AgentNode.RETRIEVE_EXTERNAL_CONTEXT_IF_NEEDED,
            description="当 objective_material_need 存在时检索外部素材，并以 news_digest 进入 compact_context。",
            required_inputs=["objective_material_need"],
            produced_outputs=["metadata.news_digest"],
            allowed_next_states=[AgentNode.GENERATE_STRATEGY],
            tools_allowed=["web_search", "news_search"],
            trace_fields=["trace_id", "objective_material_need"],
        ),
        WorkflowStepContract(
            name="generate_strategy",
            state=AgentNode.GENERATE_STRATEGY,
            description="基于 compact_context 生成策略、话术或低压维护消息，不直接引用原始客户对话。",
            required_inputs=["compact_context"],
            produced_outputs=["answer"],
            allowed_next_states=[AgentNode.COMPLIANCE_REVIEW],
            trace_fields=["trace_id", "answer"],
        ),
        WorkflowStepContract(
            name="compliance_review",
            state=AgentNode.COMPLIANCE_REVIEW,
            description="输出前检查保证收益、恐吓营销、PII 泄露、Prompt 泄露和不当承诺。",
            required_inputs=["answer"],
            produced_outputs=["guardrail_results"],
            allowed_next_states=[AgentNode.POST_RESPONSE_LOGGER, AgentNode.HUMAN_APPROVAL],
            guardrails=["insurance_output_compliance"],
            trace_fields=["trace_id", "guardrail_results"],
        ),
        WorkflowStepContract(
            name="post_response_logger",
            state=AgentNode.POST_RESPONSE_LOGGER,
            description="记录 GeneratedOutput、使用的模式 ID 和 compact_context 摘要，形成可评测闭环。",
            required_inputs=["answer", "compact_context"],
            produced_outputs=["trace_events"],
            allowed_next_states=[AgentNode.FINAL],
            trace_fields=["trace_id", "answer", "retrieved_dialogue_patterns"],
        ),
        WorkflowStepContract(
            name="final_response",
            state=AgentNode.FINAL,
            description="返回最终响应，并保留 trace_id 供审计和回放。",
            required_inputs=["answer"],
            produced_outputs=["response_package", "final_state"],
            allowed_next_states=[],
            trace_fields=["trace_id", "final_state"],
        ),
    ],
)
