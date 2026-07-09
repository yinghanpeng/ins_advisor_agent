"""Agent state and explicit state transition model.

本文件定义 Agent 的显式状态机节点和运行时状态对象。

设计目标：
1. 不把复杂流程藏在一个大 Prompt 里；
2. 每一次状态变化都能被记录、审计和回放；
3. 即使 LangSmith 不可用，本地结构化日志也能追踪完整执行链路；
4. 为 Agent 执行器（graph/builder.py 的 AgentGraph）/ Dify API 调用提供统一状态对象。

核心概念：
- AgentNode：Agent 允许进入的显式状态节点；
- AgentState：一次 Agent 请求在状态机中流转时携带的完整上下文；
- state_transitions：只记录状态从哪里跳到哪里；
- trace_events：记录更完整的执行事件，例如状态跳转、工具调用、检索、风控、错误等。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.utils.ids import new_trace_id
from agent_core.utils.time import utc_now_iso


class AgentNode(StrEnum):
    """Agent 状态机中的所有显式节点。

    注意：
    1. Agent 的流程不应由大模型自由决定；
    2. 所有复杂任务都应该通过这些状态节点表达；
    3. 后续可以配合 allowed_transitions 校验状态是否合法跳转。
    """

    # 初始空闲状态：一次请求尚未开始处理。
    IDLE = "IDLE"

    # 问候状态：用户只是打招呼、问“你是谁”、询问 Agent 能做什么。
    GREETING = "GREETING"

    # 上下文初始化状态：为本次请求建立 trace、成本预算、运行配置和第一条用户消息。
    INIT_CONTEXT = "INIT_CONTEXT"

    # 输入安全状态：在任何记忆、检索、工具调用之前检查 prompt injection 和高风险输入。
    INPUT_GUARDRAIL = "INPUT_GUARDRAIL"

    # 短期记忆恢复状态：按 tenant/session/user 读取会话、任务和偏好记忆。
    RESTORE_MEMORY = "RESTORE_MEMORY"

    # 业务记忆读取状态：读取客户事实、从业者事实、机会 case 和已问 KYC 焦点。
    LOAD_BUSINESS_MEMORY = "LOAD_BUSINESS_MEMORY"

    # 消息标准化状态：把原始输入和历史上下文合并为统一 message 结构。
    NORMALIZE_MESSAGES = "NORMALIZE_MESSAGES"

    # 意图识别状态：判断用户到底想做什么，例如查天气、联网搜索、销售破冰、文件分析等。
    CLASSIFY_INTENT = "CLASSIFY_INTENT"

    # 语义风险分级状态：把请求分为 low/medium/high，供工具、人审和输出策略使用。
    SEMANTIC_RISK_CLASSIFICATION = "SEMANTIC_RISK_CLASSIFICATION"

    # 槽位抽取状态：抽取客户画像、任务参数、工具参数等结构化槽位。
    EXTRACT_SLOTS = "EXTRACT_SLOTS"

    # 槽位校验状态：判断关键槽位是否缺失，是否需要向用户澄清。
    VALIDATE_SLOTS = "VALIDATE_SLOTS"

    # Query Understanding 状态：完成指代消解、时间解析、实体抽取和检索 filters 生成。
    QUERY_UNDERSTANDING = "QUERY_UNDERSTANDING"

    # 上下文需求规划状态：判断本轮是否需要 Memory、RAG、Tool、Human 或 Reject。
    CONTEXT_NEED_PLANNING = "CONTEXT_NEED_PLANNING"

    # 能力路由状态：判断请求应该走通用能力、业务 Skill、销售智能层，还是人工审批。
    ROUTE_CAPABILITY = "ROUTE_CAPABILITY"

    # 通用工具路由状态：判断是否需要调用天气、搜索、计算器、网页读取等通用工具。
    GENERAL_TOOL_ROUTING = "GENERAL_TOOL_ROUTING"

    # Agentic 工具循环状态：在显式状态机内有界迭代规划、执行、观察和停止工具调用。
    AGENTIC_TOOL_LOOP = "AGENTIC_TOOL_LOOP"

    # 通用工具调用状态：实际执行通用工具调用。
    GENERAL_TOOL_CALL = "GENERAL_TOOL_CALL"

    # 工具结果校验状态：校验工具返回是否成功、格式是否正确、是否需要重试或降级。
    VERIFY_TOOL_RESULT = "VERIFY_TOOL_RESULT"

    # 澄清问题生成状态：在工具、RAG、模型生成前短路，向用户补问缺失关键信息。
    GENERATE_CLARIFICATION_RESPONSE = "GENERATE_CLARIFICATION_RESPONSE"

    # 通用回答生成状态：基于通用工具结果或通用知识生成最终回答。
    GENERAL_RESPONSE_GENERATION = "GENERAL_RESPONSE_GENERATION"

    # 业务工作流路由状态：判断进入哪个 Domain Skill，例如保险顾问、文档分析、研究助手等。
    DOMAIN_WORKFLOW_ROUTING = "DOMAIN_WORKFLOW_ROUTING"

    # 销售智能层路由状态：判断是否需要调用销售实战语料、话术库、异议处理库等。
    SALES_INTELLIGENCE_ROUTING = "SALES_INTELLIGENCE_ROUTING"

    # 销售语料导入状态：用于处理采访稿、转写稿等原始销售语料。
    SALES_CORPUS_INGESTION = "SALES_CORPUS_INGESTION"

    # 销售洞察抽取状态：从原始采访语料中抽取结构化销售经验卡片。
    SALES_INSIGHT_EXTRACTION = "SALES_INSIGHT_EXTRACTION"

    # 销售洞察检索状态：检索破冰、KYC、异议处理、案例、计划书等销售经验。
    SALES_INSIGHT_RETRIEVAL = "SALES_INSIGHT_RETRIEVAL"

    # 需求采集状态：当信息不足时，向用户补问必要信息，例如客户 KYC、场景、沟通目标。
    COLLECT_REQUIREMENTS = "COLLECT_REQUIREMENTS"

    # 状态更新状态：把用户补充的信息写入 session、profile、task memory 等运行时状态。
    UPDATE_STATE = "UPDATE_STATE"

    # KYC 分析与路由状态：产出 Dify KYC 分析节点的 18 个结构化字段。
    ANALYZE_KYC_AND_ROUTE = "ANALYZE_KYC_AND_ROUTE"

    # 记忆写入提案状态：把本轮明确事实、事件、问题和快照整理成写入计划。
    MEMORY_WRITE_PROPOSAL = "MEMORY_WRITE_PROPOSAL"

    # 记忆写入校验状态：检查证据、PII、生成建议误写等问题。
    VALIDATE_MEMORY_WRITE = "VALIDATE_MEMORY_WRITE"

    # 业务记忆快照持久化状态：把通过校验的事实、事件、分析和快照写入 store。
    PERSIST_MEMORY_SNAPSHOT = "PERSIST_MEMORY_SNAPSHOT"

    # 业务紧凑上下文构建状态：生成策略节点统一使用 compact_context。
    BUILD_COMPACT_CONTEXT = "BUILD_COMPACT_CONTEXT"

    # KYC 状态路由状态：按 information_status 选择补问、策略、低压维护或降级路径。
    STATUS_ROUTER = "STATUS_ROUTER"

    # KYC 补问生成状态：按缺失字段和已问焦点生成下一轮低压问题。
    GENERATE_KYC_QUESTIONS = "GENERATE_KYC_QUESTIONS"

    # 销售对话模式检索状态：只检索已审核、非高风险的 DialoguePattern。
    RETRIEVE_DIALOGUE_PATTERNS = "RETRIEVE_DIALOGUE_PATTERNS"

    # 外部上下文检索状态：必要时补充热点新闻或公开资料摘要。
    RETRIEVE_EXTERNAL_CONTEXT_IF_NEEDED = "RETRIEVE_EXTERNAL_CONTEXT_IF_NEEDED"

    # 策略生成状态：基于 compact_context 生成 KYC 策略、话术或维护消息。
    GENERATE_STRATEGY = "GENERATE_STRATEGY"

    # 响应后记录状态：把生成输出、使用的模式和 trace 摘要写回记忆系统。
    POST_RESPONSE_LOGGER = "POST_RESPONSE_LOGGER"

    # 任务规划状态：把复杂请求拆解成可执行步骤。
    PLAN_TASK = "PLAN_TASK"

    # 上下文检索状态：检索 RAG、知识库、销售经验库、历史记忆或外部事实。
    RETRIEVE_CONTEXT = "RETRIEVE_CONTEXT"

    # 上下文组装状态：将系统指令、用户输入、状态、检索证据、工具结果等组装成模型上下文。
    BUILD_CONTEXT = "BUILD_CONTEXT"

    # 知识融合状态：合并 Memory、RAG、Tool Result 和 Conversation，形成可信上下文。
    KNOWLEDGE_FUSION = "KNOWLEDGE_FUSION"

    # 上下文压缩状态：按 token/cost budget 压缩证据和历史消息。
    CONTEXT_COMPRESSION = "CONTEXT_COMPRESSION"

    # Prompt 组装状态：把 system、memory、history、RAG、tool result 和 user query 组装成 prompt。
    PROMPT_ASSEMBLY = "PROMPT_ASSEMBLY"

    # 模型路由状态：根据任务复杂度和成本压力选择模型。
    MODEL_ROUTING = "MODEL_ROUTING"

    # 回答生成状态：生成初版回复、策略、话术或结构化结果。
    GENERATE_RESPONSE = "GENERATE_RESPONSE"

    # 事实校验状态：检查回答是否有证据支撑，是否与工具结果或知识库冲突。
    GROUNDING_VERIFICATION = "GROUNDING_VERIFICATION"

    # 合规审查状态：检查输出是否有违规、夸大、幻觉、敏感信息泄露或高风险承诺。
    COMPLIANCE_REVIEW = "COMPLIANCE_REVIEW"

    # 输出侧 PII 扫描状态：最终答案返回前二次扫描并脱敏手机号、邮箱、身份证等信息。
    OUTPUT_PII_SCAN = "OUTPUT_PII_SCAN"

    # 回答质量评估状态：检查回答是否缺证据、未回答、幻觉或应澄清未澄清。
    EVALUATE_RESPONSE_QUALITY = "EVALUATE_RESPONSE_QUALITY"

    # 回答重生成状态：在预算内最多重生成一次，复用同一上下文和工具结果。
    REGENERATE_RESPONSE = "REGENERATE_RESPONSE"

    # 响应封装状态：生成前端可展示的 answer、引用、工具卡片和下一步建议。
    RESPONSE_PACKAGING = "RESPONSE_PACKAGING"

    # 短期记忆更新状态：把本轮问题、回答、槽位和任务状态写回 session memory。
    SHORT_TERM_MEMORY_UPDATE = "SHORT_TERM_MEMORY_UPDATE"

    # 长期记忆候选状态：判断哪些用户偏好或画像值得进入长期记忆候选。
    LONG_TERM_MEMORY_CANDIDATE = "LONG_TERM_MEMORY_CANDIDATE"

    # Trace 收尾状态：补齐最终 trace、成本、状态和审计字段。
    TRACE_FINALIZE = "TRACE_FINALIZE"

    # 人工审批状态：涉及发送、删除、发布、付款等高风险动作时，等待用户确认。
    HUMAN_APPROVAL = "HUMAN_APPROVAL"

    # 恢复状态：处理工具失败、检索无结果、JSON 解析失败、上下文超长等异常。
    RECOVERY = "RECOVERY"

    # 正常结束状态：任务成功完成并返回最终答案。
    FINAL = "FINAL"

    # 异常结束状态：任务无法恢复，进入错误终止。
    ERROR = "ERROR"


class AgentState(BaseModel):
    """一次 Agent 请求在状态机中流转时携带的完整运行时状态。

    这个对象可以理解为一次任务的“黑匣子”：
    - 记录用户输入；
    - 记录当前状态；
    - 记录意图和路由结果；
    - 记录客户画像和销售画像；
    - 记录检索 query 和检索结果；
    - 记录工具调用；
    - 记录风控审查；
    - 记录最终答案；
    - 记录错误、重试和成本；
    - 记录状态转移和 trace 事件。

    Agent 执行器的每个节点（graph/nodes.py 中的节点函数）都应该接收并返回
    AgentState，避免把流程藏在不可追踪的大 Prompt 里。
    """

    # Pydantic 默认保护 model_ 前缀；本项目需要 model_name 字段记录模型路由结果，所以关闭该限制。
    model_config = ConfigDict(protected_namespaces=())

    # =========================
    # 1. 请求身份与追踪字段
    # =========================

    trace_id: str = Field(
        default_factory=new_trace_id,
        description=(
            "本次请求的唯一追踪 ID。用于串联状态流转、工具调用、RAG 检索、"
            "合规审查、本地日志和 LangSmith trace。"
        ),
    )

    session_id: str = Field(
        default="anonymous_session",
        description=(
            "会话 ID。用于标识同一用户的多轮对话。"
            "多轮销售辅导、KYC 补问、上下文记忆都依赖该字段。"
        ),
    )

    user_id: str | None = Field(
        default=None,
        description=(
            "用户 ID。登录用户可写入真实 user_id；匿名用户可以为空。"
            "用于权限控制、用户画像、审计和后续计费。"
        ),
    )

    tenant_id: str = Field(
        default="local",
        description=(
            "租户 ID。用于 SaaS、多团队、多机构隔离。"
            "例如不同分公司、不同渠道、不同客户组织可以使用不同 tenant_id。"
        ),
    )

    # =========================
    # 2. 用户输入与工作流信息
    # =========================

    input_text: str = Field(
        default="",
        description="用户本轮输入的原始文本。所有意图识别、路由、检索和生成都从该字段开始。",
    )

    workflow_name: str = Field(
        default="universal_agent_workflow",
        description=(
            "当前运行的工作流名称。"
            "例如 universal_agent_workflow、weather_workflow、break_ice_assistant_workflow、"
            "sales_interview_ingestion_workflow。"
        ),
    )

    domain_skill: str | None = Field(
        default=None,
        description=(
            "当前命中的业务 Skill 名称。"
            "例如 insurance_advisor、research_assistant、document_analysis。"
            "通用能力请求可以为空。"
        ),
    )

    # =========================
    # 3. 当前状态字段
    # =========================

    current_state: AgentNode = Field(
        default=AgentNode.IDLE,
        description=(
            "当前状态机节点。所有节点切换都应通过 move_to() 完成，"
            "不要在业务代码里直接修改该字段。"
        ),
    )

    final_state: AgentNode | None = Field(
        default=None,
        description=(
            "最终结束状态。正常结束为 FINAL，异常结束为 ERROR。"
            "任务未结束时该字段为空。"
        ),
    )

    allowed_transitions: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "预留的状态转移白名单。key 为当前状态，value 为允许进入的下一批状态。"
            "默认空字典表示本地演示模式暂不强制校验；生产环境可以写入完整状态图，"
            "由 move_to() 在状态切换前统一拦截非法跳转。"
        ),
    )

    # =========================
    # 4. 意图识别与路由字段
    # =========================

    intent: str | None = Field(
        default=None,
        description=(
            "用户意图识别结果。"
            "例如 weather_query、web_search、break_ice_help、objection_handling、file_summary。"
        ),
    )

    capability_route: str | None = Field(
        default=None,
        description=(
            "能力路由结果。用于判断请求走通用能力、业务 Skill、销售智能层还是人工审批。"
            "例如 general_capability、domain_skill、sales_intelligence、human_approval。"
        ),
    )

    sales_route: str | None = Field(
        default=None,
        description=(
            "销售智能层的具体路由。"
            "例如 kyc_question、break_ice、macro_resonance、case_evidence、"
            "objection_handling、proposal_closing。"
        ),
    )

    # =========================
    # 5. 用户画像与业务画像字段
    # =========================

    profile: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "客户画像。用于保存被沟通客户的 KYC 信息，例如年龄、职业、家庭、资产偏好、"
            "决策方式、风险偏好、沟通触发点等。"
        ),
    )

    practitioner: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "使用者画像，也就是销售 / 顾问 / 小白从业者的画像。"
            "例如销售年限、渠道、当前短板、需要的帮助类型、偏好的输出风格。"
        ),
    )

    profile_state: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dify KYC 分析节点输出的客户画像结构。"
            "它是本轮工作记忆快照，不等同于长期客户事实表；长期事实应写入 CustomerProfileFact。"
        ),
    )

    practitioner_state: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dify KYC 分析节点输出的从业者画像结构。"
            "它是本轮工作记忆快照，不等同于长期从业者事实表；长期事实应写入 AdvisorProfileFact。"
        ),
    )

    information_status: str = Field(
        default="insufficient",
        description="KYC 信息状态。insufficient 进入补问；matched 进入策略；unmatched 进入低压维护。",
    )

    subject_type: str = Field(
        default="unclear",
        description="当前沟通对象类型，例如 customer、channel、unclear。",
    )

    target_persona: str = Field(
        default="unknown",
        description="内部客群标签，例如 enterprise_owner、executive、family_planner、channel。",
    )

    advisor_stage: str = Field(
        default="unknown",
        description="从业者阶段，例如 newbie、transitioning、part_time、experienced。",
    )

    missing_fields: list[str] = Field(
        default_factory=list,
        description="当前仍缺失的关键 KYC 字段，用于生成低压补问。",
    )

    match_evidence: str = Field(
        default="",
        description="KYC 分析使用的明确事实证据。只能写用户或资料中可回溯的事实，不写推测。",
    )

    route_reason: str = Field(
        default="",
        description="当前 information_status 的路由原因，用于审计为什么补问、生成策略或低压维护。",
    )

    kyc_completeness_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="KYC 完整度分。只保存结果，不在 compact_context 中输出内部评分公式。",
    )

    opportunity_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="机会推进分。必须能追溯到 AnalysisRun，不应作为不可解释魔法数字流转。",
    )

    external_grade: str = Field(
        default="D",
        description="对从业者展示的外部等级，例如 A/B/C/D。",
    )

    trigger_module: str = Field(
        default="unknown",
        description="当前最适合的销售切入模块，例如 cashflow_pressure、family_responsibility。",
    )

    current_stage: str = Field(
        default="collect_kyc",
        description="当前沟通阶段，例如 collect_kyc、deep_conversation、cultivate、low_pressure_end。",
    )

    objective_material_need: str = Field(
        default="",
        description="本轮策略生成是否需要外部客观素材，例如热点新闻、利率变化或行业公开信息。",
    )

    support_note: str = Field(
        default="",
        description="面向从业者的鼓励摘要，用来降低新手焦虑，不作为客户事实写入长期画像。",
    )

    kyc_question_round_count: int = Field(
        default=0,
        ge=0,
        description="KYC 补问轮次。统一最多 4 轮，第 5 轮后禁止继续停在 insufficient。",
    )

    asked_focuses: list[str] = Field(
        default_factory=list,
        description="已经问过的 KYC 焦点。生产应来自 KYCQuestion 表或 store，而不是字符串拼接。",
    )

    slot_values: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "本轮槽位抽取结果。可保存客户类型、公司实体、时间范围、工具参数、"
            "澄清缺口等结构化信息，供后续 Query Understanding、Tool Planning 和生成节点使用。"
        ),
    )

    # =========================
    # 6. 消息、状态流转与 Trace 记录
    # =========================

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "对话消息或轻量事件记录。建议主要用于保存用户消息和助手消息。"
            "状态变化建议使用 state_transitions；完整事件建议使用 trace_events。"
        ),
    )

    state_transitions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "状态转移审计记录。只记录 from_state、to_state、reason、metadata 等状态跳转信息。"
            "用于回放 Agent 从开始到结束的流程路径。"
        ),
    )

    trace_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "完整 trace 事件列表。可记录状态跳转、工具调用、RAG 检索、销售洞察检索、"
            "合规审查、错误、重试、成本统计等所有关键事件。"
        ),
    )

    # stream_events 是 trace 的流式友好视图，API 未来可以直接转换成 SSE 事件。
    stream_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "流式事件骨架列表。当前用于记录节点开始/结束、工具调用、最终答案等事件，"
            "未来 API 可将其转换为 SSE；测试会检查它是否覆盖主链路关键节点。"
        ),
    )

    # streaming_enabled 只表示调用端是否希望实时消费事件；当前版本即使为 False 也保留事件骨架。
    streaming_enabled: bool = Field(
        default=False,
        description=(
            "是否启用实时流式输出的请求级开关。第一版不做 token-by-token streaming，"
            "但仍会写入 stream_events，方便未来无破坏接入 SSE。"
        ),
    )

    normalized_messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "标准化后的多轮消息。每条消息建议包含 role、content、ts、source。"
            "它和 messages 的区别是：normalized_messages 用于模型上下文，messages 更偏轻量运行记录。"
        ),
    )

    # =========================
    # 7. 检索与上下文字段
    # =========================

    rewritten_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Query Rewrite 后生成的检索 query。"
            "例如用户问“客户不想聊保险怎么办”，系统可能改写成多个销售异议处理检索 query。"
        ),
    )

    retrieved_context: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "RAG、知识库、网页、工具或销售经验库返回的检索上下文。"
            "每条记录建议包含 source、chunk_id、score、content、metadata。"
        ),
    )

    query_understanding: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Query Understanding 结果。用于保存指代消解、时间解析、实体抽取、"
            "Query Rewrite、检索 filters 等结构化信息。"
        ),
    )

    context_needs: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "上下文需求规划结果。用于声明本轮是否需要 memory、rag、tool、human、reject、"
            "以及是否需要澄清。后续 graph 根据该字段选择分支。"
        ),
    )

    memory_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "从短期/任务/偏好记忆中恢复的上下文。"
            "该字段只保存本轮需要用到的记忆摘要，不直接暴露底层存储对象。"
        ),
    )

    memory_recall_decision: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "长期记忆按需召回决策。记录本轮是否需要召回 preference、客户画像、"
            "从业者画像或 case 记忆，以及触发或跳过的原因。"
        ),
    )

    memory_recall_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "经过 hybrid search 和 rerank 后进入上下文候选的长期记忆摘要。"
            "它只保存 TopK 结果，不保存完整长期记忆库。"
        ),
    )

    sales_insight_digest: dict[str, Any] | None = Field(
        default=None,
        description=(
            "销售实战智能层压缩后的摘要。"
            "不应直接把原始采访长文塞给生成模型，而应先压缩成适用场景、经验摘要、"
            "可用话术、禁用表达、下一步建议和来源。"
        ),
    )

    compact_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "KYC 策略生成节点优先使用的紧凑上下文。"
            "它合并 confirmed/uncertain 客户事实、从业者事实、case 状态、已问焦点、"
            "缺失字段、已审核销售模式和新闻摘要，并过滤 PII 与原始对话全文。"
        ),
    )

    memory_write_proposal: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "本轮业务记忆写入提案。它只描述准备写入的事实、事件、问题、快照和分析运行，"
            "真正落库前必须经过 validate_memory_writes。"
        ),
    )

    memory_write_validation: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮记忆写入提案的校验结果，记录允许写入和被阻断的事实 ID。",
    )

    retrieved_dialogue_patterns: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "检索到的销售对话模式摘要。只能包含 approved_for_generation=True 且非 high 风险的模式，"
            "不得包含原始 CorpusMessage 全文。"
        ),
    )

    # =========================
    # 8. 工具调用与风控字段
    # =========================

    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "本次请求执行过的工具调用记录。"
            "建议记录 tool_name、input、output_summary、status、latency_ms、retry_count、error。"
        ),
    )

    tool_plan: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "工具规划结果。每条计划建议包含 tool_name、arguments、risk_level、permission_scope、"
            "requires_approval，用于执行前校验和 trace。"
        ),
    )

    tool_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "工具执行结果列表。与 tool_calls 的区别是：tool_calls 偏审计调用过程，"
            "tool_results 偏下游知识融合和回答生成可消费的结构化结果。"
        ),
    )

    # tool_loop_config 保存本次工具循环预算，允许 API/测试通过 metadata 或 state 覆盖默认值。
    tool_loop_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Agentic 工具循环配置快照，例如 max_iterations、max_total_tool_calls、"
            "是否启用模型 planner。用于 trace、API 展示和测试覆盖预算边界。"
        ),
    )

    # tool_loop_iterations 保存每轮计划、执行和 observation，不保存模型隐藏推理链。
    tool_loop_iterations: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "工具循环每轮迭代记录。每项包含 decision、tool_calls、observations 和停止原因，"
            "用于审计为什么继续或停止工具调用。"
        ),
    )

    # tool_loop_stop_reason 记录工具循环最终停止原因，方便排查 max_iterations 或重复计划风险。
    tool_loop_stop_reason: str | None = Field(
        default=None,
        description=(
            "工具循环停止原因，例如 finished、max_iterations、repeated_tool_plan、"
            "tool_error_budget_exceeded 或 human_approval。"
        ),
    )

    # tool_loop_status 记录工具循环阶段，API 可用它展示 idle/running/finished/stopped。
    tool_loop_status: str = Field(
        default="idle",
        description="工具循环状态，取值通常为 idle、running、finished 或 stopped。",
    )

    # tool_loop_budget 汇总本轮工具预算消耗，避免只看 cost 时分不清工具循环内部限制。
    tool_loop_budget: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "工具循环预算消耗摘要，例如 max_iterations、used_iterations、"
            "max_total_tool_calls、used_tool_calls 和 error_count。"
        ),
    )

    # agentic_loop_enabled 表示通用工具链是否启用有界迭代循环；关闭时可回退旧单轮工具路径。
    agentic_loop_enabled: bool = Field(
        default=True,
        description=(
            "是否启用 Agentic 工具迭代循环。默认开启；测试或灰度可以关闭并回退旧工具节点。"
        ),
    )

    guardrail_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "安全、合规、权限、输出审查结果。"
            "例如是否触发保证收益、避税避债、恐吓式营销、敏感信息泄露、prompt injection 等规则。"
        ),
    )

    # =========================
    # 9. 输出、错误、重试与成本字段
    # =========================

    answer: str | None = Field(
        default=None,
        description="最终返回给用户的答案。任务未完成或生成失败时可以为空。",
    )

    risk_level: str = Field(
        default="low",
        description=(
            "本轮请求的语义风险等级。建议取 low、medium、high。"
            "工具规划、人审、输出策略和审计都可以复用该字段。"
        ),
    )

    knowledge_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "知识融合后的统一上下文。用于合并 Memory、RAG、Tool Result、Conversation，"
            "并记录去重、冲突检测和来源信息。"
        ),
    )

    compressed_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "压缩后的上下文。用于在 Prompt Assembly 前控制 token 预算，"
            "保留最关键的证据、工具结果、记忆和用户问题。"
        ),
    )

    assembled_prompt: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "最终发送给配置化模型的 prompt 结构。"
            "其中必须保留 system、context、user、source boundary 等边界，便于审计和 replay。"
        ),
    )

    model_name: str | None = Field(
        default=None,
        description="模型路由选择结果。预算紧张时可选择小模型，复杂任务可选择 reasoning 模型。",
    )

    grounding_result: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "事实校验结果。用于记录回答是否有来源支撑、是否与工具结果冲突、"
            "是否需要重新生成或拒答。"
        ),
    )

    response_package: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "封装后的前端响应结构。可包含 answer、citations、tool_cards、next_actions、"
            "risk_level、trace_id 等用户可见或前端可消费字段。"
        ),
    )

    # clarification_question 保存本轮短路澄清问题，便于 API 区分“已回答”和“需要用户补充”。
    clarification_question: str | None = Field(
        default=None,
        description=(
            "澄清短路分支生成的问题。context_needs.clarify=True 时写入，"
            "用于 response_package、trace 和前端展示下一轮应补充的信息。"
        ),
    )

    # evaluation_result 保存 evaluator-optimizer 对候选回答的质量判断和触发原因。
    evaluation_result: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "运行时回答质量评估结果。记录是否通过、是否需要重生成、触发维度、"
            "降级警告等信息，供测试和生产审计使用。"
        ),
    )

    # regeneration_attempts 记录回答重生成次数，硬限制最多一次，防止生成闭环无限循环。
    regeneration_attempts: int = Field(
        default=0,
        ge=0,
        description="回答重生成尝试次数。默认最多 1 次，用于防止 evaluator-optimizer 无限循环。",
    )

    # output_pii_scan_result 保存输出侧二次 PII 扫描结果，只记录类型和位置摘要，不记录原始 PII。
    output_pii_scan_result: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "输出侧 PII 二次扫描结果。仅保存 PII 类型、位置摘要、是否脱敏和高敏标记，"
            "不保存手机号、身份证、邮箱、银行卡等原始敏感文本。"
        ),
    )

    memory_write_candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "长期记忆候选列表。只保存经过分类、去重和敏感性判断后可能值得写入长期记忆的内容。"
        ),
    )

    errors: list[str] = Field(
        default_factory=list,
        description=(
            "运行过程中发生的错误列表。"
            "例如 tool_timeout、json_parse_failed、rag_no_result、langsmith_unavailable。"
        ),
    )

    retry_count: int = Field(
        default=0,
        description=(
            "当前任务或当前关键步骤已经重试的次数。"
            "用于防止无限循环，并辅助 Recovery / Cost Control 判断是否降级。"
        ),
    )

    cost: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "成本与资源消耗记录。"
            "可记录 input_tokens、output_tokens、model、tool_call_count、estimated_cost、latency_ms。"
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "扩展元数据。用于保存暂时没有专门字段但需要保留的信息，"
            "例如 client=dify、channel=web、debug=true、experiment_group=A。"
        ),
    )

    def add_trace_event(self, event: str, **fields: Any) -> None:
        """追加一条结构化 trace 事件。

        使用场景：
        - 状态发生变化；
        - 工具开始或结束调用；
        - RAG 检索完成；
        - 销售经验检索完成；
        - 触发合规规则；
        - 进入重试或降级；
        - 发生异常。

        设计目的：
        1. 即使 LangSmith 关闭，本地也能保留完整审计轨迹；
        2. 后续 workflow engine 可以统一把 trace_events 写入结构化日志；
        3. 面试或排障时可以清楚解释 Agent 每一步做了什么。

        Args:
            event: 事件名称，例如 state_transition、tool_called、rag_retrieved。
            **fields: 该事件的额外结构化字段。
        """
        # trace_events 是完整审计事件流，所有事件都带统一身份字段，便于日志系统按 trace/session/workflow 聚合。
        self.trace_events.append(
            {
                # ts 使用 UTC ISO 字符串，保证跨机器、跨时区日志可排序。
                "ts": utc_now_iso(),
                # event 记录事件类型，例如 state_transition、tool_called、rag_retrieved。
                "event": event,
                # trace_id 串联一次 Agent 请求的所有节点、工具和检索事件。
                "trace_id": self.trace_id,
                # session_id 让多轮对话排障时能跨请求关联。
                "session_id": self.session_id,
                # workflow_name 方便同一服务内区分通用 Agent、保险顾问、销售语料导入等不同工作流。
                "workflow_name": self.workflow_name,
                # domain_skill 记录命中的业务 Skill，通用请求可能为空。
                "domain_skill": self.domain_skill,
                # fields 承载调用方传入的节点名、工具结果、检索摘要、成本等事件细节。
                **fields,
            }
        )

    def move_to(
        self,
        node: AgentNode,
        *,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "AgentState":
        """切换到下一个状态，并记录状态转移审计日志。

        这是状态机推荐的唯一状态切换入口。
        业务代码不应直接修改 current_state，否则状态变化无法被审计、回放和追踪。

        本方法会同时写入：
        1. messages：保留一条轻量事件，方便调试；
        2. state_transitions：专门记录状态跳转；
        3. trace_events：纳入完整 trace 事件流；
        4. current_state：更新当前状态；
        5. final_state：如果进入 FINAL 或 ERROR，则记录最终状态。

        后续生产增强建议：
        - 已预留 allowed_transitions 校验入口，防止非法状态跳转；
        - 增加状态跳转失败时的错误记录；
        - 增加 LangSmith run metadata 同步。

        Args:
            node: 目标状态节点。
            reason: 状态切换原因，方便日志排查和面试讲解。
            metadata: 状态切换相关的额外信息，例如 intent、route、tool_name。

        Returns:
            更新后的 AgentState，方便节点函数链式调用并返回。
        """
        # 读取当前状态允许进入的下一批状态；None 表示本地演示模式不强制校验。
        allowed_next_states = self.allowed_transitions.get(self.current_state.value)
        # 如果配置了 allowed_transitions 且目标节点不在白名单内，就立即阻断非法跳转。
        if allowed_next_states is not None and node.value not in allowed_next_states:
            raise ValueError(
                f"非法状态跳转：{self.current_state.value} -> {node.value}。"
                "如需允许该路径，请在 allowed_transitions 中显式配置。"
            )

        # transition 只描述状态跳转本身，不混入工具结果、检索结果或模型输出。
        transition = {
            # ts 记录跳转发生时间，方便回放状态路径。
            "ts": utc_now_iso(),
            # trace_id 把状态跳转挂到同一次请求。
            "trace_id": self.trace_id,
            # from_state 是跳转前状态。
            "from_state": self.current_state.value,
            # to_state 是目标状态。
            "to_state": node.value,
            # reason 说明为什么发生这次跳转，例如 intent_classified 或 output_guardrail_passed。
            "reason": reason,
            # metadata 保存跳转相关的小型结构化补充信息。
            "metadata": metadata or {},
        }

        # 保留一条轻量事件记录。后续如果希望 messages 只保存对话消息，
        # 可以移除这行，把状态变化完全交给 state_transitions 和 trace_events。
        self.messages.append({"type": "state_transition", **transition})

        # 专门用于审计状态跳转路径。
        self.state_transitions.append(transition)

        # 纳入完整 trace 事件流，便于统一写入本地日志或 LangSmith。
        self.add_trace_event("state_transition", **transition)

        # 更新当前状态。
        self.current_state = node

        # 如果进入终止节点，记录最终状态。
        if node in {AgentNode.FINAL, AgentNode.ERROR}:
            self.final_state = node

        return self
