"""Agent 执行图（线性顺序写法）。

# 文件说明：
# - 本文件是 Agent 主链路的唯一编排实现，采用"从上往下一步一步"的线性写法，方便直接顺着读懂流程。
# - 一次请求的入口是 WorkflowEngine.run() → self.graph.invoke(state)，self.graph 就是这里的 AgentGraph。
# - 两条链路：
#     1. 通用主链路      _run_universal：意图 → 工具/领域 RAG → 生成 → 风控 → 记忆；
#     2. KYC 教练链路    _run_kyc：业务记忆分析 → 写入 → 策略生成。
#   两条链路在 initialize_context + input_guardrail 之后按 workflow_name 分叉。
"""

from __future__ import annotations

from typing import Any

from agent_core.graph import nodes
from agent_core.graph.state import AgentNode, AgentState
from agent_core.memory.business_store import BusinessMemoryStore
from agent_core.memory.manager import MemoryManager


class AgentGraph:
    """Agent 主链路执行器：按固定顺序一步步调用各节点函数。"""

    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        business_store: BusinessMemoryStore | None = None,
    ) -> None:
        """注入记忆管理器和业务记忆 store，供各节点按需使用。"""
        # 短期/任务/偏好记忆管理器；同一个 WorkflowEngine 实例内多轮对话复用同一份。
        self.memory_manager = memory_manager
        # KYC 业务记忆 store；保存客户事实、机会 case、生成输出等。
        self.business_store = business_store

    def invoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        """执行一次完整请求：公共入口 → 按 workflow_name 分叉到对应链路。"""
        # 调用方可能传 dict，这里统一恢复成 AgentState，后续节点只处理一种类型。
        if isinstance(state, dict):
            state = AgentState(**state)

        # 第 1 步：初始化 trace、成本预算、用户消息（KYC 默认字段也在此设置）。
        #   模拟：input = "它最近有没有融资，重点看过去三个月的英文报道"
        #   产出（部分）：
        #     trace_id = "…uuid…"                                   // 贯穿全链路的追踪 ID
        #     cost     = { "request_token_budget": 12000, … }       // 成本/预算初始化
        #     messages = [ { "role": "user", "content": "它最近有没有融资，重点看过去三个月的英文报道" } ]
        state = nodes.initialize_context(state)
        # 第 2 步：输入安全检查（硬闸 + LLM Judge 灰区 + PolicyCombiner）。
        #   产出 guardrail_results[-1]：
        #     {
        #       "guardrail_name": "input_prompt_injection",
        #       "action": "pass",              // 汇总动作：pass / block
        #       "decision_action": "allow",    // 精细动作：allow / mask / review / block
        #       "risk_level": "low",
        #       "masked": false,               // 命中 PII 时为 true，并回填 sanitized_text
        #       "signals": []                  // 命中的证据链（硬闸 / PII / LLM Judge）
        #     }
        #   → 若 decision_action = block / review：state.final_state = ERROR，直接返回不再往下走。
        state = nodes.input_guardrail(state)
        # 输入被风控阻断时直接安全终止，不进入任何后续节点。
        if state.final_state == AgentNode.ERROR:
            return state

        # 第 3 步：按 workflow_name 分叉。KYC 教练走业务记忆链路，其余走通用主链路。
        if state.workflow_name == "insurance_kyc_coach_workflow":
            return self._run_kyc(state)
        return self._run_universal(state)

    def _run_universal(self, state: AgentState) -> AgentState:
        """通用 Agent 主链路：意图 → 工具/领域 RAG → 生成 → 风控 → 记忆。

        下面每一步的"模拟"注释用同一个多轮场景演示真实产出：
        第 1 轮已聊过 Anthropic，本轮 input="它最近有没有融资，重点看过去三个月的英文报道"。
        （分支 B 领域链路另用保险场景 input="帮我给这个企业主客户做保险破冰，怎么开口" 演示。）
        """
        # 读取 session/task/preference 三层记忆。
        #   产出 state.memory_context：
        #     {
        #       "session": {                          // 短期会话原始记忆（第 1 轮写入）
        #         "recent_messages": [ {user:"帮我看看 Anthropic…"}, {assistant:"Anthropic 是…"} ],
        #         "last_intent": "general_chat",
        #         "last_entity": "Anthropic",         // 指代锚点："它" = Anthropic 靠它
        #         "slot_values": { "company": "Anthropic", "missing_slots": [] }
        #       },
        #       "task": { "current_state": "FINAL", "final_answer_ready": true },
        #       "preference": {},                     // 长期偏好召回结果（本轮判定不召回 → 空）
        #       "long_term_recall": {                 // 召回决策全过程（可审计）
        #         "decision": { "should_recall": false, "status": "model_unavailable_safe_skip" },
        #         "items": []                         // 锚点"Anthropic"已在短期消息内 → 不触发长期召回
        #       }
        #     }
        state = nodes.restore_memory(state, self.memory_manager)
        # 合并历史消息与本轮输入为标准消息结构。
        #   产出 state.normalized_messages（历史 2 条 + 追加本轮 1 条）：
        #     [
        #       { "role": "user",      "content": "帮我看看 Anthropic 这家公司" },              // 历史
        #       { "role": "assistant", "content": "Anthropic 是一家 AI 公司…" },               // 历史
        #       { "role": "user",      "content": "它最近有没有融资…", "source": "current_turn" }  // 本轮
        #     ]
        state = nodes.normalize_messages(state)
        # 意图识别（模型优先、关键词兜底），决定走通用能力还是领域 Skill。
        #   产出：
        #     intent           = "web_or_news_search"
        #     capability_route = "general"        // general=通用工具层 / domain=业务 Skill
        #     domain_skill     = None
        state = nodes.classify_intent(state)
        # 语义风险分级，供工具权限与输出策略复用。
        #   产出：
        #     risk_level = "medium"   // 命中"融资 / 英文报道"等中风险词（取值 low / medium / high）
        state = nodes.semantic_risk_classification(state)
        # 抽取客户画像、公司实体、语言、主题等槽位。
        #   产出 state.slot_values：
        #     {
        #       "resolved_entity": "Anthropic",   // 由"它"结合 last_entity 消解得到
        #       "language": "en",                 // "英文报道" → language filter
        #       "topic": "funding"                // "融资" → topic filter
        #     }
        state = nodes.extract_slots(state)
        # 校验关键槽位是否缺失。
        #   产出：
        #     slot_values.missing_slots          = []      // 无缺失关键槽位
        #     slot_values.clarification_required = false   // 无需追问澄清
        state = nodes.validate_slots(state)
        # 指代消解、时间解析、query rewrite 和 filters 生成。
        #   产出 state.query_understanding：
        #     {
        #       "entity": "Anthropic",
        #       "resolved_query": "Anthropic最近有没有融资，重点看过去三个月的英文报道",  // "它"替换回实体
        #       "rewritten_query": "Anthropic funding news in the past three months",      // 供外部检索
        #       "date_range": { "start": "2026-04-08", "end": "2026-07-09" },              // "过去三个月"→绝对区间
        #       "filters": { "language": "en", "source_type": "news", "date_range": {…} }
        #     }
        state = nodes.query_understanding(state)
        # 规划本轮需要 memory/RAG/tool 中的哪些能力。
        #   产出 state.context_needs：
        #     {
        #       "memory": true, "long_term_memory": false, "rag": false,
        #       "tool": true,        // → 进入下面的分支 A（工具链）
        #       "human": false, "reject": false, "clarify": false
        #     }
        state = nodes.context_need_planning(state)

        # Clarify 短路分支：如果 Context Need 已经判定缺关键槽位，
        # 这里必须在工具/RAG/生成大模型之前中断，直接向用户补问。
        if state.context_needs.get("clarify"):
            state = nodes.generate_clarification_response(state)
            state = nodes.response_packaging(state)
            state = nodes.trace_finalize(state)
            return state

        # 分支 A：需要外部工具（天气/搜索/计算等）时走工具链。
        if state.context_needs.get("tool"):
            # 补充说明：生产级工具链不再只执行一次 routing/call/verify，
            # 而是进入有界 agentic_tool_loop；loop 内部仍复用 general_tool_routing、
            # general_tool_call 和 verify_tool_result，并在每轮执行工具 Guardrail。
            # 生成工具调用计划。
            #   产出 state.tool_plan：
            #     [
            #       {
            #         "tool_name": "summarizer",
            #         "arguments": { "text": "它最近有没有融资…", "max_chars": 300 },
            #         "risk_level": "low",
            #         "permission_scope": "llm.transform",
            #         "requires_approval": false
            #       }
            #     ]
            state = nodes.agentic_tool_loop(state)
            # 执行工具（含权限与人审检查）。
            #   产出：
            #     tool_calls   = [ { "tool_name": "summarizer", "status": "success", "latency_ms": 0 } ]
            #     tool_results = [ { "name": "summarizer", "status": "success",
            #                        "output": { "summary": "…",
            #                                    "_source_boundary": { "trust": "untrusted_external_context" } } } ]
            #     // 工具结果带 source_boundary：只作事实候选，不可当指令执行
            state = nodes.general_tool_call(state)
            # 工具触发人工审批时停在 HUMAN_APPROVAL，等待审批恢复。
            #   本例 requires_approval=false，不进入此分支。
            if state.current_state == AgentNode.HUMAN_APPROVAL:
                return state
            # 工具循环中 planner 如果判断需要补充信息，也在这里转入澄清短路返回。
            if state.context_needs.get("clarify"):
                state = nodes.generate_clarification_response(state)
                state = nodes.response_packaging(state)
                state = nodes.trace_finalize(state)
                return state
            # 校验工具结果，失败进入恢复但仍走保守回答。
            #   产出：
            #     errors      = []   // 无失败结果
            #     retry_count = 0    // 无需 recovery / 降级
            # 补充说明：verify_tool_result 已在 agentic_tool_loop 每轮内部执行，
            # 这里不再重复调用，避免同一轮工具错误被重复计入 retry_count。
        # 分支 B：保险顾问等领域请求走 Domain Skill + 销售检索。
        #   （分支 B 用另一场景演示：input="帮我给这个企业主客户做保险破冰，怎么开口"）
        elif state.capability_route == "domain":
            # 明确领域工作流。
            #   产出：
            #     sales_route   = "break_ice_assistant_workflow"
            #     current_state = "SALES_INTELLIGENCE_ROUTING"
            state = nodes.route_domain_workflow(state)
            # 检索已审核销售洞察卡片。
            #   产出：
            #     rewritten_queries = [ "帮我给这个企业主客户做保险破冰，怎么开口",
            #                           "话术策略 帮我给这个企业主客户做保险破冰，怎么开口" ]
            #     retrieved_context = [
            #       { "source_id": "sample_interview_001", "customer_type": "企业主",
            #         "scene": "饭局破冰", "business_stage": "new_customer", … }   // 仅已审核、非高风险卡片
            #     ]
            state = nodes.retrieve_sales_intelligence(state)
            # 把检索证据压缩成生成上下文。
            #   产出 state.sales_insight_digest：
            #     {
            #       "applicable_scene": "insurance_advisor",
            #       "digest": "先围绕经营现金流和家庭责任共情，再用资金分层把话题转到长期稳定安排。",
            #       "forbidden": [ "承诺收益", "避税避债", "恐吓营销", "编造案例", "贬低其他产品" ]
            #     }
            state = nodes.build_context(state)
        # 分支 C：既不需要工具也不需要领域检索时，直接进入下面的融合与生成。

        # 融合 memory/RAG/工具/对话为统一可信上下文。
        #   产出 state.knowledge_context：
        #     {
        #       "memory": { "session": {…} },          // 来自 restore_memory
        #       "tool_results": [ { summarizer … } ],  // 来自工具分支
        #       "retrieved_context": [ … ],            // 来自领域分支（如有）
        #       "conflicts": []                        // 预留冲突标记
        #     }
        state = nodes.knowledge_fusion(state)
        # 按预算压缩上下文。
        #   产出：
        #     compressed_context = {…按 token 预算裁剪后的 memory / context / tool 摘要…}
        #     cost.compressed_context_chars ≈ 1468
        state = nodes.compress_context(state)
        # 组装最终 prompt 结构。
        #   产出 state.assembled_prompt：
        #     {
        #       "system": "你是合规、低压、证据优先的保险顾问沟通助手。",
        #       "memory": {…}, "context": {…},
        #       "user":   "它最近有没有融资，重点看过去三个月的英文报道"
        #     }
        state = nodes.prompt_assembly(state)
        # 按风险/预算/复杂度选择模型。
        #   产出：
        #     model_name = "reasoning-model"   // 中风险 + 需综合证据 → 走推理模型（否则 fast-model）
        state = nodes.model_routing(state)
        # 生成初版回答。
        #   产出 state.answer：
        #     "工具 summarizer 已返回结果：{…}。"   // 工具类问题优先用工具结果，不自由发挥
        state = nodes.generate_response(state)
        # 事实校验，降低幻觉。
        #   产出 state.grounding_result：
        #     { "grounded": true, "evidence_sources": [ "summarizer" ], "conflicts": [] }
        state = nodes.grounding_verification(state)
        # 输出前合规审查。
        #   产出 guardrail_results[-1]：
        #     { "guardrail_name": "insurance_output_compliance", "action": "pass", "triggered": false }
        #   → current_state = "RESPONSE_PACKAGING"（未命中违规承诺 / 恐吓营销等）
        state = nodes.compliance_review(state)
        # 输出命中高风险时停在 HUMAN_APPROVAL。
        #   本例合规通过，不进入此分支。
        if state.current_state == AgentNode.HUMAN_APPROVAL:
            return state

        # 输出侧 PII 二次扫描：检查生成答案中是否包含手机号、邮箱、身份证、银行卡等。
        state = nodes.output_pii_scan(state)

        # Evaluator-optimizer 有界闭环：只在证据不足、风险较高、PII 脱敏等情况下最多重生成一次。
        state = nodes.evaluate_response_quality(state)
        state = nodes.regenerate_response_if_needed(state)

        # 重生成后必须再次执行 PII、grounding 和 compliance，避免优化后引入新的风险。
        state = nodes.output_pii_scan(state)
        state = nodes.grounding_verification(state)
        state = nodes.compliance_review(state)
        # 第二次合规审查仍可能进入 HUMAN_APPROVAL，必须立即返回等待审批。
        if state.current_state == AgentNode.HUMAN_APPROVAL:
            return state

        # 封装前端可消费的响应包。
        #   产出 state.response_package：
        #     {
        #       "answer": "…", "citations": [], "tool_cards": [ {summarizer…} ],
        #       "next_actions": [ … ], "risk_level": "medium", "trace_id": "…"
        #     }
        state = nodes.response_packaging(state)
        # 更新短期 session/task 记忆。
        #   写回 SESSION / TASK：
        #     session = { "recent_messages": [含本轮问答], "last_intent": "web_or_news_search",
        #                 "last_entity": "Anthropic", "slot_values": {…} }
        #     task    = { "current_state": "…", "final_answer_ready": true }
        state = nodes.update_short_term_memory(state, self.memory_manager)
        # 判断并写入长期偏好记忆候选。
        #   产出：
        #     memory_write_candidates = []   // 本轮无值得跨会话长存的偏好 / 画像
        state = nodes.long_term_memory_candidate(state, self.memory_manager)
        # trace 与成本收尾，推进到 FINAL。
        #   产出：
        #     final_state = "FINAL"
        #     cost        = { "tool_call_count": 1, "output_chars": 267, "trace_event_count": 54, … }
        state = nodes.trace_finalize(state)
        return state

    def _run_kyc(self, state: AgentState) -> AgentState:
        """保险 KYC 教练链路：业务记忆分析 → 写入 → 策略生成。

        下面每一步的"模拟"注释用同一场景演示真实产出：
        input="这个客户是企业主，40岁，两个孩子，想给孩子存教育金"（首轮、信息不足）。
        """
        # 读取业务记忆：客户/从业者事实、active case、已问 KYC 焦点。
        #   产出 state.memory_context.business：
        #     {
        #       "opportunity_case_id": "case_6b81…",   // 无 active case 时自动新建
        #       "asked_focuses": [],
        #       "recall_decision": {
        #         "should_recall": true,
        #         "recall_layers": [ "customer_profile", "advisor_profile",
        #                            "case_state", "memory_event", "domain_fact" ]
        #       }
        #     }
        state = nodes.load_business_memory(state, self.business_store)
        # 产出 KYC 分析字段并判定 information_status（含 4 轮补问上限规则）。
        #   产出：
        #     information_status       = "insufficient"
        #     missing_fields           = [ "financial_preference",
        #                                  "available_long_term_funds", "family_decision_maker" ]
        #     kyc_question_round_count = 0
        state = nodes.analyze_kyc_and_route(state)
        # 把本轮明确事实、事件、问题整理成写入提案。
        #   产出 state.memory_write_proposal：
        #     {
        #       "facts_to_upsert": [
        #         { "fact_key": "occupation", "fact_value": "企业主",
        #           "certainty": "confirmed", "confidence": 0.9 }, …
        #       ],
        #       "events_to_insert": [ … ], "questions_to_record": [ … ]
        #     }
        state = nodes.propose_memory_writes(state)
        # 校验写入提案（证据、PII、误写）。
        #   产出 state.memory_write_validation：
        #     {
        #       "is_valid": true,
        #       "allowed_fact_ids": [ "customer_fact_cdb7…", … ],  // 有证据、非 PII、非生成建议误写
        #       "blocked_fact_ids": [], "errors": []
        #     }
        state = nodes.validate_memory_writes(state)
        # 持久化通过校验的业务记忆快照。
        #   写入业务 store：
        #     CustomerProfileFact / MemoryEvent / KYCQuestion / AgentSessionState / AnalysisRun 落库，
        #     并更新 OpportunityCase(case_id = "case_6b81…")。
        state = nodes.persist_memory_snapshot(state, self.business_store)
        # 按 information_status 路由到补问 / 模式检索 / 直接策略。
        #   产出：
        #     current_state = "GENERATE_KYC_QUESTIONS"   // insufficient 且 round<4 → 进入分支 A
        state = nodes.status_router(state)

        # 分支 A：信息不足 → 生成一条低压补问。
        #   产出：
        #     answer        = "他平时更偏好哪类资金安排，比如银行理财、定存、基金或企业周转？"
        #     asked_focuses = [ "financial_preference" ]   // 每轮只问一个未问过的 focus
        if state.current_state == AgentNode.GENERATE_KYC_QUESTIONS:
            state = nodes.generate_kyc_questions(state)
        # 分支 B：信息充分（matched）→ 检索对话模式 + 外部素材，再构建 compact_context 生成策略。
        #   产出：
        #     retrieved_dialogue_patterns = [ 已审核且非高风险的对话模式 ]
        #     metadata.news_digest        = { … }            // 本地不联网，仅保留接口
        #     compact_context             = { confirmed / uncertain / case / patterns / news }
        #     answer                      = 基于 compact_context 的策略话术
        elif state.current_state == AgentNode.RETRIEVE_DIALOGUE_PATTERNS:
            state = nodes.retrieve_dialogue_patterns_node(state)
            state = nodes.retrieve_external_context_if_needed_node(state)
            # compact_context 只在这里构建一次（生成策略前），不重复构建。
            state = nodes.build_compact_context_node(state, self.business_store)
            state = nodes.generate_strategy_node(state)
        # 分支 C：信息过少（unmatched）→ 构建 compact_context 后直接生成低压维护策略。
        #   产出：
        #     compact_context = { confirmed / uncertain / case }
        #     answer          = 低压维护建议
        else:
            state = nodes.build_compact_context_node(state, self.business_store)
            state = nodes.generate_strategy_node(state)

        # 输出前合规审查（与通用链路共用同一节点）。
        #   产出：合规通过 → current_state = "RESPONSE_PACKAGING"
        state = nodes.compliance_review(state)
        # 命中高风险时停在 HUMAN_APPROVAL。
        #   本例合规通过，不进入此分支。
        if state.current_state == AgentNode.HUMAN_APPROVAL:
            return state

        # 封装响应。
        #   产出 state.response_package：
        #     { "answer": "…", "citations": [ … ], "tool_cards": [ … ],
        #       "next_actions": [ … ], "risk_level": "…", "trace_id": "…" }
        state = nodes.response_packaging(state)
        # 记录 GeneratedOutput，形成"策略→结果"审计闭环。
        #   写入：GeneratedOutput 落库，并记录 used_case_pattern_ids。
        state = nodes.post_response_logger_node(state, self.business_store)
        # trace 收尾，推进到 FINAL。
        #   产出：final_state = "FINAL"
        state = nodes.trace_finalize(state)
        return state


def build_agent_graph(
    memory_manager: MemoryManager | None = None,
    business_store: BusinessMemoryStore | None = None,
) -> AgentGraph:
    """构建 Agent 执行器。保留该函数名，调用方（WorkflowEngine）无需改动。"""
    # 直接返回线性执行器；不再有 LocalGraph / LangGraph 双图与 topology 声明层。
    return AgentGraph(memory_manager, business_store)
