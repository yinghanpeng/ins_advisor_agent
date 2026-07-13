"""Agent 执行图（线性顺序写法）。

# 文件说明：
# - 本文件是 Agent 主链路的唯一编排实现，采用"从上往下一步一步"的线性写法，方便直接顺着读懂流程。
# - 一次请求的入口是 WorkflowEngine.run() → self.graph.invoke(state)，self.graph 就是这里的 AgentGraph。
# - 所有请求都走同一个代码入口：先恢复记忆并完成双层意图识别；
# - 通用意图继续走 Tool/RAG/生成链，保险意图自动进入代码化 Insurance Conversation Handler；
# - 外部 workflow_name 不再决定保险路由，附件 Dify Workflow 仅作为迁移来源和离线参考。
"""

from __future__ import annotations

import os
from typing import Any

from agent_core.config.runtime import load_runtime_settings
from agent_core.graph import nodes
from agent_core.graph.state import AgentNode, AgentState
from agent_core.intents.router import INSURANCE_INTENTS, IntentRouter, build_intent_router
from agent_core.memory.business_store import BusinessMemoryStore
from agent_core.memory.manager import MemoryBackend
from agent_core.skills.insurance_advisor.kyc import InsuranceKycExtractor
from agent_core.skills.insurance_advisor.knowledge import (
    InsuranceKnowledgeProvider,
    LocalInsuranceKnowledgeProvider,
)


class AgentGraph:
    """Agent 主链路执行器：按固定顺序一步步调用各节点函数。"""

    def __init__(
        self,
        memory_manager: MemoryBackend | None = None,
        business_store: BusinessMemoryStore | None = None,
        intent_router: IntentRouter | None = None,
        kyc_extractor: InsuranceKycExtractor | None = None,
        insurance_knowledge_provider: InsuranceKnowledgeProvider | None = None,
        insurance_news_enabled: bool | None = None,
    ) -> None:
        """注入记忆、业务事实、双层意图路由和保险 KYC 抽取依赖。"""
        # 短期/任务/偏好记忆管理器；同一个 WorkflowEngine 实例内多轮对话复用同一份。
        self.memory_manager = memory_manager
        # KYC 业务记忆 store；保存客户事实、机会 case、生成输出等。
        self.business_store = business_store
        # Router 在 Engine 生命周期内复用意图目录和模型客户端，避免每个请求重复加载配置。
        self.intent_router = intent_router or build_intent_router()
        # KYC Extractor 同样复用低延迟模型客户端；本地未配置模型时自动走规则降级。
        self.kyc_extractor = kyc_extractor or InsuranceKycExtractor()
        # 双知识库 Provider 在本地为空实现，生产由 Runtime 注入 pgvector 实现。
        self.insurance_knowledge_provider = (
            insurance_knowledge_provider or LocalInsuranceKnowledgeProvider()
        )
        # 新闻工具是否可用由 insurance_handler.yaml 注入；即使开启也只在代码判断需要时调用。
        self.insurance_news_enabled = (
            load_runtime_settings(os.getenv("CONFIG_DIR", "configs")).insurance_knowledge.news_enabled
            if insurance_news_enabled is None
            else insurance_news_enabled
        )

    def invoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        """执行一次完整请求：公共安全入口 → 统一意图路由 → 通用或保险代码路径。"""
        # 调用方可能传 dict，这里统一恢复成 AgentState，后续节点只处理一种类型。
        if isinstance(state, dict):
            # 用 Pydantic 契约把字典校验并恢复为 AgentState，拒绝结构非法的运行状态。
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
        #       "decision_action": "allow",    // 精细动作：allow / mask / safe_fallback / block
        #       "risk_level": "low",
        #       "masked": false,               // 命中 PII 时为 true，并回填 sanitized_text
        #       "signals": []                  // 命中的证据链（硬闸 / PII / LLM Judge）
        #     }
        #   → block 进入 ERROR；safe_fallback 同步返回安全答复并进入 FINAL。
        state = nodes.input_guardrail(state)
        # 输入被阻断或已生成同步安全降级答复时，不进入任何后续节点。
        if state.final_state in {AgentNode.ERROR, AgentNode.FINAL}:
            # 返回更新后的 Agent 状态，交由主流程继续调度下一节点。
            return state

        # 第 3 步：所有请求统一恢复 Redis 会话并做意图判断；保险不再由外部 workflow_name 强制进入。
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
        #         "last_entity": "Anthropic"
        #       },
        #       "task": { "current_state": "FINAL", "final_answer_ready": true },
        #       "preference": {},                     // 长期偏好召回结果（本轮判定不召回 → 空）
        #       "long_term_recall": {                 // 召回决策全过程（可审计）
        #         "decision": { "should_recall": false, "status": "model_unavailable_safe_skip" },
        #         "items": []                         // 锚点"Anthropic"已在短期消息内 → 不触发长期召回
        #       }
        #     }
        # 执行 restore_memory 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.restore_memory(state, self.memory_manager)
        # 合并历史消息与本轮输入为标准消息结构。
        #   产出 state.normalized_messages（历史 2 条 + 追加本轮 1 条）：
        #     [
        #       { "role": "user",      "content": "帮我看看 Anthropic 这家公司" },              // 历史
        #       { "role": "assistant", "content": "Anthropic 是一家 AI 公司…" },               // 历史
        #       { "role": "user",      "content": "它最近有没有融资…", "source": "current_turn" }  // 本轮
        #     ]
        # 执行 normalize_messages 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.normalize_messages(state)
        # 双层意图识别：先读取 Redis active_intent，再走向量召回、必要的 LLM 裁定和置信度分发。
        #   产出：
        #     intent           = "web_or_news_search"
        #     capability_route = "general"        // general=通用工具层 / domain=业务 Skill
        #     domain_skill     = None
        # 执行 classify_intent 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.classify_intent(state, self.intent_router)
        # 低置信度或活跃意图变化不明确时，在工具、RAG、业务记忆读取之前主动澄清。
        if state.context_needs.get("clarify"):
            # 执行 generate_clarification_response 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.generate_clarification_response(state)
            # 执行 response_packaging 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.response_packaging(state)
            # 执行 update_short_term_memory 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.update_short_term_memory(state, self.memory_manager)
            # 执行 trace_finalize 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.trace_finalize(state)
            # 返回更新后的 Agent 状态，交由主流程继续调度下一节点。
            return state
        # 语义风险分级，供工具权限与输出策略复用。
        #   产出：
        #     risk_level = "medium"   // 命中"融资 / 英文报道"等中风险词（取值 low / medium / high）
        state = nodes.semantic_risk_classification(state)
        # 保险细分意图自动进入代码化领域处理器；不再依赖 workflow_name 或 Dify 节点编排。
        if state.intent in INSURANCE_INTENTS and state.domain_skill == "insurance_advisor":
            # 返回 _run_insurance_conversation 构造的结构化结果，供调用方继续处理。
            return self._run_insurance_conversation(state)
        # 指代消解、实体/时间解析、query rewrite 和 filters 生成。
        # 通用工具参数不再经过独立槽位层；工具选定后由该工具自己的 input_schema 校验。
        #   产出 state.query_understanding：
        #     {
        #       "entity": "Anthropic",
        #       "resolved_query": "Anthropic最近有没有融资，重点看过去三个月的英文报道",  // "它"替换回实体
        #       "rewritten_query": "Anthropic funding news in the past three months",      // 供外部检索
        #       "date_range": { "start": "2026-04-08", "end": "2026-07-09" },              // "过去三个月"→绝对区间
        #       "filters": { "language": "en", "source_type": "news", "date_range": {…} }
        #     }
        # 执行 query_understanding 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.query_understanding(state)
        # 规划本轮需要 memory/RAG/tool 中的哪些能力。
        #   产出 state.context_needs：
        #     {
        #       "memory": true, "long_term_memory": false, "rag": false,
        #       "tool": true,        // → 进入下面的分支 A（工具链）
        #       "safe_response": false, "reject": false, "clarify": false
        #     }
        # 执行 context_need_planning 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.context_need_planning(state)

        # Clarify 兼容短路：外部 planner 显式要求澄清时，在工具/RAG/生成前直接补问。
        if state.context_needs.get("clarify"):
            # 执行 generate_clarification_response 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.generate_clarification_response(state)
            # 执行 response_packaging 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.response_packaging(state)
            # 执行 trace_finalize 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.trace_finalize(state)
            # 返回更新后的 Agent 状态，交由主流程继续调度下一节点。
            return state

        # 分支 A：需要外部工具（天气/搜索/计算等）时走工具链。
        if state.context_needs.get("tool"):
            # 当前主链路使用单轮显式工具调用模式；
            # agentic_tool_loop 仅保留为未接入的实验能力，不再由 _run_universal 调用。
            # 生成工具调用计划。
            #   产出 state.tool_plan：
            #     [
            #       {
            #         "tool_name": "summarizer",
            #         "arguments": { "text": "它最近有没有融资…", "max_chars": 300 },
            #         "risk_level": "low",
            #         "permission_scope": "llm.transform",
            #         "side_effect_level": "read_only"
            #       }
            #     ]
            # 执行 general_tool_routing 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.general_tool_routing(state)
            # Tool Schema 发现必填参数缺失时，routing 会设置 clarify；必须在执行器前短路。
            if state.context_needs.get("clarify"):
                # 执行 generate_clarification_response 节点并接回更新后的 Agent 状态，保持主链路数据连续。
                state = nodes.generate_clarification_response(state)
                # 执行 response_packaging 节点并接回更新后的 Agent 状态，保持主链路数据连续。
                state = nodes.response_packaging(state)
                # 执行 trace_finalize 节点并接回更新后的 Agent 状态，保持主链路数据连续。
                state = nodes.trace_finalize(state)
                # 返回更新后的 Agent 状态，交由主流程继续调度下一节点。
                return state
            # 执行工具（含权限与副作用检查）。
            #   产出：
            #     tool_calls   = [ { "tool_name": "summarizer", "status": "success", "latency_ms": 0 } ]
            #     tool_results = [ { "name": "summarizer", "status": "success",
            #                        "output": { "summary": "…",
            #                                    "_source_boundary": { "trust": "untrusted_external_context" } } } ]
            #     // 工具结果带 source_boundary：只作事实候选，不可当指令执行
            state = nodes.general_tool_call(state)
            # 执行后澄清判断只保留给外部 planner/兼容调用；主链路缺参已在 routing 后短路。
            if state.context_needs.get("clarify"):
                # 执行 generate_clarification_response 节点并接回更新后的 Agent 状态，保持主链路数据连续。
                state = nodes.generate_clarification_response(state)
                # 执行 response_packaging 节点并接回更新后的 Agent 状态，保持主链路数据连续。
                state = nodes.response_packaging(state)
                # 执行 trace_finalize 节点并接回更新后的 Agent 状态，保持主链路数据连续。
                state = nodes.trace_finalize(state)
                # 返回更新后的 Agent 状态，交由主流程继续调度下一节点。
                return state
            # 校验工具结果，失败进入恢复但仍走保守回答。
            #   产出：
            #     errors      = []   // 无失败结果
            #     retry_count = 0    // 无需 recovery / 降级
            # 单轮模式在这里校验唯一一次 general_tool_call 的结构化结果和失败降级。
            state = nodes.verify_tool_result(state)
        # 分支 B：为未来非保险领域 Skill 保留的通用 Domain 路由。
        # 保险意图已在 Query Understanding 之前进入代码化 Handler，不会落到这里。
        elif state.capability_route == "domain":
            # 明确领域能力路由。
            #   产出：
            #     sales_route   = 领域能力内部代码路由标签
            #     current_state = "SALES_INTELLIGENCE_ROUTING"
            # 执行 route_domain_workflow 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.route_domain_workflow(state)
            # 检索已审核销售洞察卡片。
            #   产出：
            #     rewritten_queries = [ "帮我给这个企业主客户做保险破冰，怎么开口",
            #                           "话术策略 帮我给这个企业主客户做保险破冰，怎么开口" ]
            #     retrieved_context = [
            #       { "source_id": "sample_interview_001", "customer_type": "企业主",
            #         "scene": "饭局破冰", "business_stage": "new_customer", … }   // 仅已审核、非高风险卡片
            #     ]
            # 执行 retrieve_sales_intelligence 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.retrieve_sales_intelligence(state)
            # 把检索证据压缩成生成上下文。
            #   产出 state.sales_insight_digest：
            #     {
            #       "applicable_scene": "insurance_advisor",
            #       "digest": "先围绕经营现金流和家庭责任共情，再用资金分层把话题转到长期稳定安排。",
            #       "forbidden": [ "承诺收益", "避税避债", "恐吓营销", "编造案例", "贬低其他产品" ]
            #     }
            # 执行 build_context 节点并接回更新后的 Agent 状态，保持主链路数据连续。
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
        # 执行 knowledge_fusion 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.knowledge_fusion(state)
        # 按预算压缩上下文。
        #   产出：
        #     compressed_context = {…按 token 预算裁剪后的 memory / context / tool 摘要…}
        #     cost.compressed_context_chars ≈ 1468
        # 执行 compress_context 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.compress_context(state)
        # 组装最终 prompt 结构。
        #   产出 state.assembled_prompt：
        #     {
        #       "system": "你是合规、低压、证据优先的保险顾问沟通助手。",
        #       "memory": {…}, "context": {…},
        #       "user":   "它最近有没有融资，重点看过去三个月的英文报道"
        #     }
        # 执行 prompt_assembly 节点并接回更新后的 Agent 状态，保持主链路数据连续。
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
        # 执行 grounding_verification 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.grounding_verification(state)
        # 输出前合规审查。
        #   产出 guardrail_results[-1]：
        #     { "guardrail_name": "insurance_output_compliance", "action": "pass", "triggered": false }
        #   → current_state = "RESPONSE_PACKAGING"（未命中违规承诺 / 恐吓营销等）
        state = nodes.compliance_review(state)
        # 输出命中高风险时已在 compliance_review 中同步替换为安全答复。

        # 输出侧 PII 二次扫描：检查生成答案中是否包含手机号、邮箱、身份证、银行卡等。
        state = nodes.output_pii_scan(state)

        # Evaluator-optimizer 有界闭环：只在证据不足、风险较高、PII 脱敏等情况下最多重生成一次。
        state = nodes.evaluate_response_quality(state)
        # 执行 regenerate_response_if_needed 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.regenerate_response_if_needed(state)

        # 重生成后必须再次执行 PII、grounding 和 compliance，避免优化后引入新的风险。
        state = nodes.output_pii_scan(state)
        # 执行 grounding_verification 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.grounding_verification(state)
        # 执行 compliance_review 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.compliance_review(state)
        # 第二次合规审查同样只会同步放行或降级，不会挂起请求。

        # 封装前端可消费的响应包。
        #   产出 state.response_package：
        #     {
        #       "answer": "…", "citations": [], "tool_cards": [ {summarizer…} ],
        #       "next_actions": [ … ], "risk_level": "medium", "trace_id": "…"
        #     }
        # 执行 response_packaging 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.response_packaging(state)
        # 更新短期 session/task 记忆。
        #   写回 SESSION / TASK：
        #     session = { "recent_messages": [含本轮问答], "last_intent": "web_or_news_search",
        #                 "last_entity": "Anthropic" }
        #     task    = { "current_state": "…", "final_answer_ready": true }
        # 执行 update_short_term_memory 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.update_short_term_memory(state, self.memory_manager)
        # 判断并写入长期偏好记忆候选。
        #   产出：
        #     memory_write_candidates = []   // 本轮无值得跨会话长存的偏好 / 画像
        state = nodes.long_term_memory_candidate(state, self.memory_manager)
        # trace 与成本收尾，推进到 FINAL。
        #   产出：
        #     final_state = "FINAL"
        #     cost        = { "tool_call_count": 1, "output_chars": 267, "trace_event_count": 54, … }
        # 执行 trace_finalize 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.trace_finalize(state)
        # 返回更新后的 Agent 状态，交由主流程继续调度下一节点。
        return state

    def _run_insurance_conversation(self, state: AgentState) -> AgentState:
        """代码化保险对话处理器：KYC 增量 → 确定性路由 → 追问或策略。

        下面每一步的"模拟"注释用同一场景演示真实产出：
        input="这个客户是企业主，有两个孩子，家里主要配置银行理财"（首轮、信息不足）。
        """
        # 统一使用 intent_routing.yaml 中的最大补问轮次；LLM 和附件 YAML 无权修改计数器。
        state.metadata["max_kyc_question_rounds"] = self.intent_router.config.max_kyc_question_rounds
        # active intent 使用独立业务 TTL；Redis Session 可继续保留普通对话窗口。
        state.metadata["active_intent_ttl_seconds"] = self.intent_router.config.active_intent_ttl_seconds
        # 节点只读取请求级开关，不自行读取配置文件或硬编码 Provider。
        state.metadata["insurance_news_enabled"] = self.insurance_news_enabled
        # 标记代码化处理器版本，替代旧 workflow_version 作为业务输出审计字段。
        state.metadata.setdefault("insurance_handler_version", "code-native-v1")
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
        # 执行 load_business_memory 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.load_business_memory(state, self.business_store)
        # LLM 只抽取本轮明确 KYC 增量；Pydantic 校验后与业务事实合并，模型不做评分和路由。
        state = nodes.extract_insurance_kyc_slots(state, self.kyc_extractor)
        # 代码计算 KYC 字段缺口、完整度、机会分和 information_status（最大轮次由配置控制）。
        #   产出：
        #     information_status       = "insufficient"
        #     missing_fields           = [ "insurance_experience",
        #                                  "decision_authority", "available_long_term_funds" ]
        #     kyc_question_round_count = 0
        # 执行 analyze_kyc_and_route 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.analyze_kyc_and_route(state)
        # 按 information_status 路由到补问 / 模式检索 / 直接策略。
        #   产出：
        #     current_state = "GENERATE_KYC_QUESTIONS"   // insufficient 且未达到配置轮次 → 分支 A
        state = nodes.status_router(state)

        # 分支 A：信息不足 → 生成一条低压补问。
        #   产出：
        #     answer        = "方便了解一下：这位客户以前接触或配置过保险吗？"
        #     asked_focuses = [ "insurance_experience" ]   // 每轮只问一个未问过的 focus
        if state.current_state == AgentNode.GENERATE_KYC_QUESTIONS:
            # 执行 generate_kyc_questions 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.generate_kyc_questions(state)
        # 分支 B：信息充分（matched）→ 检索对话模式 + 外部素材，再构建 compact_context 生成策略。
        #   产出：
        #     retrieved_dialogue_patterns = [ 已审核且非高风险的对话模式 ]
        #     metadata.news_digest        = { … }            // 仅按需调用已配置的只读新闻 Provider
        #     compact_context             = { confirmed / uncertain / case / patterns / news }
        #     answer                      = 基于 compact_context 的策略话术
        elif state.current_state == AgentNode.RETRIEVE_DIALOGUE_PATTERNS:
            # 同一脱敏 Query 分别检索沟通方法库和合同合规库，两个结果保持独立分区。
            state = nodes.retrieve_insurance_knowledge_node(
                state,
                self.insurance_knowledge_provider,
            )
            # 执行 retrieve_dialogue_patterns_node 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.retrieve_dialogue_patterns_node(state)
            # 执行 retrieve_external_context_if_needed_node 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.retrieve_external_context_if_needed_node(state)
            # compact_context 只在这里构建一次（生成策略前），不重复构建。
            state = nodes.build_compact_context_node(state, self.business_store)
            # 执行 generate_strategy_node 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.generate_strategy_node(state)
        # 分支 C：信息过少（unmatched）→ 构建 compact_context 后直接生成低压维护策略。
        #   产出：
        #     compact_context = { confirmed / uncertain / case }
        #     answer          = 低压维护建议
        else:
            # 执行 build_compact_context_node 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.build_compact_context_node(state, self.business_store)
            # 执行 generate_strategy_node 节点并接回更新后的 Agent 状态，保持主链路数据连续。
            state = nodes.generate_strategy_node(state)

        # 回答或补问已经成功生成后，再把本轮明确事实、实际展示的问题和分析结果整理成写入提案。
        # 这样可以避免生成阶段异常时，数据库提前把“客户尚未看到的问题”记成 asked。
        state = nodes.propose_memory_writes(state)
        # Validator 检查证据、PII、误写和整包原子性；模型输出不能绕过该契约。
        state = nodes.validate_memory_writes(state)
        # 只有通过校验且具备用途级 Consent 时才持久化；缺少 Consent 会安全切换为无记忆模式。
        state = nodes.persist_memory_snapshot(state, self.business_store)

        # Grounding 先记录本轮策略使用的业务事实、案例模式和新闻来源。
        state = nodes.grounding_verification(state)
        # 输出侧 PII 扫描防止模型或模板复述手机号、身份证、银行卡和邮箱。
        state = nodes.output_pii_scan(state)
        # 输出前合规审查（与通用链路共用同一节点）。
        state = nodes.compliance_review(state)
        # 命中高风险时同步改写为安全答复，不挂起客户请求。

        # 封装响应。
        #   产出 state.response_package：
        #     { "answer": "…", "citations": [ … ], "tool_cards": [ … ],
        #       "next_actions": [ … ], "risk_level": "…", "trace_id": "…" }
        # 执行 response_packaging 节点并接回更新后的 Agent 状态，保持主链路数据连续。
        state = nodes.response_packaging(state)
        # 记录 GeneratedOutput，形成"策略→结果"审计闭环。
        #   写入：GeneratedOutput 落库，并记录 used_case_pattern_ids。
        state = nodes.post_response_logger_node(state, self.business_store)
        # information_status=insufficient 时创建/续接 Redis 活跃意图；完成、取消或转策略后清空。
        state = nodes.sync_active_intent_state(state)
        # 统一更新最近消息和 active_intent 信封，下一轮先做意图变化判断而不是重新全量分类。
        state = nodes.update_short_term_memory(state, self.memory_manager)
        # trace 收尾，推进到 FINAL。
        #   产出：final_state = "FINAL"
        state = nodes.trace_finalize(state)
        # 返回更新后的 Agent 状态，交由主流程继续调度下一节点。
        return state


def build_agent_graph(
    memory_manager: MemoryBackend | None = None,
    business_store: BusinessMemoryStore | None = None,
    intent_router: IntentRouter | None = None,
    kyc_extractor: InsuranceKycExtractor | None = None,
    insurance_knowledge_provider: InsuranceKnowledgeProvider | None = None,
    insurance_news_enabled: bool | None = None,
) -> AgentGraph:
    """构建 Agent 执行器。保留该函数名，调用方（WorkflowEngine）无需改动。"""
    # 直接返回线性执行器；不再有 LocalGraph / LangGraph 双图与 topology 声明层。
    return AgentGraph(
        memory_manager,
        business_store,
        intent_router,
        kyc_extractor,
        insurance_knowledge_provider,
        insurance_news_enabled,
    )
