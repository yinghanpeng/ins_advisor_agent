# 生产级 Agent 架构检查表对照

本文档逐项对照当前实现，说明每个生产级能力落在哪些代码、文档和测试中。原则是：能实现的直接实现；需要外部服务的能力提供 adapter、接口预留、替代方案和扩展路径。

## 1. 显式状态机

- 代码：`src/agent_core/graph/state.py`
- 代码：`src/agent_core/graph/nodes.py`
- 代码：`src/agent_core/graph/builder.py`
- 文档：`docs/state-machine.md`
- 测试：`tests/test_workflow_engine.py`、`tests/test_trace_and_security.py`

说明：`AgentNode` 定义完整状态枚举，`AgentState.move_to()` 记录每次状态迁移，`WorkflowEngine` 将迁移写入结构化日志。

## 2. Workflow Step Contract

- 代码：`src/agent_core/workflow/contracts.py`
- 代码：`src/agent_core/graph/builder.py`
- 代码：`src/agent_core/graph/nodes.py`
- 文档：`docs/workflow.md`
- 测试：`tests/test_production_contracts.py`

说明：`WorkflowStepContract` 保留为节点契约模型；原 `workflow/steps.py` 已删除，实际执行顺序只由 `AgentGraph` 和节点函数定义，避免声明配置与运行代码出现两套真相。

## 3. Tool Schema 和权限等级

- 代码：`src/agent_core/tools/schemas.py`
- 代码：`src/agent_core/tools/registry.py`
- 代码：`src/agent_core/tools/permissions.py`
- 代码：`src/agent_core/guardrails/tool_guardrails.py`
- 文档：`docs/tool-system.md`
- 测试：`tests/test_tools_and_cost.py`

说明：`ToolSpec` 包含 input schema、output schema、risk level、permission level、scope、side effect、retry、timeout、error schema。客户渠道不支持待审批动作：只读且获准的工具可以执行，写入、外部动作、金融动作或越权工具同步拒绝。

## 4. RAG Query Rewrite / Hybrid Search / Metadata / Rerank

- 代码：`src/agent_core/rag/query_rewrite.py`
- 代码：`src/agent_core/rag/schemas.py`
- 代码：`src/agent_core/rag/retriever.py`
- 代码：`src/agent_core/rag/reranker.py`
- 代码：`src/agent_core/rag/vector.py`
- 代码：`src/agent_core/sales_intelligence/retriever.py`
- 文档：`docs/rag.md`
- 测试：`tests/test_rag_hybrid_memory.py`

说明：当前实现为本地 deterministic hybrid retriever，融合 lexical score、local vector-like score、metadata score，并支持 library、tag、risk、approved filter。生产扩展路径是替换 `HybridRetriever` 内部实现为 Elasticsearch / OpenSearch + Vector DB + reranker model。

## 5. Memory 分层

- 代码：`src/agent_core/memory/session.py`
- 代码：`src/agent_core/memory/preference.py`
- 代码：`src/agent_core/memory/manager.py`
- 代码：`src/agent_core/memory/policy.py`
- 文档：`docs/memory.md`
- 测试：`tests/test_rag_hybrid_memory.py`

说明：本地测试继续使用 `MemoryManager`；FastAPI lifespan 已注入 `ProductionMemoryManager`：Session
使用 Redis TTL、LRU 容量限制和 CAS，Preference 使用 PostgreSQL 真 Upsert 和独立 Embedding 表，本轮消息
追加到 `short_term_messages` 加密审计，业务记忆使用 `PostgresBusinessMemoryStore`。同步链路不保存独立任务层
或 Checkpoint；保险工作状态与已问焦点分别进入 `agent_session_states`、`kyc_questions`。数据库启用
RLS、复合租户外键、原文/证据字段加密、Consent、
Retention、用户导出/删除和 Proposal Unit of Work。规范化 KYC 与 Session/Analysis JSONB 当前依靠
RLS + Consent，不属于应用层密文，详见 `docs/memory-system.md` 的数据边界。

## 6. Context Builder

- 代码：`src/agent_core/context/builder.py`
- 代码：`src/agent_core/context/source_boundary.py`
- 代码：`src/agent_core/context/compression.py`
- 文档：`docs/context-engineering.md`
- 测试：`tests/test_workflow_engine.py`

说明：Context Builder 将检索证据压缩为 digest，并注入 source boundary rules，避免外部资料变成系统指令。

## 7. Prompt Injection 防护

- 代码：`src/agent_core/guardrails/prompt_injection.py`
- 代码：`src/agent_core/guardrails/insurance_input.py`
- 代码：`src/agent_core/guardrails/input.py`
- 代码：`src/agent_core/graph/nodes.py`
- 文档：`docs/guardrails.md`
- 测试：`tests/test_trace_and_security.py`
- 测试：`tests/test_input_guardrail_hardening.py`

说明：输入进入 `CLASSIFY_INTENT` 时先执行 Input Guardrail。规则层先做 HTML/URL/Unicode/零宽字符归一化，
再扫描确定性动作短语、软可疑短语、伪造指令结构、Base64/Hex 编码和 Typoglycemia 变体；弱信号按分值
聚合后才进入 LLM Judge。保险业务违规和代操作请求使用独立类别，避免与 Prompt Injection 混淆。

## 8. 高风险同步阻断与降级

- 代码：`src/agent_core/guardrails/input.py`
- 代码：`src/agent_core/guardrails/tool_guardrails.py`
- 代码：`src/agent_core/graph/nodes.py`
- 测试：`tests/test_input_guardrail_hardening.py`

说明：客户系统不会创建待审批任务。输入高风险代操作请求返回安全替代说明，非只读工具直接 deny，高风险输出当场改写。

## 9. Structured Trace Log

- 代码：`src/agent_core/graph/state.py`
- 代码：`src/agent_core/observability/logger.py`
- 代码：`src/agent_core/workflow/engine.py`
- 代码：`src/agent_core/observability/langsmith_client.py`
- 文档：`docs/observability.md`
- 文档：`docs/langsmith-integration.md`
- 测试：`tests/test_trace_and_security.py`

说明：每个状态迁移写入 `state_transitions` 和 `trace_events`，`WorkflowEngine` 将其写入本地结构化日志；启用
LangSmith 后还会创建一个根 Run 和动态节点子 Run。可选择控制面或完整业务内容，完整模式包含状态前后快照、
模型、工具、RAG、Prompt 和回答；认证凭据始终递归清除。SDK 异步批量发送，关闭时有限等待 flush，远端失败
不会阻断客户请求。上线前仍需确定客户授权、Workspace 权限、采样率、保留周期和告警策略。

## 10. Eval Dataset

- 数据：`evals/dataset.jsonl`
- 代码：`evals/run_evals.py`
- 代码：`src/agent_core/evals/evaluators.py`
- 文档：`docs/evaluation.md`
- 测试：`tests/test_integrations_and_evals.py`

说明：当前 dataset 覆盖通用任务、模糊输入、工具失败、prompt injection、成本压力、多轮状态、Dify、LangSmith、销售破冰、KYC、异议、案例、计划书等场景。

## 11. Retry / Recovery

- 代码：`src/agent_core/recovery/retry.py`
- 代码：`src/agent_core/recovery/fallback.py`
- 文档：`docs/retry-recovery.md`

说明：当前包含 retry helper、fallback answer、`RecoveryPlan`，用于单次请求内的失败收敛。同步客户链路不保存
持久化任务检查点，也不支持 checkpoint resume；若未来引入长任务，应作为独立异步产品边界重新设计。

## 12. Cost Budget

- 代码：`src/agent_core/cost/budget.py`
- 代码：`src/agent_core/cost/model_router.py`
- 配置：`configs/cost_budget.yaml`
- 文档：`docs/cost-control.md`
- 测试：`tests/test_tools_and_cost.py`

说明：`CostBudget` 能判断是否可消费 token，并返回结构化 `CostDecision`。生产扩展路径是接入模型 token usage 回调和租户级预算。

## 13. Dify 集成文档

- 文档：`docs/dify-integration.md`
- 配置：`dify/workflow.yml`
- 节点说明：`dify/nodes/*.md`
- 代码：`src/agent_core/integrations/dify_webhook.py`

说明：Dify 是可选调用端，通过 HTTP 节点调用 Agent Gateway。保险意图、KYC、双知识库和策略均在 Python 中执行，Dify Workflow 不能选择或绕过保险路由。

## 14. 面试讲解文档

- 文档：`docs/interview-guide.md`
- 文档：`docs/project-structure.md`
- 文档：`docs/architecture.md`
- 文档：`docs/sales-intelligence-layer.md`

说明：面试讲解重点是 Control Plane / Data Plane、显式状态机、Sales Intelligence 不是普通 RAG、可观测性和评估闭环。

## 15. 注释、测试、日志

- 注释：所有 Python 模块有 module docstring，关键类/函数有 docstring。
- 日志：`StructuredLogger` 输出 JSON 日志，状态迁移和 trace event 会被写入本地日志。
- 测试：`tests/` 覆盖 schema、pipeline、retrieval、guardrails、tools、cost、workflow、trace、安全、同步阻断和 memory。
- 文档：`docs/logging-and-comments-policy.md`

当前限制：

- 外部模型、真实搜索、真实向量库和 LangSmith 远程写入都需要各自的密钥、网络和服务配置。
- LangSmith 运行时 Run Tree 已实现；Dataset/Experiment 自动执行仍是评估扩展项。
- 接口预留已经在 capabilities、rag、observability、integrations 中完成。
- 后续扩展时替换 adapter 内部实现，不需要重写 Agent Core 边界。

## 16. 单轮工具链与 Agentic 实验能力

- 代码：`src/agent_core/agentic_loop/schemas.py`
- 代码：`src/agent_core/agentic_loop/planner.py`
- 代码：`src/agent_core/graph/nodes.py`
- 文档：`docs/agentic-tool-loop.md`
- 测试：`tests/test_agentic_tool_loop.py`

说明：通用主链路当前使用单次 `general_tool_routing → general_tool_call → verify_tool_result`。
`agentic_tool_loop`、预算和重复计划检测代码只作为实验能力保留，不由 `_run_universal` 调用。
单轮链路仍执行 ToolGuardrail、同步权限拒绝、结果清洗、`_source_boundary` 和结果校验；不存在 Human Approval 或挂起分支。

## 17. Clarify 短路分支

- 代码：`src/agent_core/graph/builder.py`
- 代码：`src/agent_core/graph/nodes.py`
- 文档：`docs/clarify-and-interrupt.md`
- 测试：`tests/test_clarify_branch.py`

说明：通用工具选定后，`general_tool_routing` 使用该工具的 `input_schema` 校验参数；缺参会写入 `context_needs["clarify"]`，并在执行器和生成前调用 `generate_clarification_response → response_packaging → trace_finalize → return`。KYC 缺失信息则继续由 `missing_fields` 和专用补问节点管理。

## 18. Evaluator-Optimizer 有界闭环

- 代码：`src/agent_core/graph/nodes.py`
- 文档：`docs/evaluator-optimizer.md`
- 测试：`tests/test_response_evaluator_optimizer.py`

说明：回答生成后增加 `output_pii_scan → evaluate_response_quality → regenerate_response_if_needed`。重生成默认最多一次，复用同一 `compressed_context` 和 `tool_results`，不重新调用外部工具。重生成后再次执行 PII、grounding 和 compliance。

## 19. Streaming 事件骨架

- 代码：`src/agent_core/graph/state.py`
- 代码：`src/agent_core/graph/nodes.py`
- 代码：`src/agent_core/api/routes.py`
- 文档：`docs/streaming-events.md`
- 测试：`tests/test_stream_events.py`

说明：新增 `stream_events` 和 `streaming_enabled`。当前版本不做 token streaming，但会记录节点、工具、工具循环和最终答案事件，API 层提供 `/agent/stream` adapter-ready 入口，后续可接 SSE。

## 20. 输出侧 PII 二次扫描

- 代码：`src/agent_core/guardrails/output_pii.py`
- 代码：`src/agent_core/graph/nodes.py`
- 测试：`tests/test_output_pii_scan.py`

说明：输入 PII 扫描保护的是用户输入进入记忆、工具和模型之前的边界；输出 PII 扫描保护的是最终答案返回前的边界。它会脱敏手机号、身份证、邮箱、微信、银行卡和精确地址，只在 trace 中记录 PII 类型和位置摘要，不保存原始敏感文本。

## 21. 双层意图路由与代码化保险处理器

- 代码：`src/agent_core/intents/knowledge_base.py`
- 代码：`src/agent_core/intents/router.py`
- 代码：`src/agent_core/skills/insurance_advisor/kyc.py`
- 代码：`src/agent_core/skills/insurance_advisor/knowledge.py`
- 配置：`configs/intent_routing.yaml`、`configs/intent_catalog.yaml`、`configs/insurance_handler.yaml`
- 文档：`docs/intent-routing-and-insurance-handler.md`
- 测试：`tests/test_intent_routing.py`、`tests/test_kyc_workflow_contract.py`

说明：向量相似度按 `0.85/0.60` 分层，LLM 裁定后按 `0.80/0.60` 分发；Redis active intent 优先处理续接和换题。保险 KYC 只抽本轮明确增量，Python 负责合并、缺口、轮次和路由；方法库与合同合规库独立检索。所有阈值、TopK、TTL、模型和 Provider 都来自配置文件或环境变量。

## 22. 专业 Agent Registry 与计划书占位边界

- 代码：`src/agent_core/agents/contracts.py`
- 代码：`src/agent_core/agents/registry.py`
- 代码：`src/agent_core/agents/advisor_coach/agent.py`
- 代码：`src/agent_core/agents/insurance_proposal/`
- 测试：`tests/test_domain_agent_registry.py`

说明：总控在意图裁定后按 `intent + domain_skill` 精确选择一个已启用专业 Agent；两个已启用 Agent
重复声明同一条路由时，Registry 在装配阶段直接失败。现有保险 Handler 已迁入 `AdvisorCoachAgent`，
`AgentGraph` 只保留兼容薄代理。计划书 Agent 当前为 `enabled=False` 的占位实现；其进程健康检查返回
`True`，但业务结果固定为 `available=False/status=not_configured`，不会伪造计划书或影响现有请求。
