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
- 代码：`src/agent_core/workflow/steps.py`
- 文档：`docs/workflow.md`
- 测试：`tests/test_production_contracts.py`

说明：`WorkflowStepContract` 明确每个 step 的输入、输出、允许下一状态、guardrails、工具权限、重试策略和 trace 字段。

## 3. Tool Schema 和权限等级

- 代码：`src/agent_core/tools/schemas.py`
- 代码：`src/agent_core/tools/registry.py`
- 代码：`src/agent_core/tools/permissions.py`
- 代码：`src/agent_core/guardrails/tool_guardrails.py`
- 文档：`docs/tool-system.md`
- 测试：`tests/test_tools_and_cost.py`

说明：`ToolSpec` 包含 input schema、output schema、risk level、permission level、scope、side effect、approval、retry、timeout、error schema。

## 4. RAG Query Rewrite / Hybrid Search / Metadata / Rerank

- 代码：`src/agent_core/rag/query_rewrite.py`
- 代码：`src/agent_core/rag/schemas.py`
- 代码：`src/agent_core/rag/retriever.py`
- 代码：`src/agent_core/rag/reranker.py`
- 代码：`src/agent_core/rag/vector.py`
- 代码：`src/agent_core/sales_intelligence/retriever.py`
- 文档：`docs/rag.md`
- 测试：`tests/test_rag_hybrid_memory_approval.py`

说明：当前实现为本地 deterministic hybrid retriever，融合 lexical score、local vector-like score、metadata score，并支持 library、tag、risk、approved filter。生产扩展路径是替换 `HybridRetriever` 内部实现为 Elasticsearch / OpenSearch + Vector DB + reranker model。

## 5. Memory 分层

- 代码：`src/agent_core/memory/session.py`
- 代码：`src/agent_core/memory/task.py`
- 代码：`src/agent_core/memory/preference.py`
- 代码：`src/agent_core/memory/manager.py`
- 代码：`src/agent_core/memory/policy.py`
- 文档：`docs/memory.md`
- 测试：`tests/test_rag_hybrid_memory_approval.py`

说明：`MemoryManager` 统一访问 session/task/preference 三层 memory，并记录 audit log。生产扩展路径是将底层 map 替换为 Redis/Postgres/tenant-isolated storage。

## 6. Context Builder

- 代码：`src/agent_core/context/builder.py`
- 代码：`src/agent_core/context/source_boundary.py`
- 代码：`src/agent_core/context/compression.py`
- 文档：`docs/context-engineering.md`
- 测试：`tests/test_workflow_engine.py`

说明：Context Builder 将检索证据压缩为 digest，并注入 source boundary rules，避免外部资料变成系统指令。

## 7. Prompt Injection 防护

- 代码：`src/agent_core/guardrails/prompt_injection.py`
- 代码：`src/agent_core/guardrails/input.py`
- 代码：`src/agent_core/graph/nodes.py`
- 文档：`docs/guardrails.md`
- 测试：`tests/test_trace_and_security.py`

说明：输入进入 `CLASSIFY_INTENT` 时先执行 Input Guardrail，命中越权/注入模式会直接进入 `ERROR`。

## 8. Human Approval 机制

- 代码：`src/agent_core/guardrails/human_approval.py`
- 代码：`src/agent_core/graph/nodes.py`
- 文档：`docs/human-in-the-loop.md`
- 测试：`tests/test_rag_hybrid_memory_approval.py`

说明：当前实现包含审批请求、审批决定和内存审批队列。生产扩展路径是接入审批数据库、运营后台和通知系统。

## 9. Structured Trace Log

- 代码：`src/agent_core/graph/state.py`
- 代码：`src/agent_core/observability/logger.py`
- 代码：`src/agent_core/workflow/engine.py`
- 代码：`src/agent_core/observability/langsmith_client.py`
- 文档：`docs/observability.md`
- 文档：`docs/langsmith-integration.md`
- 测试：`tests/test_trace_and_security.py`

说明：每个状态迁移写入 `state_transitions` 和 `trace_events`，`WorkflowEngine` 将其写入本地结构化日志。LangSmith 是可选增强层。

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

说明：当前包含 retry helper、fallback answer、`RecoveryPlan`。生产扩展路径是将 `RecoveryPlan` 接入每个 `AgentGraph` 节点的异常处理路径和 checkpoint resume。

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

说明：Dify 是 Control Plane，通过 HTTP 节点调用 Agent Gateway。

## 14. 面试讲解文档

- 文档：`docs/interview-guide.md`
- 文档：`docs/project-structure.md`
- 文档：`docs/architecture.md`
- 文档：`docs/sales-intelligence-layer.md`

说明：面试讲解重点是 Control Plane / Data Plane、显式状态机、Sales Intelligence 不是普通 RAG、可观测性和评估闭环。

## 15. 注释、测试、日志

- 注释：所有 Python 模块有 module docstring，关键类/函数有 docstring。
- 日志：`StructuredLogger` 输出 JSON 日志，状态迁移和 trace event 会被写入本地日志。
- 测试：`tests/` 覆盖 schema、pipeline、retrieval、guardrails、tools、cost、workflow、trace、安全、审批和 memory。
- 文档：`docs/logging-and-comments-policy.md`

当前限制：

- 外部模型、真实搜索、真实向量库、真实 LangSmith 远程写入都需要密钥、网络和服务配置。
- 当前替代方案是 adapter + mock/local deterministic implementation。
- 接口预留已经在 capabilities、rag、observability、integrations 中完成。
- 后续扩展时替换 adapter 内部实现，不需要重写 Agent Core 边界。

