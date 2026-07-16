# 项目结构与职责

## 顶层

| 路径 | 职责 |
| --- | --- |
| `main.py` | CLI 单轮和交互入口 |
| `README.md` | 项目总说明、架构、配置和运行方式 |
| `configs/` | 模型、意图、保险双知识库、Memory、工具和安全配置 |
| `src/agent_core/` | Agent 运行代码 |
| `tests/` | 阈值、链路、Memory、工具、安全、RAG 和文档测试 |
| `migrations/` | PostgreSQL、pgvector、业务记忆、RLS 和遗留运行时表清理迁移 |
| `dify/` | 可选 HTTP Adapter 和历史节点说明，不承载保险运行逻辑 |
| `docs/` | 架构与专题文档 |

## 关键配置

| 文件 | 运行语义 |
| --- | --- |
| `intent_routing.yaml` | 相似度/置信度阈值、TopK、active intent TTL、KYC 轮次和证据阈值 |
| `api.yaml` | API Key、固定窗口限流、请求体、CORS 和生产后端要求 |
| `intent_catalog.yaml` | 可执行意图白名单、标准表达、route、domain 和 KYC 标记 |
| `insurance_handler.yaml` | 方法库、合同合规库、TopK、阈值和新闻开关 |
| `models.yaml` | Chat、意图裁定、漂移检测、KYC 抽取、Embedding、Reranker |
| `database.yaml` | PostgreSQL/Redis 连接池、健康检查、Socket 超时和重试 |
| `memory.yaml` | Session/Preference TTL、Redis 容量、加密和业务保留策略 |
| `retrieval.yaml` | 通用 Hybrid RAG 权重和阈值 |
| `tools.yaml` | 工具元数据说明；运行时白名单仍由 ToolRegistry 控制 |
| `workflow.yaml` | 仅保留通用运行标签兼容声明，保险不读取 |

## graph

| 文件 | 职责 |
| --- | --- |
| `graph/state.py` | `AgentNode`、`AgentState`、状态转移和 Trace |
| `graph/builder.py` | 总控代码顺序；公共安全、意图、Registry 路由与通用路径 |
| `graph/nodes.py` | Guardrail、Memory、意图、工具、KYC、知识、生成和输出节点函数 |

旧 `graph/intent_classifier.py` 已删除。意图识别不再是“模型优先 + 关键词兜底”，而是独立 Intent Layer。

## agents

| 文件 | 职责 |
| --- | --- |
| `agents/contracts.py` | `AgentDescriptor` 与 `DomainAgent` 最小运行协议 |
| `agents/registry.py` | 按 `intent + domain_skill` 精确选择已启用专业 Agent |
| `agents/bootstrap.py` | 使用 Runtime 共享依赖装配默认 Agent 集合 |
| `agents/advisor_coach/agent.py` | 原保险 Handler 的唯一可执行业务顺序 |
| `agents/insurance_proposal/port.py` | 真实计划书 Agent 必须实现的 `submit()` 端口 |
| `agents/insurance_proposal/schemas.py` | 计划书任务、客户快照、结果和 Artifact 契约 |
| `agents/insurance_proposal/placeholder.py` | 默认禁用占位实现；健康检查为真但业务不可用 |

`AgentGraph._run_insurance_conversation()` 只保留兼容薄代理。在线保险请求由
`DomainAgentRegistry.resolve()` 进入 `AdvisorCoachAgent`，计划书占位不会被选中。

## intents

| 文件 | 职责 |
| --- | --- |
| `intents/schemas.py` | 意图目录、向量命中、LLM 裁定、漂移和 active-intent 契约 |
| `intents/knowledge_base.py` | 本地 n-gram 向量与生产 pgvector 适配器 |
| `intents/router.py` | 0.85/0.60 相似度分层、0.80/0.60 执行度、活跃意图处理 |

## skills/insurance_advisor

| 文件 | 职责 |
| --- | --- |
| `kyc.py` | 保险领域 Pydantic 槽位、LLM 抽取、规则降级、合并、缺口、评分和追问 |
| `knowledge.py` | 沟通方法库和合同合规库 Provider、Query 与结果契约 |
| `skill.yaml` | 保险细分意图到 `AdvisorCoachAgent` 的声明性映射 |
| `prompts/strategy_generator.md` | 生成内容边界，不能拥有路由和状态写权限 |

旧 `skills/insurance_advisor/workflow.py` 已删除。KYC、知识 Provider 和 Prompt 仍是领域资产，实际步骤顺序
由 `agents/advisor_coach/agent.py` 维护。

## Memory

| 模块 | 职责 |
| --- | --- |
| `memory/manager.py` | Session、Preference 统一接口和内存实现 |
| `memory/redis_store.py` | Redis Session CAS、TTL、消息窗口和租户 LRU 容量 |
| `memory/production_manager.py` | Redis Session + PostgreSQL 偏好/短期消息审计组合 |
| `memory/business_schemas.py` | 客户/顾问事实、Case、Question、Analysis、Output |
| `memory/business_store.py` | 业务记忆协议和内存实现 |
| `memory/postgres_business_store.py` | 证据/原文加密、事实版本化、Consent 与事务化生产实现 |
| `memory/compact_context.py` | confirmed/uncertain、双知识库和新闻的安全生成上下文 |

Redis active intent 只保存控制信封，客户 KYC 值进入业务记忆。同步客户请求不保存独立任务层，也不提供
Checkpoint 断点恢复；保险工作快照和已问焦点分别由 `agent_session_states`、`kyc_questions` 承担。

## Tool 与 RAG

| 模块 | 职责 |
| --- | --- |
| `tools/schemas.py` | ToolSpec、ToolCall、ToolResult |
| `tools/registry.py` | 工具白名单和 Schema |
| `tools/executor.py` | 权限、Schema、超时、重试、Sanitizer、Verifier |
| `rag/production.py` | 文档入库、Embedding、pgvector 和 Reranker |
| `persistence/postgres.py` | 租户隔离 Repository 和向量查询 |
| `sales_intelligence/` | 访谈脱敏、结构化卡片、模式和评估资产 |

## Guardrails 与可观测性

- `guardrails/input.py`：硬规则、灰区 Judge、PolicyCombiner 和 PII；
- `guardrails/metadata.py`：公开 metadata 的知识正文与业务记录 ID 信任边界；
- `guardrails/tool_guardrails.py`：权限和副作用同步拒绝；
- `guardrails/output.py` / `output_pii.py`：输出合规和隐私；
- `observability/`：结构化日志、Trace、Metrics 和 LangSmith Adapter。

客户渠道没有人工审批。所有高风险动作在当前请求内收敛到 allow、mask、safe fallback 或 block。

## 生产 Runtime

`api/runtime.py` 在 FastAPI lifespan 中创建共享 Redis、PostgreSQL、Memory Manager、Business Store、
Intent Router、KYC Extractor，并按 provider 配置装配本地或 pgvector 意图/保险知识 Provider。连接信息
和模型名全部来自配置/环境变量；`APP_ENV=staging/stage/prod/production` 时强制两个 provider 使用
pgvector，并要求意图裁定、漂移检测和 KYC 抽取三类模型完整，否则启动阶段 fail-fast。

## 重点测试

| 文件 | 覆盖 |
| --- | --- |
| `test_intent_routing.py` | 相似度/置信度边界、active intent、漂移和短回答 |
| `test_workflow_engine.py` | 统一入口、通用工具和保险自动路由 |
| `test_domain_agent_registry.py` | Registry、保险顾问迁移和计划书占位不影响现有运行 |
| `test_kyc_workflow_contract.py` | 文件名保留兼容；实际测试代码化 KYC 和记忆 |
| `test_tool_input_schema.py` | Tool Schema 缺参和执行前澄清 |
| `test_input_guardrail_hardening.py` | 注入、编码、保险风险和误报 |
| `test_production_memory_runtime.py` | Redis/PostgreSQL 生产 Memory 边界 |
| `test_generation_metadata_trust_boundary.py` | 生成上下文与业务身份 metadata 防注入 |
| `test_documentation_quality.py` | Pydantic 描述、注释和状态约束 |

完整请求顺序见 [request-lifecycle-flowchart.md](request-lifecycle-flowchart.md)。
