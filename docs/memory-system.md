# 业务记忆系统设计

本文说明代码化保险高客沟通处理器如何管理可持久化、可审计、可评测的业务记忆。

## 核心原则

真实客户事实不是普通 Prompt 变量，必须进入结构化记忆：

- `CustomerProfileFact` 保存客户 KYC 事实。
- `AdvisorProfileFact` 保存从业者画像事实。
- `OpportunityCase` 保存一个客户机会的长期推进状态。
- `AgentSessionState` 保存每一轮工作记忆快照，不覆盖历史。
- `KYCQuestion` 保存已问焦点，避免重复追问。
- `AnalysisRun` 保存每次 KYC 分析和评分的输入输出。
- `GeneratedOutput` 保存每次生成的话术、策略、补问或维护消息。
- `MemoryEvent` 和 `CaseOutcome` 保存情节事件和结果闭环。

## 保险业务状态映射

| 业务字段 | 工程化位置 | 说明 |
| --- | --- | --- |
| `information_status` | `AgentState` / `AgentSessionState` / `AnalysisRun` | 决定补问、策略或低压维护。 |
| `subject_type` | `AgentState` / `OpportunityCase` | 区分客户、渠道或不明确对象。 |
| `target_persona` | `AgentState` / `OpportunityCase` | 内部客群标签，不直接对客户展示。 |
| `profile_state` | `AgentState` / `AgentSessionState` | 本轮客户画像快照，长期事实进入 `CustomerProfileFact`。 |
| `practitioner_state` | `AgentState` / `AgentSessionState` | 本轮从业者画像快照，长期事实进入 `AdvisorProfileFact`。 |
| `advisor_stage` | `AgentState` / `AgentSessionState` | 从业者阶段。 |
| `missing_fields` | `AgentState` / `AgentSessionState` | 驱动低压补问。 |
| `match_evidence` | `AgentState` / `AnalysisRun` | 只写明确事实证据，不写推测。 |
| `route_reason` | `AgentState` / `AnalysisRun` | 解释当前路由原因。 |
| `kyc_completeness_score` | `AgentState` / `OpportunityCase` / `AnalysisRun` | 可追溯 KYC 完整度分。 |
| `opportunity_score` | `AgentState` / `OpportunityCase` / `AnalysisRun` | 可追溯机会推进分。 |
| `external_grade` | `AgentState` / `OpportunityCase` / `AnalysisRun` | 对从业者展示的等级。 |
| `trigger_module` | `AgentState` / `OpportunityCase` | 切入模块。 |
| `current_stage` | `AgentState` / `OpportunityCase` | 当前沟通阶段。 |
| `objective_material_need` | `AgentState` / `AgentSessionState` | 是否需要外部新闻或公开素材。 |
| `support_note` | `AgentState` / `AgentSessionState` | 给从业者的鼓励摘要，不写成客户事实。 |
| `kyc_question_round_count` | `AgentState` / `AgentSessionState` | 最大轮次来自 `intent_routing.yaml`，默认 3。 |
| `asked_focuses` | `KYCQuestion` / `AgentState` | 已问焦点来自结构化问题表。 |

## 写入策略

长期事实写入必须满足：

- 必须有 `tenant_id`。
- 必须有 `source_type`。
- 必须有 `evidence_text`。
- 明确事实使用 `certainty="confirmed"`。
- 推测、转述、弱信号使用 `certainty="uncertain"`。
- 没有证据的事实不得写入长期事实表。
- 客户姓名、电话、微信、身份证、精确地址等 PII 默认不进入长期 prompt 记忆。
- 模型生成的建议只能写入 `GeneratedOutput`，不能写成客户事实。
- 冲突事实不覆盖旧事实，旧版本置为 `is_current=false` 并记录 `valid_to`。

## 长期记忆召回策略

长期记忆不是所有请求都需要召回。

系统把记忆分成两类：

- 短期会话记忆：`session`，用于多轮指代、最近消息和 active intent，可以每轮读取；
- 长期记忆：`preference`、客户画像、从业者画像、case 事件，只在当前请求确实需要时召回。

同步客户请求不保存独立任务层，也没有 Checkpoint 断点恢复。保险工作状态由业务表
`agent_session_states` 按轮追加快照，已展示的 KYC 焦点由 `kyc_questions` 去重记录。

本地实现位于 `src/agent_core/memory/recall.py`，流程如下：

1. `plan_long_term_memory_recall` 判断是否需要长期记忆。
2. 命中偏好、客户、从业者、case 或历史事件信号时，生成多条 `RetrievalQuery`。
3. 把长期记忆转成带 metadata 的 `RetrievalDocument`。
4. 使用 `HybridRetriever` 做关键词 + 向量近似 + metadata 的 hybrid search。
5. 使用二阶段 rerank 加权当前性、确定性、业务层级和中文字符相关性。
6. 只把 TopK 召回摘要写入 `memory_recall_results` 和 `memory_context.long_term_recall`。

示例：

- `计算 12*8+3`：不召回长期偏好，也不读取 preference。
- `今天上海天气`：不召回长期客户画像。
- `按我喜欢的风格写客户沟通策略`：召回 preference。
- `这个客户喜欢银行理财，我怎么切入`：召回客户画像、从业者画像、case 和事件记忆。

这样做的目的：

- 降低 token 和检索成本；
- 避免旧偏好污染无关任务；
- 避免把所有客户事实塞进 prompt；
- 让每次记忆召回都有原因、得分和 trace。

## compact_context

策略生成节点必须优先使用 `build_compact_context` 的输出：

```python
{
    "customer_profile": {"confirmed": {}, "uncertain": {}},
    "advisor_profile": {},
    "case_state": {
        "subject_type": "",
        "target_persona": "",
        "trigger_module": "",
        "current_stage": "",
        "kyc_completeness_score": 0,
        "opportunity_score": 0,
        "external_grade": ""
    },
    "missing_fields": [],
    "asked_focuses": [],
    "support_note": "",
    "retrieved_patterns": [],
    "method_knowledge": [],
    "compliance_knowledge": [],
    "news_digest": ""
}
```

它不会输出内部评分公式，不会输出客户真实 PII，不会把 `uncertain` 混成 `confirmed`，也不会直接塞历史全文。

## 本地与生产边界

第一版使用 `InMemoryBusinessMemoryStore` 保持本地 demo 和测试开箱可跑。

生产落地时应替换为 PostgreSQL：

- 业务表使用 UUID 主键。
- 所有业务表必须携带 `tenant_id`。
- JSON 字段使用 JSONB。
- 向量检索使用 pgvector。
- 当前事实表使用 partial index，例如 `(tenant_id, customer_id, fact_key) WHERE is_current = true`。

生产 DDL 以 `migrations/` 中按编号执行到最新版本的文件为准。
[database-schema.sql](database-schema.sql) 仅保留历史设计对照，其中的旧明文字段不能作为部署脚本。

## 补充说明：当前生产实现

上面的“第一版使用内存 Store”仍适用于 `main.py`、直接构造 `WorkflowEngine()` 和单元测试。FastAPI
路径已经完成生产装配：应用 lifespan 只创建一次 `ProductionRuntime`，共享 Redis Client、SQLAlchemy
连接池、`WorkflowEngine`、`ProductionMemoryManager` 和 `PostgresBusinessMemoryStore`。后端不可用时
启动失败，不回退内存实现。

### 短期记忆：Redis

| 层 | Key | TTL 默认值 | 内容 | 并发控制 |
| --- | --- | --- | --- | --- |
| Session | `agent:{tenant}:{session}:session` + `:messages` | 7 天 | Hash 保存 last_intent/entity/active_intent，List 保存最近消息 | WATCH/MULTI/EXEC + version CAS |

Redis Session Hash 保存 `payload`、`version`、`updated_at`，消息单独保存为有界 List，租户访问顺序保存为
LRU ZSet。写入还会执行消息窗口裁剪、单 Payload 字节上限、
租户级 LRU Sorted Set 容量限制和 CAS 指数退避。并发请求如果基于旧版本提交，节点会读取最新版本，
仅合并本轮消息后再尝试一次，并写 `session_memory_cas_retried` trace。

`active_intent` 只保存 intent、pending focus、asked focuses、置信度和独立 `expires_at`。客户家庭、
资产和保险事实不进入该 Redis 信封，而是写入受 RLS、Consent 和版本控制保护的业务事实表；对应
用户 evidence 加密。完成、取消、换题或业务 TTL 到期后清空 Redis 控制信封。

PostgreSQL 不参与在线 Session 窗口读取。`short_term_messages` 仅保存本轮消息的加密原文、脱敏文本和
Hash，作为幂等审计副本；保险工作快照和已问焦点分别保存在 `agent_session_states`、`kyc_questions`。
系统不创建独立任务状态表，也不把审计消息用于自动恢复或断点续跑。

### 长期记忆：PostgreSQL

通用偏好进入 `memory_items`，唯一键为
`(tenant_id, user_id, scope, memory_key)`。`upsert_long_term_memory_item()` 使用真正的
`ON CONFLICT DO UPDATE` 并递增 `version`。向量位于独立 `memory_item_embeddings`，事实、TTL 或 Consent
变化不会重写大向量。通用 Memory 与 RAG 统一为 `halfvec(3072)`，Runtime 启动和 Repository 写入都会
验证维度。

偏好抽取只保存短结构化值，例如 `response_style=简洁一点`、`response_language=中文`。完整用户原句、
客户画像、联系方式、健康、财务和身份信息不会进入通用 Preference。新候选与历史候选按
`type + normalized value` 合并去重；缺少 `preference_memory` Consent 时写入返回 0，并记录跳过 trace。

### 业务表与一致性

生产业务表位于 `migrations/002_business_memory.sql`，不再依赖 `docs/database-schema.sql`：

- 所有业务主体和子表使用 `(tenant_id, id)` 复合主键/外键，允许不同租户使用相同外部 ID；
- `customer_profile_facts`、`advisor_profile_facts` 使用唯一 partial index 保证同一 fact_key 只有一个 current；
- 事实冲突先 `SELECT ... FOR UPDATE`，关闭旧版本，再插入新版本；
- 一份 `MemoryWriteProposal` 在同一个 PostgreSQL Unit of Work 中执行，任一 SQL 失败整体回滚；
- Validator 覆盖 Fact、Event、KYCQuestion、Session Snapshot、AnalysisRun、GeneratedOutput 和 `do_not_store`；
- 风险过滤使用 `risk_rank` 数字，不使用 `risk_level` 文本字典序；
- `_profile_state_to_customer_facts` 按 confirmed/uncertain 的 `source_items` 分区，避免重复写入。

### 加密、RLS 和隐私治理

`short_term_messages`、分析输入/证据和生成输出原文使用 pgcrypto AES-256
字段加密；通用 `memory_items` 的事实证据也使用独立 ciphertext/hash 列。低权限审计只读取脱敏副本、PII 类型摘要和内容 Hash。密钥由
`MEMORY_ENCRYPTION_KEY`/Secret Manager 注入，不写配置、trace 或数据库 metadata。

需要明确区分“证据/原文加密”和“所有业务字段加密”：当前 `customer_profile_facts.fact_value`、
`normalized_value`、`agent_session_states.profile_state/practitioner_state`、`analysis_runs.output_json` 以及
`business_generated_outputs.input_context` 仍以 JSONB 保存，依赖强制 RLS、用途级 Consent、最小权限和
Retention 保护，并未做应用层字段密文。生产环境应同时启用云数据库/磁盘加密；如合规基线要求高敏
KYC 值在数据库管理员视角也不可见，需要新增迁移，把这些 JSONB 拆分或加密并为必要查询建立令牌化索引。

`migrations/003_memory_rls.sql` 对租户表启用并强制 RLS；Repository 每个事务都设置
`app.tenant_id`。客户长期事实写入和读取要求 `memory_processing` Consent，用户偏好要求
`preference_memory` Consent。

FastAPI 提供：

- `POST /memory/consent/grant`
- `POST /memory/consent/revoke`
- `POST /memory/export`
- `POST /memory/delete`

删除客户时按 Lineage、KYC、Event、Output、Analysis、业务 Session、短期消息审计、Conversation、Case、
Fact、Consent 的顺序清理；Embedding 通过外键级联。导出默认不解密原始消息。`make memory-retention` 根据
`expires_at` 分租户、分表、分批清理，不执行全表大事务。

### Migration 约束

`scripts/db_upgrade.py` 使用 advisory lock、`schema_migrations` 台账、SHA-256 校验和和单 migration
事务。遗留数据升级会把旧长期记忆拆成事实表和向量表、加密需保留的审计正文，并由后续迁移删除
已经退出运行架构的任务状态与重复会话消息表。已执行 migration 不能原地修改，结构变化
必须新增下一编号文件。
