# 业务记忆系统设计

本文说明保险高客沟通教练如何从 Dify workflow 的字符串变量，升级为可持久化、可审计、可评测的业务记忆系统。

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

## Dify 18 字段映射

| Dify 字段 | 工程化位置 | 说明 |
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
| `kyc_question_round_count` | `AgentState` / `AgentSessionState` | 最多 4 轮，第 5 轮后不再卡 insufficient。 |
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

- 短期/任务记忆：`session`、`task`，用于多轮指代、最近消息和 workflow 恢复，可以每轮读取；
- 长期记忆：`preference`、客户画像、从业者画像、case 事件，只在当前请求确实需要时召回。

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

DDL 见 [database-schema.sql](database-schema.sql)。
