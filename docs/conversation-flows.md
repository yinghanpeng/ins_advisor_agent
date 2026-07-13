# 典型对话链路

## 1. 保险破冰首轮

输入：

```text
客户喜欢银行理财，我怎么破冰
```

本地意图目录包含完全匹配样例，因此 `vector_score=1.0`，不调用 LLM 意图裁定，直接命中
`insurance_break_ice`。保险 Handler 抽取 `active_asset_types=[银行理财]`，发现客户角色等核心字段
缺失，只问一句温和问题，并写入 Redis active intent。

```text
classify_intent(vector_direct)
→ semantic_risk
→ load_business_memory
→ extract_insurance_kyc_slots
→ analyze_kyc_and_route(insufficient)
→ generate_kyc_questions
→ validate and persist shown question
→ save active_intent
→ FINAL
```

## 2. 回答上一轮问题

下一轮输入：

```text
他是企业主
```

系统先读取 active intent，小模型/保守判断认为这是上一问的回答，跳过全局向量检索，直接续接
`insurance_break_ice`。KYC Extractor 把 `customer_role=企业主` 合并进业务事实，然后选择下一个焦点。

如果 pending focus 是 `children_count`，用户只回答“两个”，也会被解释成孩子数量，而不是一个全新意图。

## 3. 对话中换题

保险补问期间输入：

```text
先不聊这个，帮我查上海天气
```

漂移检测输出 `switch`，系统重新执行向量路由并命中 `weather_query`，随后走 Tool Schema 和天气工具。
只有新路由达到执行阈值后才替换旧 active intent；若新意图低置信，则先澄清并用 `switch_pending`
保留旧任务，避免一次模糊表达永久丢失补问状态。

## 4. 中低置信表达

- 向量分 0.60~0.85：LLM 在 TopK 候选中裁定；
- 向量分低于 0.60：LLM 仍只能选白名单意图；
- LLM confidence 0.60~0.80：执行并记录中置信日志；
- confidence 低于 0.60：主动询问用户目标，不调用工具和保险 Handler。

## 5. 信息达到上限后生成策略

保险 KYC 默认每轮一个问题，最多三轮。达到上限或用户明确要求“先给策略”后：

```text
methods KB
→ contract/compliance KB
→ optional news tool + Python cleaner
→ compact context
→ nine-section initial strategy
→ proposal / validation / persistence
→ grounding / PII / compliance
→ clear active intent
```

## 6. Prompt Injection

输入 Guardrail 始终发生在 Memory、向量检索和模型之前。明确提示劫持、越权或保险欺诈请求会同步
阻断/降级，不会因高向量相似度进入任何执行路径。
