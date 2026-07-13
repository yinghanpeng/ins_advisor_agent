# 面试讲解指南

## 一句话介绍

这个项目把原来由 Dify 编排的保险销售沟通教练，重构成 FastAPI + 显式状态机 + 双层意图路由 + 代码化保险会话处理器。Dify 只保留为可选调用端和离线 Prompt 参考，保险运行时不依赖 Dify Workflow。

## 为什么保险逻辑不再放在 Dify

Dify 适合可视化调试，但公网流量和多轮保险状态需要：

- 鉴权；
- 限流；
- 租户隔离；
- 请求契约；
- Redis 活跃意图恢复；
- PostgreSQL KYC 事实版本化；
- 成本控制；
- 结构化日志。

这些职责更适合放在 FastAPI Agent Gateway 和 Agent Core。迁移后，缺失字段、最多补问轮数、知识库选择和风险动作都由可测试的 Python 代码决定，不再由一个大 Prompt 同时承担抽取、评分、路由和生成。

## 为什么用自研显式状态机（AgentGraph）

复杂任务不能靠模型自由发挥。`AgentGraph`（`src/agent_core/graph/builder.py`）用线性顺序的显式状态机表达，把主链路拆成可追踪的节点函数：

- 当前在哪个状态；
- 下一步去哪；
- 工具失败怎么恢复；
- 高风险输出怎么同步阻断或降级；
- 每一步怎么 trace。

## 为什么接 LangSmith

LangSmith 用于：

- 根 Run 与状态节点 Run Tree 可视化；
- 节点 `state_before/state_after` 差异、耗时和失败定位；
- 真实模型 messages、供应商响应、工具、RAG、KYC、Prompt 和最终回答回放；
- dataset；
- evaluator；
- experiment；
- run comparison。

项目支持控制面和完整业务内容两种数据策略，但 API Key、密码、Authorization、Cookie 和认证 Token 永远强制
脱敏。与此同时，本地结构化日志必须始终可用，LangSmith 不可用不能影响主业务。

## 为什么 Sales Intelligence 不是普通 RAG

销售访谈不是“资料”，而是一线销售经验资产。它要被加工成：

- 洞察卡片；
- 话术库；
- 异议处理库；
- 案例库；
- 计划书成交库；
- Eval case。

原始访谈不能直接进入最终 Prompt，高风险话术必须被拦截。

## 面试时可以这样讲主链路

```text
用户输入
→ FastAPI / main.py
→ WorkflowEngine
→ Input Guardrail（硬规则 / 灰区模型 / 统一决策）
→ Redis Session 与 active_intent 恢复
→ active intent 漂移判断，或向量 TopK + LLM 意图裁定
→ 0.80 / 0.60 执行度判断
→ 通用 Tool Schema 路径，或代码化 Insurance Handler
→ 保险 KYC 增量抽取与代码缺口判断
→ 状态分支：温和追问 / 双知识库与可选新闻后生成策略 / 低压维护
→ 本轮结果 Proposal + Validate + Persist（缺 Consent 时无持久化降级）
→ Grounding + PII + Compliance
→ Response Packaging + Output Logger + Active Intent / 短期状态更新
→ Final Answer + Trace
```

双知识库只在信息足以生成策略的 `matched` 分支使用；`insufficient` 只生成一个问题，不为追问请求做
无关检索。业务写入在追问或策略已经生成后、输出检查前执行，因此不会把尚未展示的问题标成已问。

意图路由的两个阈值层次要分开讲：第一层的 `0.85/0.60` 是向量相似度；第二层的 `0.80/0.60` 是 LLM 裁定置信度。前者决定是否调用 LLM，后者决定直接分发、分发并记录还是主动澄清。

保险多轮也不是全局 Slot Manager。通用工具只使用自己的 `ToolSpec.input_schema`；只有保险领域维护 `InsuranceKycDelta`，用于解释“两个”“他自己拍板”这类上一问的短回答，并把明确事实写入业务记忆。

## 如何证明不是 Demo

可以展示：

- `WorkflowStepContract`；
- `ToolSpec`；
- `HybridRetriever`；
- `MemoryManager`；
- `InputGuardrail` / `OutputGuardrail`；
- `IntentRouter` / `ActiveIntentState`；
- `InsuranceKycExtractor` / `InsuranceKnowledgeProvider`；
- `CostBudget`；
- `evals/dataset.jsonl`；
- `tests/`。

## 当前限制怎么讲

不要假装都接好了。可以诚实说：

- 本地没有模型或知识库配置时使用明确的规则/空知识降级，生产模式不会伪造外部事实；
- 向量阈值是初始值，上线前必须按实际 Embedding 模型和验证集校准；
- 最终策略生成当前仍以确定性模板为主，真实生成模型是后续接入项；
- 生产接入时替换 provider，不改 Agent Core 边界。

客户请求没有人工审批或挂起状态。高风险输入、工具和输出在当前请求内直接阻断、脱敏或返回无副作用的安全替代方案；离线知识条目的发布标记不等于运行时人工审批。
