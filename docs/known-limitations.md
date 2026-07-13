# 当前限制

本项目已经具备生产级结构和本地可运行骨架，但仍有一些能力需要接真实服务。

## 1. 最终生成仍以确定性实现为主

意图裁定、活跃意图变化和 KYC 抽取已经提供 OpenAI-compatible 模型适配器；最终通用回答和保险策略仍以可测试的确定性实现为主。接入真实生成模型时必须只消费经过 Source Boundary 的上下文，并保留 Grounding、PII 和 Compliance 复检。

## 2. 本地意图向量不代表生产 Embedding

`intent_routing.provider=local` 使用字符 n-gram 稀疏向量，便于离线测试；生产应切换 `pgvector` 并使用真实脱敏意图语料。`0.85/0.60` 和 `0.80/0.60` 是初始阈值，必须按实际模型校准。

FastAPI Runtime 在 `APP_ENV=staging/stage/prod/production` 时会拒绝 local 意图/保险知识 Provider，并
要求意图裁定、漂移检测和 KYC 抽取模型完整；这能防止配置遗漏时静默降级，但不能替代上线前阈值评估。

## 3. 外部工具仍需要真实 Provider

`web_search`、`news_search`、`weather`、`file_parser` 当前是 adapter/mock。后续应接真实 API，并补 timeout、retry、error schema。

## 4. LangSmith 运行时 Trace 已接入，Experiment 尚未自动化

配置 API Key、网络、Project 并启用 tracing 后，系统会写入根 Run 和状态节点子 Run；未配置或远端故障时
自动降级为本地日志。当前尚未自动创建 Dataset、执行远程 Experiment 或回写 evaluator feedback，生产环境也
仍需补充 Workspace 权限、采样率、保留周期和告警治理。

## 5. Sales Insight 抽取是本地实现

生产可替换为 LLM + JSON Schema + repair + compliance review。知识条目是否允许用于生成是离线发布治理，不是客户请求中的人工审批步骤。

## 6. 生产持久化需要外部服务与运维保障

代码已提供 Redis Session/Task、PostgreSQL 业务记忆与 pgvector Provider；生产仍需配置真实服务、迁移、备份、监控、密钥管理和租户隔离验证。本地 CLI/测试默认使用内存 Store，不代表生产配置。

## 7. 没有后台人工审批渠道

这是面向客户的同步系统，故意不提供人工审批队列、挂起和恢复审批能力。高风险输入、越权工具、写操作、外部动作和高风险输出会在当前请求中直接阻断或返回安全替代；如果未来要支持后台运营系统，应作为新的产品边界单独设计，不能复用客户请求链路暗中加审批。

## 8. 规范化 KYC JSONB 尚未做应用层字段加密

用户原文、事实 evidence、分析输入和生成输出正文已经使用 pgcrypto；但规范化事实值、Session 画像、
分析结构化输出和生成输入上下文当前仍是 JSONB，主要依赖强制 RLS、用途级 Consent、最小权限、保留
期限和数据库/磁盘加密。若部署基线要求这些值对数据库管理员也不可见，需要新增字段密文迁移与必要
的令牌化检索设计，不能把当前实现描述为“全字段加密”。

## 9. 用户主体仍需由企业身份网关绑定

当前 API Key 已绑定 tenant，公开 metadata 也不能指定客户、顾问、会话或 Case ID；但基础 API 尚未
内置 OAuth/JWT。面向浏览器上线时必须由可信 Gateway 根据登录态覆盖 `user_id/session_id`，并禁止客户端
持有租户 API Key 或自行声明主体。否则只能视为受信服务间接口，不能视为完整的终端用户鉴权方案。
