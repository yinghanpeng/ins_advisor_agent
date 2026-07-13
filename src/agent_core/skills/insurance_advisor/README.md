# Insurance Advisor Skill 保险顾问技能

这是当前项目的第一个垂直业务 Skill，也是附件 Dify 保险工作流迁移后的代码化运行实现。

## 这个 Skill 负责

- 保险/金融从业者沟通教练场景；
- 高客破冰；
- KYC 追问；
- 异议处理；
- 计划书推进；
- 业务 Prompt；
- 业务合规表达。
- Redis active intent 的续接结果消费；
- 保险领域 KYC 增量抽取、合并、缺口和最多补问轮次；
- 沟通方法库与合同合规库的独立检索；
- 公开新闻的按需调用、清洗和证据压缩。

## 这个 Skill 不负责

- 通用工具；
- Memory 基础设施；
- RAG 基础设施；
- Trace；
- Recovery；
- Cost Control；
- Gateway。

这些能力都在 Agent Core 中。

## 运行边界

- 外部调用方不传保险 `workflow_name`；`IntentRouter` 命中四个保险细分意图后自动进入本 Skill。
- 通用工具参数由各自 `ToolSpec.input_schema` 校验。本目录没有全局 Slot Manager，`InsuranceKycDelta` 只服务保险多轮业务状态。
- LLM 只抽本轮明确事实；每个模型值还必须携带当前输入中的精确 evidence span，并达到
  `kyc_evidence_min_confidence`。Python 负责证据校验、合并、评分、缺失字段、最多轮次和下一状态。
- 一轮最多问一个温和问题。任务完成、取消、切换或 TTL 到期后清除 active intent。
- 追问/策略先生成，再创建并校验记忆 Proposal；只有实际展示的问题才记录为已问。缺少 Consent 时
  继续本轮回答但不读写业务记忆。
- 新保险细分任务重置旧问题和轮次、保留已验证客户事实；低置信换题使用 `switch_pending`，确认前不
  销毁旧 active intent。
- 客户请求没有人工审批或挂起分支；高风险操作同步阻断或降级。

## 关键文件

- `kyc.py`：领域字段、模型/规则抽取、合并、评分和温和问题。
- `knowledge.py`：方法库与合同合规库 Provider、Query 和证据结构。
- `skill.yaml`：可执行保险意图和代码处理器声明。
- `configs/intent_routing.yaml`：相似度、置信度、active TTL 和最大补问轮数。
- `configs/insurance_handler.yaml`：双知识库、TopK、阈值和新闻开关。
- `configs/models.yaml`：意图裁定、漂移检测和 KYC 抽取模型端点。

完整说明见仓库 `docs/intent-routing-and-insurance-handler.md`。
