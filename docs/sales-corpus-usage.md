# 销售语料资产化使用说明

真实销售/客户对话不能作为普通 RAG 知识库直接检索给生成模型。

原因很简单：真实对话里包含个体客户事实、隐私、上下文偶然性和未通过生成准入的话术。如果直接 RAG，模型可能把某个真实客户事实泛化给另一个客户，或者编造“以前有个客户后来成交了”之类不可审计故事。

## 允许用途

真实语料只能作为以下资产使用：

- 原始语料归档；
- 脱敏语料；
- KYC 抽取训练/评测数据；
- 销售动作模式；
- 客户反应模式；
- 异议处理模式；
- 下一最佳动作模式；
- 话术风格样本；
- 销售质量 eval cases。

## 结构化模型

Sales Intelligence Layer 新增四类语料资产：

- `CorpusBatch`：一次导入批次，记录来源、上传人、文件 URI 和 PII 状态。
- `CorpusCase`：一个脱敏整理后的销售案例资产。
- `CorpusMessage`：脱敏消息，只能用于抽取和评测，不直接进最终 Prompt。
- `DialoguePattern`：从真实对话中抽取、脱敏、审查后的销售动作模式。

## 生成准入规则

最终生成节点只能使用：

- `approved_for_generation=True` 的 `DialoguePattern`；
- `approved_for_generation=True` 的 `SalesInsightCard`；
- 已经过合规审查的模式摘要；
- 外部新闻或公开素材的摘要。

最终生成节点禁止使用：

- 原始 `CorpusMessage`；
- 未脱敏对话；
- 未通过生成准入的 `DialoguePattern`；
- `risk_level="high"` 的模式；
- 单个真实客户故事；
- 无证据的成交归因；
- “以前有客户这样做后来成交了”这类不可审计表达。

## 检索返回什么

销售语料检索返回的是“模式摘要”，不是完整客户故事。

一个安全的 `DialoguePattern` 摘要通常包含：

- `pattern_type`
- `scene_type`
- `target_persona`
- `trigger_module`
- `situation_summary`
- `customer_signal`
- `recommended_move`
- `bad_move`
- `example_wording`
- `outcome_label`
- `confidence`

这些字段表达的是可迁移模式，而不是某个真实客户的事实。

## 评测闭环

语料资产还会进入 eval：

- KYC 抽取是否只写明确事实；
- uncertain 线索是否正确标记；
- 销售动作是否低压；
- 异议处理是否合规；
- 是否错误引用真实客户故事；
- 是否把未通过生成准入的模式用于生成；
- 是否达到 `intent_routing.max_kyc_question_rounds` 后仍错误卡在 insufficient。

这些 eval case 可以来自 `CorpusCase` 和 `DialoguePattern`，但评测样本也必须脱敏。
