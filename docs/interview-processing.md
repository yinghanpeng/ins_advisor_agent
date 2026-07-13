# 销售访谈处理流水线

销售访谈是本项目的核心业务资产，不能直接作为普通文本塞进 Prompt。

## 标准流水线

1. `ingest_raw_interview`：接入原始访谈；
2. `anonymize_interview`：脱敏姓名、电话、邮箱、金额等；
3. `clean_transcript`：清理转写稿；
4. `segment_by_scene`：按销售场景切片；
5. `extract_structured_insight`：抽取结构化销售洞察卡片；
6. `review_card`：做合规审查；
7. `SalesInsightIndexer.save`：保存卡片；
8. `SalesIntelligenceRetriever.retrieve`：检索通过生成准入的卡片；
9. `build_sales_insight_digest`：压缩为生成可用 digest；
10. `generate_eval_case`：生成评估样本。

## 当前限制

当前抽取是本地 deterministic 实现，适合测试工程链路。生产环境应替换为：

- LLM 抽取；
- JSON Schema 校验；
- JSON repair；
- 合规模型或规则审查；
- 自动脱敏、风险与合规生成准入；
- 持久化索引。
