# Sales Intelligence Layer 子模块说明

这个模块负责把一线销售访谈语料加工成可检索、可审查、可评估的业务资产。

## 标准流程

1. `ingest_raw_interview`
2. `anonymize_interview`
3. `clean_transcript`
4. `segment_by_scene`
5. `extract_structured_insight`
6. `review_card`
7. `SalesInsightIndexer.save`
8. `SalesIntelligenceRetriever.retrieve`
9. `build_sales_insight_digest`
10. `generate_eval_case`

保险运行时由代码化 Insurance Handler 通过受控 Retriever 消费已发布的脱敏卡片；原始访谈、未发布卡片和高风险片段不会直接拼入最终 Prompt。该离线发布状态不等于客户请求需要人工审批。

## 关键原则

- 原始访谈不直接进入最终 Prompt；
- 高风险话术不能用于生成；
- 每张卡片必须能追溯 `source_id` 和 `chunk_id`；
- 检索结果要经过 evidence compression；
- 高频销售问题可以转成 eval case。
- 方法案例与合同合规知识必须使用不同 library，避免相似案例挤掉强制合规边界。
