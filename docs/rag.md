# RAG 检索设计

本项目区分两类检索：

- 通用 RAG：检索普通知识文档；
- Sales Intelligence RAG：检索已审核的销售洞察卡片。

原始销售访谈不能直接进入最终生成，只能先变成结构化卡片。

## 检索契约

当前 RAG 契约包含：

- query rewrite：原始 query、销售痛点 query、客户类型 query、场景 query、策略 query；
- metadata：source id、chunk id、library、tenant id、tags、risk level、approval flag；
- hybrid search：lexical score + vector-like score + metadata score；
- rerank：加权总分和可追踪分数组件；
- evidence compression：把选中证据压缩为 digest。

## 当前代码

- `src/agent_core/rag/schemas.py`
- `src/agent_core/rag/query_rewrite.py`
- `src/agent_core/rag/retriever.py`
- `src/agent_core/rag/reranker.py`
- `src/agent_core/sales_intelligence/retriever.py`

## 生产替换路径

1. 用 Elasticsearch / OpenSearch 替换本地词法打分；
2. 用 embedding + vector database 替换本地 vector-like score；
3. 接入模型 reranker；
4. 保持 `RetrievalResult` 输出结构不变，避免影响下游 Context Builder。

