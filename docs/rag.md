# RAG 检索设计

本项目区分四类检索用途，不能混用分数和 library：

- 通用 RAG：检索普通知识文档；
- 意图知识库：按纯向量相似度执行 0.85/0.60 双层路由；
- 保险双知识库：方法/匿名案例与合同/合规内容使用独立 library、TopK 和阈值；
- Sales Intelligence RAG：只检索已通过静态生成准入的销售洞察卡片。

原始销售访谈不能直接进入最终生成，只能先变成结构化卡片。

## 检索契约

当前 RAG 契约包含：

- query rewrite：原始 query、销售痛点 query、客户类型 query、场景 query、策略 query；
- metadata：source id、chunk id、library、tenant id、tags、risk level、静态生成准入标志；
- hybrid search：lexical score + vector-like score + metadata score；
- rerank：加权总分和可追踪分数组件；
- evidence compression：把选中证据压缩为 digest。

## 当前代码

- `src/agent_core/rag/schemas.py`
- `src/agent_core/rag/query_rewrite.py`
- `src/agent_core/rag/retriever.py`
- `src/agent_core/rag/reranker.py`
- `src/agent_core/sales_intelligence/retriever.py`
- `src/agent_core/intents/knowledge_base.py`
- `src/agent_core/skills/insurance_advisor/knowledge.py`
- `src/agent_core/persistence/postgres.py`

## 本地与生产路径

本地 `HybridRetriever` 与字符 n-gram 意图库用于确定性测试。FastAPI Runtime 可按配置注入 PostgreSQL /
pgvector 意图库与保险双知识库，并复用同一个 3072 维 Embedding Client；`staging/prod` 禁止 local
Provider。意图阈值只读取 `vector_score`，不能用混合 `final_score` 替代余弦相似度。

未来可以为通用大规模检索增加 Elasticsearch/OpenSearch 和模型 Reranker，但应保持现有
`RetrievalResult`、租户过滤、Source Boundary 与静态生成准入契约不变。

`approved_for_generation` 是离线/自动内容治理后的静态发布位，不表示客户请求要等待人工审批。缺失、
`false` 或字符串 `"true"` 都按 default deny 处理。
