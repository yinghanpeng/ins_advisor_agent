# Dify 集成说明

Dify 在本项目中是 Control Plane，不是生产 Data Plane。

## Dify 负责什么

- Prompt 管理；
- 可视化调用编排和内部调试；
- 内部运营调试；
- 业务人员测试入口；
- 通过 HTTP 节点调用 FastAPI Agent Core。

## Agent Core 负责什么

- 公网请求处理；
- 鉴权和限流；
- 租户隔离；
- 显式状态机；
- 工具路由；
- RAG / Memory / Context；
- Guardrails；
- Recovery；
- Cost Control；
- 本地结构化日志；
- LangSmith trace / eval adapter。
- 双层意图识别、Redis active intent 和代码化保险会话处理器。

## Dify Dataset 到 pgvector 的知识映射

Dify Dataset 是知识维护源，在线客户请求仍由 Agent Core 的 PostgreSQL/pgvector 执行检索。映射配置位于
`configs/dify_knowledge.yaml`：`intent_catalog`、`insurance_methods` 和 `insurance_compliance` 必须分别绑定
独立 Dataset 与独立 Dataset API Key。真实值写入 `.env` 的 `DIFY_*_DATASET_*` 环境变量，不能写入 YAML。

当前配置只完成映射和密钥占位；Dataset 同步入库脚本需要在三个 Dataset 均可访问后实现。Dify App/Workflow 的
`DIFY_API_KEY` 不应假定能替代 Dataset API Key。

## 推荐 HTTP 节点请求体

```json
{
  "input": "{{sys.query}}",
  "session_id": "{{conversation.id}}",
  "tenant_id": "{{agent_core_tenant_id}}",
  "metadata": {"source": "dify"}
}
```

HTTP Header 同时发送 `X-Tenant-ID: {{agent_core_tenant_id}}` 和保存在 Dify Secret 中的
`X-API-Key: {{agent_core_api_key}}`。Header 与 body tenant 不一致时 Gateway 返回 403。Dify 只消费客户
安全的 `PublicAgentRunResponse`；完整 Trace 与知识正文留在 Agent Core 服务端。

不要从 Dify 传保险 workflow 名或强制 `domain_skill`。Agent Core 必须先执行 Input Guardrail，再根据 active intent、向量相似度和 LLM 裁定自动路由。Dify 不保存保险 KYC 控制状态，也不能绕过同步高风险阻断。

## 相关文件

- `dify/workflow.yml`
- `dify/nodes/*.md`
- `src/agent_core/integrations/dify_webhook.py`
- `docs/intent-routing-and-insurance-handler.md`
