# Dify Control Plane 说明

Dify 在本项目中是可选调用端和离线 Prompt 参考，不负责保险意图、KYC、知识检索或生产状态编排。

## 推荐集成流程

1. Dify 接收用户输入；
2. Dify 通过 HTTP 节点调用 `POST /agent/run`；
3. FastAPI Agent Gateway 把请求交给 Agent Core；
4. Agent Core 先做 Input Guardrail，再处理 Redis active intent、向量 + LLM 意图路由、代码化保险 Handler、工具、RAG、Guardrails 和日志；
5. Dify 展示最终回答和 trace id。

HTTP 节点必须同时发送 `X-Tenant-ID`、租户绑定的 `X-API-Key` 和相同的 body `tenant_id`。API Key 放在
Dify Secret 中，不写入导出 YAML，也不下发给客户浏览器。`/agent/run` 返回精简
`PublicAgentRunResponse`，不会暴露完整 Trace、知识正文或工具审计。

## 为什么这样拆

这样可以保留 Dify 的可视化调试优势，同时保证所有渠道复用同一套 Data Plane。Dify 不应传保险 workflow 名、强制 `domain_skill` 或保存 KYC 控制状态，也不能绕过高风险同步阻断。

`dify/workflow.yml` 仅描述推荐 HTTP 调用变量；附件旧工作流只作为迁移输入，不是运行时依赖。

节点说明见：`dify/nodes/*.md`。
