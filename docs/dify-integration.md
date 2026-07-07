# Dify 集成说明

Dify 在本项目中是 Control Plane，不是生产 Data Plane。

## Dify 负责什么

- Prompt 管理；
- 可视化 Workflow 配置；
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

## 推荐 HTTP 节点请求体

```json
{
  "input": "{{sys.query}}",
  "session_id": "{{conversation.id}}",
  "domain_skill": "insurance_advisor",
  "metadata": {"source": "dify"}
}
```

## 相关文件

- `dify/workflow.yml`
- `dify/nodes/*.md`
- `src/agent_core/integrations/dify_webhook.py`

