# Product Contract 产品接口契约

Agent Gateway 的请求和响应必须稳定，否则 Dify、前端、测试和外部系统都会难以接入。

## 输入字段

- `input`：用户本轮输入；
- `session_id`：会话 ID；
- `user_id`：用户 ID，可为空；
- `tenant_id`：租户 ID；
- `workflow_name`：使用的 workflow；
- `domain_skill`：指定业务 Skill，可为空；
- `metadata`：来源、调试信息、Dify 参数等。

## 输出字段

- `trace_id`：本次请求唯一 trace；
- `session_id`：会话 ID；
- `final_state`：最终状态；
- `answer`：最终回答；
- `intent`：意图识别结果；
- `domain_skill`：命中的业务 Skill；
- `guardrails`：安全合规审查结果；
- `retrieved_context`：检索证据；
- `trace_events`：结构化 trace；
- `state_transitions`：状态迁移记录。

## 代码位置

- `src/agent_core/workflow/contracts.py`

所有进入下游逻辑的模型输出都必须有结构化契约或校验机制。

