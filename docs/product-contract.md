# Product Contract 产品接口契约

Agent Gateway 的请求和响应必须稳定，否则 Dify、前端、测试和外部系统都会难以接入。

## 输入字段

- `input`：用户本轮输入；
- `session_id`：会话 ID；
- `user_id`：用户 ID，可为空；
- `tenant_id`：租户 ID；
- `workflow_name`：兼容调用标签，默认 `universal_agent_workflow`；不得用于强制选择保险路径；
- `domain_skill`：指定业务 Skill，可为空；
- `metadata`：default-deny 允许列表，只接受 `source/client/channel/experiment_group/eval_id/request_id/locale`；
  不得携带生成知识正文、新闻、对话模式、业务记录 ID、预算/重试控制或 `_trusted_*` 内部来源标记。

公开请求中的 `advisor_id/customer_id/conversation_id/opportunity_case_id` 会被 Pydantic 契约拒绝，
直接构造 `AgentState` 时图入口也会删除未受信值。业务主体由网关绑定的 `user_id/session_id` 派生；当前
基础 API Key 只绑定租户，企业部署还必须由 JWT/API Gateway 覆盖 `user_id/session_id`，不能让浏览器
自行声明身份。

## 客户 HTTP 输出字段

- `trace_id`：本次请求唯一 trace；
- `session_id`：会话 ID；
- `final_state`：最终状态；
- `answer`：最终回答；
- `intent`：意图识别结果；
- `active_intent`：不含客户 KYC 值、模型置信度和内部来源的多轮控制信封；
- `insurance_kyc_status`：仅返回信息状态、缺失字段名、已问焦点和轮次，不返回机会/完整度评分；
- `domain_skill`：命中的业务 Skill；
- `citations`：只包含 source/chunk/risk 等脱敏标识，不含知识正文；
- `next_actions` / `warnings` / `clarification_question`：客户可展示的下一步、降级提示与澄清问题。

`WorkflowEngine` 内部仍返回 `AgentRunResponse`，供单元测试、CLI 与受信诊断使用。它包含
`intent_routing_result`、`guardrails`、`retrieved_context`、`trace_events`、`state_transitions`、
`tool_calls/tool_results`、`query_understanding/context_needs`、Grounding、Evaluation 与 Cost。
FastAPI `/agent/run` 和 `/agent/stream.final_response` 只返回 `PublicAgentRunResponse`，不会把这些诊断字段
或检索正文发给客户。

## 代码位置

- `src/agent_core/workflow/contracts.py`

所有进入下游逻辑的模型输出都必须有结构化契约或校验机制。

低置信意图返回澄清问题，不执行工具或保险处理器。客户渠道不会返回 `pending_approval`：高风险请求在同一响应内同步阻断、脱敏或降级。
