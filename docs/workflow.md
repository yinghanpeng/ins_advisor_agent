# 代码执行流设计

项目类名仍保留 `WorkflowEngine` 作为 API 兼容门面，但保险不再使用独立 Workflow 或
`workflow_name` 分叉。实际顺序位于 `AgentGraph` 和节点函数中。

## 统一入口

```text
initialize_context
→ input_guardrail
→ restore_memory
→ normalize_messages
→ active intent / vector intent / LLM adjudication
→ confidence dispatch
→ general path 或 insurance code handler
→ output checks
→ memory update
→ trace_finalize
```

## 保险代码处理器

```text
load_business_memory
→ extract_insurance_kyc_slots
→ deterministic merge / scoring / route
→ one gentle question 或 dual-KB strategy
→ memory proposal / validation / persistence
→ grounding / PII / compliance
→ active-intent sync
```

写入放在生成之后：`generate_kyc_questions` 先设置实际展示的 `presented_kyc_focus`，Proposal 才记录该
问题。这样生成异常不会把客户没有看到的问题误算成已问。

附件 Dify Workflow 的 KYC、新闻、双知识库和策略规则已经分别迁到 `kyc.py`、`knowledge.py`、
`nodes.py` 和 `builder.py`。原 `workflow/steps.py` 与 Insurance Workflow 类已删除。

公共 Pydantic 契约仍保留在 `workflow/contracts.py`，用于 API 请求/响应和通用结构，不再作为保险
业务编排的第二套事实来源。
