# workflow_router 工作流路由节点

职责：保留 Dify 调用标签的兼容说明；真实业务路由由 Agent Core 决定。

示例：

- `universal_agent_workflow`（兼容调用标签；保险由 Agent Core 意图路由自动处理）

保险运行逻辑不再在 Dify 中编排。Dify 只调用 `/agent/run`，不得通过 workflow 名绕过
Input Guardrail、活跃意图、向量/LLM 裁定或代码化 KYC 处理器。

旧 `kyc_question_workflow`、`objection_handling_workflow` 和 `proposal_closing_workflow` 不再是可执行运行分支。
