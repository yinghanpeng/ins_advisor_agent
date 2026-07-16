# Domain Skills 业务技能

Domain Skill 是 Agent Core 之上的垂直业务能力；可执行能力由 `DomainAgentRegistry` 注册和发现。

## Skill 负责什么

- 业务代码处理器及显式状态；
- 业务 Prompt；
- 业务路由；
- 业务术语；
- 业务合规边界；
- 业务示例和 eval case。

## Skill 不负责什么

- 网关；
- 通用工具；
- 通用 RAG；
- Memory 基础设施；
- Trace；
- Recovery；
- Cost Control。

当前已实现的 Skill：

- `src/agent_core/agents/advisor_coach/`：可执行保险顾问 Agent，承接领域 KYC、双知识库和沟通策略顺序；
- `src/agent_core/skills/insurance_advisor/`：保险顾问 Agent 使用的 KYC、知识 Provider 和 Prompt 资产；
- `src/agent_core/agents/insurance_proposal/`：计划书契约和默认禁用占位 Agent。占位健康检查返回 `True`，但 `available=False` 且不参与在线路由。

## 运行时选择顺序

```text
IntentRouter
  → intent + domain_skill
  → DomainAgentRegistry.resolve()
  → AdvisorCoachAgent（已启用）
  → InsuranceProposalAgentPlaceholder（默认禁用，不会被 resolve 选中）
```

`AgentGraph` 仍负责输入安全、会话恢复、意图识别和通用能力；一旦 Registry 命中专业 Agent，领域内
部步骤由该 Agent 自己维护。调用方不能通过 `workflow_name` 绕过意图白名单。

## 接入真实计划书 Agent

真实实现需要遵守 `DomainAgent`，并使用 `ProposalTaskRequest` / `ProposalAgentResult` 作为远程边界。
启动装配时通过 `WorkflowEngine(proposal_agent=real_agent)` 注入即可替换默认占位对象，现有保险顾问
和通用请求不需要修改。

未来可以继续增加：

- 研究助手；
- 文档分析助手；
- 面试助手；
- 客服助手；
- 数据分析助手。
