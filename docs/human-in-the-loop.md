# Human-in-the-loop 人工审批

当 Agent 遇到高风险动作时，不应该强行自动执行，而应该进入人工审批。

## 需要人工审批的场景

- 高风险工具调用；
- 输出中出现保险/金融高风险表达；
- 销售访谈中抽取出高风险话术；
- 卡片有业务价值但需要合规人员修改；
- 系统无法安全判断应该 block 还是 rewrite。

## 当前实现

代码位置：

- `src/agent_core/guardrails/human_approval.py`

当前包含：

- `ApprovalRequest`：审批请求；
- `ApprovalDecision`：审批结果；
- `InMemoryApprovalStore`：本地内存审批队列。

## 生产扩展

后续可以接：

- 数据库；
- 审批后台；
- 企业微信/飞书通知；
- 审批 SLA；
- 审批结果回写 LangSmith trace。

