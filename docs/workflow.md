# Workflow 设计

Workflow 是比单个 Prompt 更稳定的任务流程定义。

## 通用流程

```text
用户输入
→ CLASSIFY_INTENT
→ ROUTE_CAPABILITY
→ 通用能力 或 Domain Skill
→ 检索 / 工具 / 上下文构建
→ GENERATE_RESPONSE
→ COMPLIANCE_REVIEW
→ FINAL / HUMAN_APPROVAL / ERROR
```

## 保险破冰 Workflow

1. classify intent；
2. extract customer KYC；
3. extract sales pain；
4. classify scene；
5. retrieve Sales Intelligence；
6. optionally retrieve external news；
7. build compact context；
8. generate response；
9. compliance review；
10. final response。

## Step Contract

Workflow step 定义在：

- `src/agent_core/workflow/contracts.py`
- `src/agent_core/workflow/steps.py`

每个 step 声明：

- required inputs；
- produced outputs；
- allowed next states；
- guardrails；
- allowed tools；
- retry policy；
- trace fields。

这样做的好处是：节点不再是“隐形 Prompt”，而是有明确输入输出边界的工程模块。

