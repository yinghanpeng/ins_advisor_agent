# 成本控制

成本控制用于避免 Agent 在复杂任务里无限调用模型、工具和检索。

## 当前能力

- `CostBudget` 记录请求级 token budget；
- `CostDecision` 返回结构化预算决策；
- `model_router.py` 预留模型选择策略；
- `configs/cost_budget.yaml` 保存预算配置。

## 预算压力下的策略

1. 降低 retrieval top-k；
2. 跳过可选新闻检索；
3. 压缩上下文；
4. 切换更便宜的模型；
5. 返回保守降级答案，并记录原因。

## 后续扩展

生产环境应接入真实模型 token usage callback，并按 tenant / user / workflow 做预算隔离。

