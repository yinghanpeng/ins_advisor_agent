# 面试讲解指南

## 一句话介绍

这个项目把一个 Dify 保险销售沟通教练，升级为 FastAPI + LangGraph + LangSmith + Dify Control Plane + Agent Core + Sales Intelligence Layer 的生产级 Agent Framework。

## 为什么 Dify 不做主入口

Dify 适合 Prompt 管理和运营调试，但公网流量需要：

- 鉴权；
- 限流；
- 租户隔离；
- 请求契约；
- 状态恢复；
- 成本控制；
- 结构化日志。

这些职责更适合放在 FastAPI Agent Gateway 和 Agent Core。

## 为什么用 LangGraph

复杂任务不能靠模型自由发挥。LangGraph 用显式状态机表达：

- 当前在哪个状态；
- 下一步去哪；
- 工具失败怎么恢复；
- 高风险输出怎么进入人工审批；
- 每一步怎么 trace。

## 为什么接 LangSmith

LangSmith 用于：

- trace 可视化；
- dataset；
- evaluator；
- experiment；
- run comparison。

但本地结构化日志必须始终可用，LangSmith 不可用不能影响主业务。

## 为什么 Sales Intelligence 不是普通 RAG

销售访谈不是“资料”，而是一线销售经验资产。它要被加工成：

- 洞察卡片；
- 话术库；
- 异议处理库；
- 案例库；
- 计划书成交库；
- Eval case。

原始访谈不能直接进入最终 Prompt，高风险话术必须被拦截。

## 面试时可以这样讲主链路

```text
用户输入
→ FastAPI / main.py
→ WorkflowEngine
→ LangGraph 状态机
→ Intent Router
→ Capability Router
→ Insurance Advisor Skill
→ Sales Intelligence Retriever
→ Context Builder
→ Strategy Generator
→ Compliance Review
→ Final Answer + Trace
```

## 如何证明不是 Demo

可以展示：

- `WorkflowStepContract`；
- `ToolSpec`；
- `HybridRetriever`；
- `MemoryManager`；
- `InputGuardrail` / `OutputGuardrail`；
- `InMemoryApprovalStore`；
- `CostBudget`；
- `evals/dataset.jsonl`；
- `tests/`。

## 当前限制怎么讲

不要假装都接好了。可以诚实说：

- 当前外部模型和工具是 adapter；
- 本地流程和 contract 已经完整；
- 生产接入时替换 provider，不改 Agent Core 边界。

