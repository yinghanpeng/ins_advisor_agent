# Agentic 工具迭代循环

> 当前状态：实验代码保留，但已从 `_run_universal` 主链路移除。当前工具请求使用单轮
> `general_tool_routing → general_tool_call → verify_tool_result`。

本文保留有界工具循环的实验设计。它不是黑盒 ReAct，而是在显式状态机中把单轮工具链包装成可审计、可停止、可降级的工具迭代；当前生产主链路不执行该包装。

## 为什么单次工具调用不足

原链路只执行一次工具计划，适合计算器、天气等简单确定性任务，但不适合这些场景：

- 搜索、新闻、多步研究需要先查 A，再根据 observation 决定是否查 B；
- 文件解析、网页读取可能先拿目录、再拿具体页面；
- 工具失败后需要明确降级，而不是让生成节点误以为已经拿到证据；
- 工具结果可能包含外部指令或 PII，必须每轮重新做边界标注和风控；
- 多轮工具调用必须有预算，否则容易形成无限 loop。

## 实验链路（当前未接入）

实验调用方可以在 `context_needs.tool=true` 后显式进入：

```text
agentic_tool_loop
  loop up to max_tool_iterations:
    plan_next_tool_or_finish
    tool_guardrail
    execute_tool
    observe_tool_result
    verify_tool_result
    decide_continue_or_finish
```

当前实现仍复用旧节点：

- `general_tool_routing`
- `general_tool_call`
- `verify_tool_result`

这些节点和单元测试仍然保留，但当前工具调用保持“单次执行”，没有升级为主链路循环。

## 如何避免无限循环

循环预算来自 `ToolLoopConfig`：

- `max_iterations`：默认 4；
- `max_tool_calls_per_iteration`：默认 2；
- `max_total_tool_calls`：默认 6；
- `stop_on_tool_error`：默认 false；
- `fallback_to_rule_router`：默认 true。

停止条件包括：

- 不需要工具：`tool_loop_stop_reason=no_tool_needed`；
- planner 判断完成：`finished`；
- 达到最大轮次：`max_iterations`；
- 连续两轮计划完全相同：`repeated_tool_plan`；
- 工具错误超过预算：`tool_error_budget_exceeded`；
- 工具被权限或副作用策略拒绝：记录 `blocked` 结果并同步降级；
- planner 请求澄清：`ask_clarification`。

每轮计划会生成稳定 fingerprint。若连续两轮完全相同，系统在执行第二轮前停止，避免“查同一个东西，得到同一个结果，再查同一个东西”的循环风险。

## Fast Path 和 Loop Path

适合 fast path 的任务：

- calculator；
- time/date；
- 单次 weather 查询；
- 简单文本 summarizer。

适合进入 agentic loop 的任务：

- search / news_search；
- 需要查询近期事实的任务；
- 需要先查实体再查细节的任务；
- 文件解析后再总结；
- corrective/self-RAG：证据不足时改写 query 再检索一次。

第一版本地实现以规则 planner 为主。`ModelToolLoopPlanner` 是可选接口壳，模型不可用时只记录降级事件，绝不伪造外部事实或工具结果。

## 工具 Guardrail

每次工具执行仍经过：

- 工具注册表白名单；
- `ToolGuardrail` 权限检查；
- permission scope；
- `side_effect_level`（只允许 `none/read_only`）；
- 工具结果清洗；
- `_source_boundary` 标注；
- `verify_tool_result` 结构校验。

工具结果只能作为 data 进入上下文，不能作为 instruction 覆盖系统规则。

## 预留路线

多 Agent / handoff 暂不在本次代码中实现。未来可以基于 `domain_skill` 扩展：

- `orchestrator`
- `research_worker`
- `insurance_advisor_worker`
- `data_analysis_worker`

缓存也只做文档占位，后续可扩展：

- prompt cache；
- semantic cache；
- tool result cache。

不强接 Redis。需要分布式缓存时，应先接入统一租户隔离和审计封装。
