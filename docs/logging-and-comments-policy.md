# 注释与日志规范

本项目要求代码不仅能运行，还要能解释、能追踪、能排错。

## 注释规范

- 每个 Python 文件必须有 module docstring，说明文件职责。
- 公开类和关键函数必须有 docstring。
- 每个 `if/elif/for/while/try/with/raise/match` 必须有紧邻中文注释，精确说明触发条件、输入输出、
  风险边界或失败降级；只写文件头或函数总述不能替代分支级解释。
- 每一条运行时赋值、独立函数/Provider/数据库/模型调用、`return/assert/break/continue/pass` 都必须在
  相邻行解释“为什么执行”和“下游消费什么”，不能只复述 Python 语法。
- `else/except/finally` 也必须在分支边界前写中文说明，异常路径不能只依赖 `try` 上方的总述。
- import、括号、纯数据字面量和已经具备 `Field(description=...)` 的简单契约字段不机械增加噪声注释。
- Pydantic 契约字段必须提供 `Field(description=...)`；`tests/test_documentation_quality.py` 会自动约束核心模型。
- `tests/test_documentation_quality.py` 使用 AST 扫描全部 `src/agent_core` 和 `scripts` 生产代码：既检查
  控制流，也检查赋值、调用、返回、分支边界、模块说明和函数 docstring；失败信息返回精确文件与行号。
- 意图相似度、执行置信度、TTL、TopK、KYC 轮次、Provider 和模型端点不得散落硬编码，必须从 `configs/` 与环境变量读取。
- mock、adapter、placeholder 必须明确标注，不能伪装成真实生产实现。

## 日志规范

本地结构化日志始终可用，LangSmith 只是增强层。

每次请求至少记录：

- `trace_id`
- `session_id`
- `tenant_id`
- `workflow_name`（仅兼容标签）
- `domain_skill`
- `intent`
- `intent_routing_result` 的来源、分数、置信度和动作；
- `active_intent_action`，但不记录 KYC 原值；
- `current_state`
- `final_state`

每次状态迁移记录：

- `from_state`
- `to_state`
- `reason`
- `metadata`
- `ts`

工具、RAG、Guardrail、Sales Intelligence 应记录：

- 输入摘要；
- 输出摘要；
- 选择的工具或卡片；
- score / rerank score；
- risk level；
- retry count；
- error；
- latency。

## 当前实现

- `AgentState.move_to()` 写入 `state_transitions` 和 `trace_events`。
- `WorkflowEngine` 将 state transitions 和 trace events 写入 `StructuredLogger`。
- `LangSmithAdapter` 将一次请求和真实状态节点写成远程 Run Tree；无 API Key 或不可用时降级，不影响主业务。
- 客户请求没有审批日志或等待状态；高风险动作只记录同步 allow/mask/safe_fallback/block/deny 决策。

## 后续扩展

- 将 `StructuredLogger` 输出接入 JSON log 文件、OpenTelemetry 或云日志。
- 给每个 tool adapter 增加 latency 和 error 统计。
- 使用 LangSmith Dataset/Experiment 自动执行回归评估并对比 Prompt、模型和路由版本。
