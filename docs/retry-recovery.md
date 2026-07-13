# Retry / Recovery 重试与恢复

生产 Agent 必须能处理失败，而不是失败后直接崩溃或胡编答案。

## 当前实现

- `src/agent_core/recovery/retry.py`：通用 retry helper；
- `src/agent_core/recovery/fallback.py`：降级回答和 `RecoveryPlan`；
- `src/agent_core/recovery/json_repair.py`：JSON 提取和修复辅助。

## 典型失败处理

- 工具不可用：记录错误，返回降级建议；
- 检索为空：不编造证据，说明当前没有可靠来源；
- JSON 格式错误：尝试 repair，再做 schema 校验；
- 成本超预算：压缩上下文、降低 top-k、跳过可选工具；
- 高风险输出：阻断原内容，同步替换为安全说明。

## 后续扩展

把 `RecoveryPlan` 接入 `AgentGraph` 各节点的异常处理路径，并保持所有降级都是同步、有界且可审计的。
