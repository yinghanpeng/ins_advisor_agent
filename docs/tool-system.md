# Tool System 工具系统

工具不能只是“模型想调就调”。每个工具都必须有清晰契约、权限和日志。

## ToolSpec 字段

- `name`：工具名；
- `description`：工具说明；
- `input_schema`：输入 schema；
- `output_schema`：输出 schema；
- `risk_level`：风险等级；
- `permission`：权限等级和 scope；
- `side_effect`：是否有副作用；
- `requires_approval`：是否需要人工审批；
- `retryable`：是否可重试；
- `timeout_seconds`：超时时间；
- `idempotency_required`：是否要求幂等；
- `error_schema`：错误结构。

## 调用流程

```text
用户输入
→ ToolRouter 选择工具
→ ToolGuardrail 检查权限
→ Tool Adapter 执行
→ Tool Result Verifier 校验结果
→ 写结构化日志和 trace
```

相关代码：

- `src/agent_core/tools/schemas.py`
- `src/agent_core/tools/registry.py`
- `src/agent_core/tools/router.py`
- `src/agent_core/tools/permissions.py`
- `src/agent_core/guardrails/tool_guardrails.py`

