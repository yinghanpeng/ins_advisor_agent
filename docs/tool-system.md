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
- `side_effect_level`：`none/read_only` 可执行，`write/external_action/financial` 直接禁止；
- `retryable`：是否可重试；
- `timeout_seconds`：超时时间；
- `idempotency_required`：是否要求幂等；
- `error_schema`：错误结构。

## 调用流程

```text
用户输入
→ ToolRouter 选择工具
→ 根据 Query Understanding 构造最小参数
→ ToolInputValidator 按该工具的 input_schema 校验
   ├─ 缺少必填参数：Clarify，本轮不执行工具
   └─ 参数有效：继续
→ ToolGuardrail 检查权限和副作用（不允许则同步 deny）
→ Executor 在信任边界再次校验 input_schema
→ Tool Adapter 执行
→ Tool Result Verifier 校验结果
→ 写结构化日志和 trace
```

相关代码：

- `src/agent_core/tools/schemas.py`
- `src/agent_core/tools/registry.py`
- `src/agent_core/tools/router.py`
- `src/agent_core/tools/permissions.py`
- `src/agent_core/tools/verifier.py`
- `src/agent_core/guardrails/tool_guardrails.py`

通用链路不再维护独立的全局槽位表。工具参数定义、必填项和类型约束均以
`ToolSpec.input_schema` 为准；KYC 的 `profile_state/missing_fields` 属于另一套领域状态。
保险领域状态的具体 Schema 位于 `skills/insurance_advisor/kyc.py`，只在保险代码处理器内使用。
