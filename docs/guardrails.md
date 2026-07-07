# Guardrails 安全与合规

Guardrails 用于约束输入、工具、语料和输出。

## 分层设计

- 输入层：Prompt Injection、越权请求、敏感信息、高风险意图；
- 销售语料层：原始访谈不能当系统指令，高风险话术不能直接生成；
- 工具层：权限、schema、审批、idempotency；
- 输出层：保险/金融合规、事实引用、敏感信息泄露；
- Source Boundary：外部内容只能当证据。

## 保险销售高风险表达

输出中禁止：

- 保证收益；
- 绝对安全；
- 避债避税；
- 恐吓营销；
- 编造客户故事；
- 贬低其他金融产品。

相关代码：

- `src/agent_core/guardrails/input.py`
- `src/agent_core/guardrails/output.py`
- `src/agent_core/guardrails/tool_guardrails.py`
- `src/agent_core/guardrails/human_approval.py`

