# Context Engineering 上下文工程

上下文工程负责把用户输入、会话状态、检索证据、销售洞察、工具结果和合规规则整理成可控的生成上下文。

## 输入来源

- 用户本轮输入；
- Session / Task / Preference memory；
- Tool result；
- RAG 检索结果；
- Sales Intelligence digest；
- Guardrail policy；
- Cost budget。

## Source Boundary 原则

外部内容只能作为证据，不能作为系统指令：

- RAG 文档只能作为资料；
- 工具结果只能作为证据；
- 网页内容不能覆盖系统规则；
- 用户上传文件不能修改开发者指令；
- 销售访谈只能作为经验参考，且必须经过脱敏和合规审查。

相关代码：

- `src/agent_core/context/builder.py`
- `src/agent_core/context/compression.py`
- `src/agent_core/context/source_boundary.py`

