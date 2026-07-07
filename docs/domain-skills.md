# Domain Skills 业务技能

Domain Skill 是 Agent Core 之上的垂直业务插件。

## Skill 负责什么

- 业务 workflow；
- 业务 Prompt；
- 业务路由；
- 业务术语；
- 业务合规边界；
- 业务示例和 eval case。

## Skill 不负责什么

- 网关；
- 通用工具；
- 通用 RAG；
- Memory 基础设施；
- Trace；
- Recovery；
- Cost Control。

当前已实现的 Skill：

- `src/agent_core/skills/insurance_advisor/`

未来可以继续增加：

- 研究助手；
- 文档分析助手；
- 面试助手；
- 客服助手；
- 数据分析助手。

