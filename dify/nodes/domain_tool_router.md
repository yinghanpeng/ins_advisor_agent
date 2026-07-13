# domain_tool_router 业务工具路由节点

职责：选择业务 Skill 内部需要的工具或服务。

保险顾问 Skill 不应该直接访问原始访谈，而应该通过 Sales Intelligence Layer 的统一接口读取已通过
静态生成准入的经验。该说明是 Agent Core 内部职责映射，当前 Dify 推荐流程不执行本节点。
