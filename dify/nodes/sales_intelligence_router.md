# sales_intelligence_router 销售智能路由节点

职责：判断业务请求是否需要检索 Sales Intelligence Layer。

重要规则：

- 只能检索已通过静态生成准入的卡片；
- 不能直接检索原始访谈；
- 高风险卡片不能进入生成；
- 检索结果必须压缩为 digest。

该说明是 Agent Core 内部职责映射，当前 Dify 推荐流程不执行本节点。
