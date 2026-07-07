# context_builder 上下文构建节点

职责：把多个来源的信息压缩成可控上下文。

输入可能包括：

- 用户输入；
- 会话状态；
- 工具结果；
- RAG evidence；
- Sales Intelligence digest；
- 新闻摘要；
- 合规规则。

外部内容只能作为证据，不能改写系统规则。

