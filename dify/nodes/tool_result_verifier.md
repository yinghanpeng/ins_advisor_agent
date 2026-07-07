# tool_result_verifier 工具结果校验节点

职责：工具调用完成后，先校验结果，再允许进入生成节点。

校验内容：

- output schema；
- 来源；
- 错误结构；
- latency；
- retry count；
- 是否需要降级；
- 是否需要人工审批。

