# capability_router 能力路由节点

职责：把意图分发到通用能力层或业务 Skill 层。

它本身不直接调用工具，只决定请求应该进入：

- `General Capability Layer`
- `Domain Skill Layer`
- `Human Approval`
- `Recovery`

