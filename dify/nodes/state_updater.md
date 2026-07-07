# state_updater 状态更新节点

职责：在 Dify 侧保存轻量会话信息和 `trace_id`。

注意：长期状态、checkpoint、memory、恢复逻辑都由 Agent Core 管理，Dify 只做 Control Plane。

