# Memory 记忆层

Memory 不是简单把所有聊天历史塞进 Prompt，而是按用途分层。

## 三层 Memory

- Session Memory：当前会话状态，例如客户画像、KYC 阶段、已问过的问题；
- Task Memory：当前任务状态，例如 workflow step、工具结果、恢复信息；
- Preference Memory：长期偏好，例如用户喜欢低压话术、输出格式偏好。

## 当前实现

- `src/agent_core/memory/session.py`
- `src/agent_core/memory/task.py`
- `src/agent_core/memory/preference.py`
- `src/agent_core/memory/manager.py`
- `src/agent_core/memory/policy.py`

`MemoryManager` 提供统一入口，并带有 audit log。

## 生产要求

- 默认不保存敏感个人信息；
- 必须有租户隔离；
- 需要数据保留周期；
- 需要删除和导出机制。

