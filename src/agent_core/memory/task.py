"""Task memory placeholder."""

# 文件说明：
# - 本文件属于 Memory 层，负责 session/task/preference 分层记忆和策略。
# - 生产环境需要替换为带租户隔离的持久化存储。
from dataclasses import dataclass, field


@dataclass
class TaskMemory:
    tasks: dict[str, dict] = field(default_factory=dict)

