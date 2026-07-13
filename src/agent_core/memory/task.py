"""任务记忆容器。"""

# 文件说明：
# - 本文件属于 Memory 层，负责 session/task/preference 分层记忆和策略。
# - 生产环境通过 MemoryManager 或 PostgreSQL repository 统一做租户隔离。
from dataclasses import dataclass, field


@dataclass
class TaskMemory:
    """保存当前任务进度及可恢复工作流状态的内存映射。"""

    # tasks 以租户 Session 复合 Key 索引任务级状态。
    tasks: dict[str, dict] = field(default_factory=dict)
