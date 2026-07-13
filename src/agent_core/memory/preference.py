"""偏好记忆容器。"""

# 文件说明：
# - 本文件属于 Memory 层，负责 session/task/preference 分层记忆和策略。
# - 生产环境通过 MemoryManager 或 PostgreSQL repository 统一做租户隔离。
from dataclasses import dataclass, field


@dataclass
class PreferenceMemory:
    """保存跨 Session 复用的低风险用户交互偏好映射。"""

    # preferences 以租户主体复合 Key 索引结构化偏好字段。
    preferences: dict[str, dict] = field(default_factory=dict)
