"""Session memory."""

# 文件说明：
# - 本文件属于 Memory 层，负责 session/task/preference 分层记忆和策略。
# - 生产环境需要替换为带租户隔离的持久化存储。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMemory:
    """会话级短期记忆，保存同一 session 内需要延续的信息。"""

    # data 以租户化 Session Key 索引当前会话的可变状态映射。
    data: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, session_id: str) -> dict[str, Any]:
        """按 session_id 读取会话记忆；不存在时创建空字典。"""
        # 首次访问创建空状态，后续访问返回同一 Session 映射供显式更新。
        return self.data.setdefault(session_id, {})

    def update(self, session_id: str, values: dict[str, Any]) -> None:
        """合并更新会话记忆，不覆盖未传入字段。"""
        # 复用 get 的首次初始化语义，并以 merge 方式更新传入字段。
        self.get(session_id).update(values)
