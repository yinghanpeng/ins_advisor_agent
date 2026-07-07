"""Layered memory manager.

The manager gives Agent Core one boundary for session, task, and preference
memory. Production storage can replace these in-memory maps without changing
workflow nodes.
"""

# 文件说明：
# - 本文件属于 Memory 层，负责 session/task/preference 分层记忆和策略。
# - 生产环境需要替换为带租户隔离的持久化存储。
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_core.memory.preference import PreferenceMemory
from agent_core.memory.session import SessionMemory
from agent_core.memory.task import TaskMemory
from agent_core.utils.time import utc_now_iso


class MemoryLayer(StrEnum):
    # SESSION 只在当前会话内有效，保存 recent_messages、last_entity、last_intent 等短期上下文。
    SESSION = "session"
    # TASK 保存当前任务进度，例如是否已经生成答案、当前状态机节点是什么。
    TASK = "task"
    # PREFERENCE 保存跨会话可复用的长期偏好或画像候选，需要比 session 更谨慎写入。
    PREFERENCE = "preference"


@dataclass
class MemoryManager:
    """统一管理 session、task、preference 三层记忆，并记录访问审计。"""

    # session 记忆存储同一 session 内的短期对话上下文。
    session: SessionMemory = field(default_factory=SessionMemory)
    # task 记忆存储当前任务级状态，适合恢复未完成 workflow。
    task: TaskMemory = field(default_factory=TaskMemory)
    # preference 记忆存储跨 session 的稳定偏好或画像候选。
    preference: PreferenceMemory = field(default_factory=PreferenceMemory)
    # audit_log 记录每次 read/write 的层级、key 和字段，便于本地审计。
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def _key(self, tenant_id: str, subject_id: str) -> str:
        """把租户和主体 ID 合成隔离后的存储 key。"""
        # key 中强制包含 tenant_id，避免不同租户使用相同 session_id/user_id 时串数据。
        return f"{tenant_id}:{subject_id}"

    def read(self, layer: MemoryLayer, tenant_id: str, subject_id: str) -> dict[str, Any]:
        """读取指定记忆层，并把读取行为写入 audit_log。"""
        # 先合成租户隔离 key，后续所有 memory backend 都以它为索引。
        key = self._key(tenant_id, subject_id)
        # SESSION 层通过 SessionMemory.get 读取，不存在时由底层返回空字典。
        if layer == MemoryLayer.SESSION:
            value = self.session.get(key)
        # TASK 层用 setdefault 保证首次读取时自动创建空任务状态。
        elif layer == MemoryLayer.TASK:
            value = self.task.tasks.setdefault(key, {})
        # PREFERENCE 层同样首次读取自动创建空偏好对象，方便后续 update。
        else:
            value = self.preference.preferences.setdefault(key, {})
        # 每次读取都写审计日志，生产替换持久化存储时也应保留这类访问记录。
        self.audit_log.append({"ts": utc_now_iso(), "action": "read", "layer": layer.value, "key": key})
        # 返回可变 dict；workflow 节点可以读取，也可以通过 write 显式更新。
        return value

    def write(
        self,
        layer: MemoryLayer,
        tenant_id: str,
        subject_id: str,
        values: dict[str, Any],
    ) -> None:
        """向指定记忆层写入字段，并记录本次写入涉及哪些 key。"""
        # 生成审计用 key，确保日志能定位写入的是哪个租户和主体。
        key = self._key(tenant_id, subject_id)
        # 通过 read 获取目标层；这样首次写入时也能自动初始化对应 dict。
        target = self.read(layer, tenant_id, subject_id)
        # 合并写入字段而不是整体覆盖，避免一个节点写入时抹掉其他节点的记忆。
        target.update(values)
        # 写入审计只记录字段名，不记录完整内容，降低敏感记忆泄露风险。
        self.audit_log.append(
            {
                "ts": utc_now_iso(),
                "action": "write",
                "layer": layer.value,
                "key": key,
                "fields": sorted(values.keys()),
            }
        )
