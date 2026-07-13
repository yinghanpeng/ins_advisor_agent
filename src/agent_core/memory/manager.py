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
from typing import Any, Protocol

from agent_core.memory.preference import PreferenceMemory
from agent_core.memory.session import SessionMemory
from agent_core.memory.task import TaskMemory
from agent_core.utils.time import utc_now_iso


class MemoryLayer(StrEnum):
    """统一标识短期 Session、任务状态和长期 Preference 三层记忆。"""

    # SESSION 只在当前会话内有效，保存 recent_messages、last_entity、last_intent 等短期上下文。
    SESSION = "session"
    # TASK 保存当前任务进度，例如是否已经生成答案、当前状态机节点是什么。
    TASK = "task"
    # PREFERENCE 保存跨会话可复用的长期偏好或画像候选，需要比 session 更谨慎写入。
    PREFERENCE = "preference"


class MemoryBackend(Protocol):
    """记忆读写协议；生产 Redis/PostgreSQL 与测试 Store 必须遵循同一边界。"""

    def read(self, layer: MemoryLayer, tenant_id: str, subject_id: str) -> dict[str, Any]:
        """读取指定租户、主体和层级的记忆快照。"""

    def write(
        self,
        layer: MemoryLayer,
        tenant_id: str,
        subject_id: str,
        values: dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> int:
        """原子合并记忆并返回新版本；expected_version 用于 CAS。"""

    def export_subject(self, tenant_id: str, subject_id: str) -> dict[str, Any]:
        """导出指定主体可访问的记忆，供隐私请求和审计使用。"""

    def delete_subject(self, tenant_id: str, subject_id: str) -> int:
        """删除指定主体记忆并返回删除记录数。"""


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
            # 读取或初始化会话层状态。
            value = self.session.get(key)
        # TASK 层用 setdefault 保证首次读取时自动创建空任务状态。
        elif layer == MemoryLayer.TASK:
            # 读取或初始化任务层状态。
            value = self.task.tasks.setdefault(key, {})
        # PREFERENCE 层同样首次读取自动创建空偏好对象，方便后续 update。
        else:
            # 读取或初始化长期偏好层状态。
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
        *,
        expected_version: int | None = None,
    ) -> int:
        """向指定记忆层写入字段，并记录本次写入涉及哪些 key。"""
        # 生成审计用 key，确保日志能定位写入的是哪个租户和主体。
        key = self._key(tenant_id, subject_id)
        # 通过 read 获取目标层；这样首次写入时也能自动初始化对应 dict。
        target = self.read(layer, tenant_id, subject_id)
        # 测试 Store 也维护单调递增版本，确保节点测试可覆盖生产 CAS 语义。
        current_version = int(target.get("_version", 0))
        # expected_version 不匹配时拒绝覆盖，模拟 Redis/PostgreSQL 的乐观锁冲突。
        if expected_version is not None and expected_version != current_version:
            # 抛出冲突而不是覆盖当前值，调用方可以读取新版本后重新合并。
            raise RuntimeError(
                f"memory version conflict: expected={expected_version}, actual={current_version}"
            )
        # _trace_id 是审计关联字段，不属于业务记忆；测试 Store 与生产 Store 都不持久化它。
        persisted_values = {key: value for key, value in values.items() if key != "_trace_id"}
        # 合并写入字段而不是整体覆盖，避免一个节点写入时抹掉其他节点的记忆。
        target.update(persisted_values)
        # 新版本写回测试 Store；读取节点可以忽略该内部字段。
        target["_version"] = current_version + 1
        # 写入审计只记录字段名，不记录完整内容，降低敏感记忆泄露风险。
        self.audit_log.append(
            {
                "ts": utc_now_iso(),
                "action": "write",
                "layer": layer.value,
                "key": key,
                "fields": sorted(persisted_values.keys()),
                "version": current_version + 1,
            }
        )
        # 返回新版本，生产调用方可把它作为下一次 CAS 的 expected_version。
        return current_version + 1

    def export_subject(self, tenant_id: str, subject_id: str) -> dict[str, Any]:
        """导出测试 Store 中指定主体的三层记忆。"""
        # 每层都使用同一个租户隔离 key；不存在时返回空对象。
        return {
            layer.value: dict(self.read(layer, tenant_id, subject_id))
            for layer in MemoryLayer
        }

    def delete_subject(self, tenant_id: str, subject_id: str) -> int:
        """删除测试 Store 中指定主体的全部记忆。"""
        # 测试实现按 tenant:subject 精确删除，不使用模糊扫描。
        key = self._key(tenant_id, subject_id)
        # deleted 累计实际存在并删除的记忆层数量。
        deleted = 0
        # 依次处理三个独立内存映射，累计真实存在并删除的层数。
        for mapping in [self.session.data, self.task.tasks, self.preference.preferences]:
            # 当前层包含精确租户主体 Key 时才执行删除并增加计数。
            if key in mapping:
                # 从当前层映射删除精确复合 Key。
                del mapping[key]
                # 累加一个成功删除层。
                deleted += 1
        # 记录不含记忆正文的删除审计摘要。
        self.audit_log.append(
            {"ts": utc_now_iso(), "action": "delete", "key": key, "deleted": deleted}
        )
        # 返回三个层级实际删除的映射数量。
        return deleted
