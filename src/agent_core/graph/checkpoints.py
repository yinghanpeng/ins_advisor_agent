"""Checkpoint storage adapters."""

# 文件说明：
# - 本文件属于显式状态机层，负责状态对象、节点函数、边或 checkpoint。
# - 所有复杂任务都应通过状态流转表达，避免把流程藏在大 Prompt 中。
from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.graph.state import AgentState


@dataclass
class InMemoryCheckpointStore:
    """Simple checkpoint store for local development and tests."""

    states: dict[str, AgentState] = field(default_factory=dict)

    def save(self, state: AgentState, checkpoint_id: str | None = None) -> str:
        """保存一次 AgentState 快照，避免后续修改影响已存 checkpoint。"""
        key = checkpoint_id or state.trace_id
        self.states[key] = state.model_copy(deep=True)
        return key

    def get(self, checkpoint_id: str) -> AgentState | None:
        """按 checkpoint_id 读取状态快照；不存在时返回 None。"""
        state = self.states.get(checkpoint_id)
        return state.model_copy(deep=True) if state else None
