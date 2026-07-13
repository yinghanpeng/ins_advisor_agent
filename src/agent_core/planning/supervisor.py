"""Supervisor adapter."""

# 文件说明：
# - 本文件属于 Planner / Executor / Verifier / Supervisor 层。
# - 它为复杂任务规划和监督预留生产级扩展点。
from __future__ import annotations


def supervise(status: str) -> str:
    """依据执行状态选择恢复分支或正常继续分支。"""
    # 只有明确的 error 状态进入恢复链路，其余状态继续，避免误触发重试。
    return "recover" if status == "error" else "continue"
