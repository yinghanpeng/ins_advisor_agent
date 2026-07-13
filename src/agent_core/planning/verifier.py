"""Verifier adapter."""

# 文件说明：
# - 本文件属于 Planner / Executor / Verifier / Supervisor 层。
# - 它为复杂任务规划和监督预留生产级扩展点。
from __future__ import annotations


def verify_result(result: dict) -> dict:
    """校验执行状态，并保留原始结果供调用方追踪。"""
    # success 与 planned_only 都属于当前适配器认可的完成状态，其他状态统一视为无效。
    return {"valid": result.get("status") in {"success", "planned_only"}, "result": result}
