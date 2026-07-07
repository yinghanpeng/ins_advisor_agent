"""Executor adapter."""

# 文件说明：
# - 本文件属于 Planner / Executor / Verifier / Supervisor 层。
# - 它为复杂任务规划和监督预留生产级扩展点。
from __future__ import annotations


def execute_plan(steps: list[str]) -> dict:
    return {"steps": steps, "status": "planned_only"}

