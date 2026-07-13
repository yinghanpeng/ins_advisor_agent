"""Executor adapter."""

# 文件说明：
# - 本文件属于 Planner / Executor / Verifier / Supervisor 层。
# - 它为复杂任务规划和监督预留生产级扩展点。
from __future__ import annotations


def execute_plan(steps: list[str]) -> dict:
    """把规划步骤封装成统一的“仅规划”执行结果。"""
    # 此适配器当前不产生外部副作用，只原样回传步骤并标记 planned_only 供后续校验。
    return {"steps": steps, "status": "planned_only"}
