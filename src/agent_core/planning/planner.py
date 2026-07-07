"""Planner adapter."""

# 文件说明：
# - 本文件属于 Planner / Executor / Verifier / Supervisor 层。
# - 它为复杂任务规划和监督预留生产级扩展点。
from __future__ import annotations


def plan_task(intent: str) -> list[str]:
    if intent == "insurance_advisor_help":
        return ["retrieve_sales_intelligence", "build_context", "generate_response", "review"]
    return ["route_tool", "verify", "respond"]

