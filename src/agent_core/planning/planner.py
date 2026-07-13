"""Planner adapter."""

# 文件说明：
# - 本文件属于 Planner / Executor / Verifier / Supervisor 层。
# - 它为复杂任务规划和监督预留生产级扩展点。
from __future__ import annotations


def plan_task(intent: str) -> list[str]:
    """把白名单意图映射为静态高层步骤，供演示与离线规划使用。"""
    # 保险细分意图统一进入代码化 KYC/双知识库/策略处理器，不再返回外部 Workflow 名。
    if intent in {
        "insurance_break_ice",
        "insurance_objection_handling",
        "insurance_strategy",
        "insurance_kyc_collection",
    }:
        # 保险步骤只描述 Handler 内部阶段，不创建独立 Workflow 或多 Agent Handoff。
        return ["extract_insurance_kyc", "route_insurance_state", "retrieve_if_ready", "review"]
    # 非保险意图使用通用工具规划、结果校验和回答三步摘要。
    return ["route_tool", "verify", "respond"]
