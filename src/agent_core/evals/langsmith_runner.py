"""LangSmith eval runner adapter."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations


def run_langsmith_eval(_: list[dict]) -> dict:
    return {
        "status": "skipped",
        "reason": "LangSmith remote eval requires API key and network access.",
    }

