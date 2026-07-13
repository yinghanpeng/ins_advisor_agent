"""LangSmith eval runner adapter."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations


def run_langsmith_eval(_: list[dict]) -> dict:
    """在未配置远程凭据的本地实现中返回明确的跳过状态。"""

    # 不伪造远程评测结果；结构化说明跳过原因，便于 CI 区分 skipped 与 passed。
    return {
        "status": "skipped",
        "reason": "LangSmith remote eval requires API key and network access.",
    }
