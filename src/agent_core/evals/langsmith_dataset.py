"""LangSmith dataset adapter."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations

from agent_core.workflow.contracts import EvalCase


def to_langsmith_examples(cases: list[EvalCase]) -> list[dict]:
    """把内部 EvalCase 列表转换为 LangSmith examples 的输入输出结构。"""

    # inputs 只暴露待测用户输入，outputs 保留完整期望约束用于各类 evaluator。
    return [{"inputs": {"input": case.input}, "outputs": case.model_dump()} for case in cases]
