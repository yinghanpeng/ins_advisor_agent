"""Feedback schemas."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations

from pydantic import BaseModel, Field


class HumanFeedback(BaseModel):
    """人工反馈记录，用于补充自动 eval 之外的质量判断。"""

    trace_id: str = Field(..., description="被评价的 Agent 运行 trace_id。")
    score: int = Field(..., description="人工评分，建议使用统一量表，例如 1-5 或 1-10。")
    comment: str = Field(
        default="",
        description="人工评价备注，例如回答是否清晰、是否合规、是否贴合销售场景。",
    )
    reviewer: str | None = Field(
        default=None,
        description="评价人标识。匿名反馈可以为空。",
    )
