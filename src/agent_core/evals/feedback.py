"""Feedback schemas."""

# 文件说明：
# - 本文件属于评估层，负责 eval 数据集转换、评估器或反馈结构。
# - 评估应覆盖正常任务、失败恢复、安全合规和销售业务质量。
from __future__ import annotations

from pydantic import BaseModel, Field


class HumanFeedback(BaseModel):
    """人工反馈记录，用于补充自动 eval 之外的质量判断。"""

    # trace_id 将人工评价关联到唯一 Agent 运行，便于回放当时的完整链路。
    trace_id: str = Field(..., description="被评价的 Agent 运行 trace_id。")
    # score 保存评价量表上的整数分，由上层产品统一约定具体区间。
    score: int = Field(..., description="人工评分，建议使用统一量表，例如 1-5 或 1-10。")
    # comment 保存评分背后的定性原因，缺省为空以兼容只打分场景。
    comment: str = Field(
        default="",
        description="人工评价备注，例如回答是否清晰、是否合规、是否贴合销售场景。",
    )
    # reviewer 是可选评价人标识，匿名反馈无需填充。
    reviewer: str | None = Field(
        default=None,
        description="评价人标识。匿名反馈可以为空。",
    )
