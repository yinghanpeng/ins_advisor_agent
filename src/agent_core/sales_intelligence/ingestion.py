"""Raw interview ingestion."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent_core.utils.ids import new_id
from agent_core.utils.time import utc_now_iso


class RawInterview(BaseModel):
    """原始销售访谈记录。该对象只负责承载原文，不能直接进入最终回答生成。"""

    source_id: str = Field(
        default_factory=lambda: new_id("interview"),
        description="原始访谈来源 ID。用于把后续 segment、洞察卡片和审计记录串回原文。",
    )
    text: str = Field(
        ...,
        description="访谈原文或转写文本。后续会经过清洗、脱敏、分段和结构化抽取。",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="访谈来源扩展信息，例如文件路径、采访对象、渠道、日期或上传人。",
    )
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="访谈导入时间，ISO 字符串，用于审计和增量索引。",
    )


def ingest_raw_interview(text: str, metadata: dict | None = None) -> RawInterview:
    """创建原始访谈记录；持久化由调用方或 indexer 负责。"""
    return RawInterview(text=text, metadata=metadata or {})


def ingest_text_file(path: str | Path, metadata: dict | None = None) -> RawInterview:
    """从本地文本文件导入访谈内容，并把文件路径写入 metadata 方便追溯。"""
    file_path = Path(path)
    return ingest_raw_interview(file_path.read_text(encoding="utf-8"), metadata or {"path": str(path)})
