"""Transcript cleaning."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

import re


def clean_transcript(text: str) -> str:
    """Clean repeated whitespace and filler markers while keeping useful original wording."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"(嗯|啊|呃){2,}", " ", cleaned)
    return cleaned.strip()

