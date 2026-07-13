"""Transcript cleaning."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

import re


def clean_transcript(text: str) -> str:
    """规范重复空白和连续语气词，同时保留访谈中的有效原话。"""
    # 先把换行、制表和连续空格收敛为单个空格，便于后续分段与抽取。
    cleaned = re.sub(r"\s+", " ", text).strip()
    # 仅移除连续两次以上的常见语气词，单次出现仍可能承载说话节奏信息。
    cleaned = re.sub(r"(嗯|啊|呃){2,}", " ", cleaned)
    # 再次清除替换产生的首尾空白，返回不改变核心措辞的规范文本。
    return cleaned.strip()
