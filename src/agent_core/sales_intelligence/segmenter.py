"""Segment interviews by sales scene."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from pydantic import BaseModel, Field


class InterviewSegment(BaseModel):
    """销售访谈分段结果。每段会进入后续洞察抽取和合规审查。"""

    source_id: str = Field(..., description="原始访谈来源 ID，用于把分段结果回溯到 RawInterview。")
    chunk_id: str = Field(..., description="当前分段 ID，通常由 source_id 和序号组成。")
    scene: str = Field(..., description="该分段识别出的销售场景，例如 icebreaking、objection_handling。")
    text: str = Field(..., description="分段后的访谈正文。长度应控制在后续抽取模型可处理范围内。")


SCENE_KEYWORDS = {
    "kyc_deep_dive": ["KYC", "家庭", "资产", "收入", "顾虑", "真实需求"],
    "icebreaking": ["破冰", "开场", "闲聊", "饭局"],
    "macro_resonance": ["宏观", "行业", "经济", "共鸣"],
    "case_evidence": ["案例", "事实", "数据", "身边"],
    "objection_handling": ["拒绝", "异议", "不想", "观望"],
    "proposal_closing": ["计划书", "成交", "收口", "加保"],
}


def classify_scene(text: str) -> str:
    """用关键词快速判断访谈片段所属销售场景；生产环境可替换为模型分类器。"""
    for scene, keywords in SCENE_KEYWORDS.items():
        # 重点逻辑：本地版本使用关键词分类，生产可替换为模型分类器。
        if any(keyword in text for keyword in keywords):
            return scene
    return "unknown"


def segment_by_scene(source_id: str, text: str, max_chars: int = 1000) -> list[InterviewSegment]:
    """按固定长度切分访谈文本，并为每段补充销售场景标签。"""
    segments: list[InterviewSegment] = []
    for index, start in enumerate(range(0, len(text), max_chars), 1):
        # 重点逻辑：先按长度切片，保证后续 LLM 抽取不会一次吃进过长 transcript。
        chunk = text[start : start + max_chars]
        if chunk.strip():
            segments.append(
                InterviewSegment(
                    source_id=source_id,
                    chunk_id=f"{source_id}_chunk_{index:03d}",
                    scene=classify_scene(chunk),
                    text=chunk,
                )
            )
    return segments
