"""Segment interviews by sales scene."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

from pydantic import BaseModel, Field


class InterviewSegment(BaseModel):
    """销售访谈分段结果。每段会进入后续洞察抽取和合规审查。"""

    # source_id 继承自 RawInterview，用于从分段追溯到原始访谈。
    source_id: str = Field(..., description="原始访谈来源 ID，用于把分段结果回溯到 RawInterview。")
    # chunk_id 是当前分段的唯一 ID，用于后续洞察卡片和 RAG chunk 溯源。
    chunk_id: str = Field(..., description="当前分段 ID，通常由 source_id 和序号组成。")
    # scene 是这段访谈所属销售场景，检索和抽取都会使用它。
    scene: str = Field(..., description="该分段识别出的销售场景，例如 icebreaking、objection_handling。")
    # text 是该分段的正文，长度应被控制在抽取模型可处理范围内。
    text: str = Field(..., description="分段后的访谈正文。长度应控制在后续抽取模型可处理范围内。")


# SCENE_KEYWORDS 是本地关键词场景分类器；生产可替换为模型分类，但输出 scene 名称保持一致。
SCENE_KEYWORDS = {
    # KYC 深挖关注家庭、资产、收入、顾虑、真实需求等关键词。
    "kyc_deep_dive": ["KYC", "家庭", "资产", "收入", "顾虑", "真实需求"],
    # 破冰场景关注开场、闲聊、饭局等关键词。
    "icebreaking": ["破冰", "开场", "闲聊", "饭局"],
    # 宏观共鸣关注行业、经济、宏观环境等关键词。
    "macro_resonance": ["宏观", "行业", "经济", "共鸣"],
    # 案例证据关注案例、事实、数据等关键词。
    "case_evidence": ["案例", "事实", "数据", "身边"],
    # 异议处理关注拒绝、不想、观望等关键词。
    "objection_handling": ["拒绝", "异议", "不想", "观望"],
    # 计划书/成交收口关注计划书、成交、加保等关键词。
    "proposal_closing": ["计划书", "成交", "收口", "加保"],
}


def classify_scene(text: str) -> str:
    """用关键词快速判断访谈片段所属销售场景；生产环境可替换为模型分类器。"""
    # 遍历所有场景关键词，只要命中一个关键词就返回对应 scene。
    for scene, keywords in SCENE_KEYWORDS.items():
        # 本地版本使用关键词分类，生产可替换为模型分类器。
        if any(keyword in text for keyword in keywords):
            # 返回首个命中的稳定场景标签，保证规则分类结果可复现。
            return scene
    # 没有命中任何关键词时返回 unknown，避免随意猜测销售场景。
    return "unknown"


def segment_by_scene(source_id: str, text: str, max_chars: int = 1000) -> list[InterviewSegment]:
    """按固定长度切分访谈文本，并为每段补充销售场景标签。"""
    # segments 收集切分后的 InterviewSegment。
    segments: list[InterviewSegment] = []
    # 按 max_chars 固定窗口切分原始访谈，index 从 1 开始便于生成可读 chunk_id。
    for index, start in enumerate(range(0, len(text), max_chars), 1):
        # 先按长度切片，保证后续 LLM 抽取不会一次吃进过长 transcript。
        chunk = text[start : start + max_chars]
        # 空白片段不进入后续抽取。
        if chunk.strip():
            # 为每个有效片段创建结构化 Segment，并补充 scene 标签。
            segments.append(
                InterviewSegment(
                    source_id=source_id,
                    chunk_id=f"{source_id}_chunk_{index:03d}",
                    scene=classify_scene(chunk),
                    text=chunk,
                )
            )
    # 返回所有分段，后续 extractor 会逐段生成 SalesInsightCard。
    return segments
