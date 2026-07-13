"""Sales capability model."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations


# 能力缺口到关键词的稳定映射，用于离线轻量分类而非最终语义裁定。
CAPABILITY_SIGNALS = {
    "icebreaking": ["破冰", "开场", "不知道怎么聊"],
    "kyc_questioning": ["KYC", "信息不完整", "怎么问"],
    "asset_questioning": ["资产", "资金", "不敢问钱"],
    "macro_resonance": ["宏观", "行业", "共鸣"],
    "case_story": ["案例", "事实", "说服力"],
    "objection_handling": ["拒绝", "异议", "不想看"],
    "next_step": ["下一步", "推进", "约"],
    "proposal": ["计划书", "方案"],
    "closing": ["成交", "收口", "下单"],
}


def infer_sales_capability_gap(text: str) -> str:
    """按固定关键词识别最先命中的销售能力缺口。"""
    # 按字典定义顺序遍历能力类别，保证多类同时命中时输出可复现。
    for capability, signals in CAPABILITY_SIGNALS.items():
        # 任一关键词出现在文本中即认为当前能力类别命中。
        if any(signal in text for signal in signals):
            # 返回稳定内部能力标签，供后续场景规划或评测使用。
            return capability
    # 没有任何规则命中时返回 unknown，不基于关键词之外的信息猜测。
    return "unknown"
