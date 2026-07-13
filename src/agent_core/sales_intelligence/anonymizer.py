"""Interview anonymization."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

import re


# 脱敏模式按手机号、邮箱、金额和称谓姓名依次替换，保留占位符以维持业务语义。
PATTERNS = [
    (re.compile(r"1[3-9]\d{9}"), "[PHONE]"),
    (re.compile(r"[\w.-]+@[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\d+(?:\.\d+)?\s*(?:万|万元|亿|亿元)"), "[AMOUNT]"),
    (re.compile(r"([\u4e00-\u9fa5]{2,4})(先生|女士|总)"), "[NAME]\\2"),
]


def anonymize_interview(text: str) -> tuple[str, list[dict]]:
    """逐类遮蔽访谈敏感值，并返回脱敏文本和不含原值的命中日志。"""
    # 日志只累计正则摘要和数量，绝不记录命中的敏感原文。
    logs: list[dict] = []
    # 从输入文本副本开始替换，字符串不可变且不会修改调用方原值。
    masked = text
    # 按固定模式顺序逐类扫描，使每类敏感信息都能独立审计。
    for pattern, replacement in PATTERNS:
        # 重点逻辑：每类敏感信息都记录命中次数，方便后续审计脱敏是否生效。
        matches = pattern.findall(masked)
        # 只有实际命中时才写审计日志和执行替换。
        if matches:
            # 保存模式和命中数量，不保存手机号、邮箱、姓名或金额本身。
            logs.append({"pattern": pattern.pattern, "count": len(matches)})
            # 重点逻辑：替换为占位符而不是删除，尽量保留原文业务语义。
            masked = pattern.sub(replacement, masked)
    # 同时返回可继续清洗/分段的文本与脱敏审计摘要。
    return masked, logs
