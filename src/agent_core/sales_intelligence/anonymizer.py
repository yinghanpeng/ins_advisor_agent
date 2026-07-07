"""Interview anonymization."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

import re


PATTERNS = [
    (re.compile(r"1[3-9]\d{9}"), "[PHONE]"),
    (re.compile(r"[\w.-]+@[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\d+(?:\.\d+)?\s*(?:万|万元|亿|亿元)"), "[AMOUNT]"),
    (re.compile(r"([\u4e00-\u9fa5]{2,4})(先生|女士|总)"), "[NAME]\\2"),
]


def anonymize_interview(text: str) -> tuple[str, list[dict]]:
    """Mask sensitive values while preserving business meaning."""
    logs: list[dict] = []
    masked = text
    for pattern, replacement in PATTERNS:
        # 重点逻辑：每类敏感信息都记录命中次数，方便后续审计脱敏是否生效。
        matches = pattern.findall(masked)
        if matches:
            logs.append({"pattern": pattern.pattern, "count": len(matches)})
            # 重点逻辑：替换为占位符而不是删除，尽量保留原文业务语义。
            masked = pattern.sub(replacement, masked)
    return masked, logs
