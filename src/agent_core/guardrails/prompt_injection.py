"""Prompt injection detection."""

# 文件说明：
# - 本文件属于 Guardrails 层，负责输入安全、工具权限、输出合规或人工审批。
# - 保险金融场景必须拦截收益承诺、避税避债、恐吓营销和编造案例。
from __future__ import annotations


SUSPICIOUS_PATTERNS = [
    "ignore previous",
    "忽略以上",
    "忽略之前",
    "system prompt",
    "开发者指令",
    "越权",
]


def detect_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in SUSPICIOUS_PATTERNS)

