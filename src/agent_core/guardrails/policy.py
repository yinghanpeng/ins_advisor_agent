"""Guardrail policy constants."""

# 文件说明：
# - 本文件属于 Guardrails 层，负责输入安全、工具权限、输出合规或人工审批。
# - 保险金融场景必须拦截收益承诺、避税避债、恐吓营销和编造案例。
SALES_FORBIDDEN_CLAIMS = [
    "No guaranteed investment returns.",
    "No tax/debt evasion promises.",
    "No fear-based marketing.",
    "No fabricated case stories.",
    "No disparagement of other financial products.",
]

