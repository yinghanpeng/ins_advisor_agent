"""Output guardrails for insurance/financial advice."""

# 文件说明：
# - 本文件属于 Guardrails 层，负责输入安全、工具权限、输出合规或人工审批。
# - 保险金融场景必须拦截收益承诺、避税避债、恐吓营销和编造案例。
from __future__ import annotations


BLOCKED_TERMS = [
    "保证收益",
    "绝对安全",
    "避债避税",
    "谁都动不了",
    "一定更好",
    "制造焦虑",
]


class OutputGuardrail:
    """保险金融输出合规检查器，拦截高风险承诺和误导性表达。"""

    def review(self, text: str) -> dict:
        """检查最终回答是否包含禁用表达，并决定 pass 或 block。"""
        # 重点逻辑：用规则表先拦截最明确的保险/金融禁用表达。
        hits = [term for term in BLOCKED_TERMS if term in text]
        return {
            "guardrail_name": "insurance_output_compliance",
            "triggered": bool(hits),
            "reason": ",".join(hits),
            "action": "block" if hits else "pass",
        }
