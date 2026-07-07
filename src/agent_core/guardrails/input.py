"""Input guardrails."""

# 文件说明：
# - 本文件属于 Guardrails 层，负责输入安全、工具权限、输出合规或人工审批。
# - 保险金融场景必须拦截收益承诺、避税避债、恐吓营销和编造案例。
from __future__ import annotations

from agent_core.guardrails.prompt_injection import detect_prompt_injection


class InputGuardrail:
    """输入安全检查器，负责在路由和工具调用前拦截越权提示。"""

    def review(self, text: str) -> dict:
        """检查用户输入是否疑似 prompt injection，并返回结构化风控结果。"""
        # 重点逻辑：输入 Guardrail 只做“是否允许继续流转”的快速判断，不生成业务回答。
        triggered = detect_prompt_injection(text)
        return {
            "guardrail_name": "input_prompt_injection",
            "triggered": triggered,
            "reason": "prompt injection pattern detected" if triggered else "",
            "action": "block" if triggered else "pass",
        }
