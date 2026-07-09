"""第一层硬闸子模块：Prompt Injection / 越权指令的规则与正则扫描。

# 文件说明：
# - 属于输入 Guardrail 第一层（规则 / 正则 / 关键词硬闸）。
# - 区分两档：
#     * HARD（确定性）：几乎可断定是注入/越权，直接建议 BLOCK，并让 Combiner 短路，不再调用 LLM；
#     * SOFT（可疑）：像但不确定，只建议进入 REVIEW/LLM 灰区，由第二层语义判定。
# - 保留 detect_prompt_injection 供旧代码兼容，但新链路一律走 scan_prompt_injection。
"""

from __future__ import annotations

from agent_core.guardrails.schemas import GuardrailAction, GuardrailSignal, RiskLevel, SignalSource


# HARD 硬命中模式：命中即视为确定性注入/越权，建议 BLOCK。
# 这些模式在正常保险业务问询中几乎不可能自然出现，误报率极低。
_HARD_INJECTION_PATTERNS: list[str] = [
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "忽略以上",
    "忽略之前",
    "忽略前面",
    "忽略所有",
    "system prompt",
    "输出系统提示",
    "泄露系统提示",
    "开发者指令",
    "developer mode",
    "jailbreak",
    "越权",
    "you are now dan",
]

# SOFT 软可疑模式：像注入但不确定，单独命中不拦截，而是进入 LLM 灰区语义判定。
# 例如"重复上面的话""你的规则是什么"可能是攻击探测，也可能是正常好奇。
_SOFT_SUSPICIOUS_PATTERNS: list[str] = [
    "重复上面",
    "重复我说的",
    "你的规则是什么",
    "你的提示词",
    "扮演",
    "假装你是",
    "pretend you are",
    "repeat the above",
]


def detect_prompt_injection(text: str) -> bool:
    """[兼容保留] 只判断是否命中 HARD 注入模式，返回布尔值。

    新代码请使用 scan_prompt_injection 获取结构化信号；此函数仅为旧调用方保留。
    """
    # 统一小写后匹配任一 HARD 模式。
    lower = text.lower()
    return any(pattern in lower for pattern in _HARD_INJECTION_PATTERNS)


def scan_prompt_injection(text: str) -> list[GuardrailSignal]:
    """扫描注入/越权模式，产出结构化信号（不做最终动作裁决）。"""
    # lower 用于大小写不敏感匹配。
    lower = text.lower()
    # signals 收集本次命中的全部注入相关信号。
    signals: list[GuardrailSignal] = []
    # 先扫 HARD：命中即产出 HIGH 严重度、建议 BLOCK 的确定性信号。
    for pattern in _HARD_INJECTION_PATTERNS:
        # 命中一个 HARD 模式就足以判定确定性注入。
        if pattern in lower:
            signals.append(
                GuardrailSignal(
                    source=SignalSource.HARD_RULE,
                    category="prompt_injection",
                    severity=RiskLevel.HIGH,
                    matched=pattern,
                    detail="命中确定性 Prompt Injection / 越权模式。",
                    suggested_action=GuardrailAction.BLOCK,
                )
            )
    # 再扫 SOFT：命中产出 MEDIUM 严重度、建议 REVIEW 的灰区信号，交第二层语义判定。
    for pattern in _SOFT_SUSPICIOUS_PATTERNS:
        # 软模式单独命中不拦截，只标记为灰区可疑。
        if pattern in lower:
            signals.append(
                GuardrailSignal(
                    source=SignalSource.HARD_RULE,
                    category="soft_suspicious",
                    severity=RiskLevel.MEDIUM,
                    matched=pattern,
                    detail="命中可疑但不确定的指令模式，需语义判定。",
                    suggested_action=GuardrailAction.REVIEW,
                )
            )
    # 返回全部注入相关信号。
    return signals
