"""第三层：PolicyCombiner —— 输入 Guardrail 的最终动作裁决器。

# 文件说明：
# - 这是三层架构里唯一"做动作决定"的地方，前两层只产出证据信号。
# - 设计要求：纯函数、确定性、可审计、带严格优先级。相同输入永远得到相同裁决。
# - 优先级（严格从高到低）：
#     1) 任一 HIGH 且建议 BLOCK 的信号（硬闸确定性注入 / LLM 判 malicious） → BLOCK
#     2) LLM Judge 明确判 suspicious（建议 REVIEW）                          → REVIEW（人工复核）
#     3) 任一 PII 信号                                                       → MASK（脱敏续跑）/ 按策略 BLOCK
#     4) 仅有硬闸灰区软信号（soft_suspicious）：
#          - LLM 判定 safe        → ALLOW
#          - LLM 不可用/未覆盖     → 按 policy.gray_zone_default 兜底
#     5) 无任何信号                                                          → ALLOW
# - 关键设计：硬闸的"软可疑信号"本身不等于 REVIEW，它只是"灰区触发器"；
#   最终灰区如何处置由 LLM 判定或 policy.gray_zone_default 决定，从而实现"策略与机制解耦"。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_core.guardrails.pii import redact_pii
from agent_core.guardrails.schemas import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailSignal,
    RiskLevel,
    SignalSource,
)


class InputGuardrailPolicy(BaseModel):
    """输入 Guardrail 的租户级策略，控制灰区兜底与 PII 处置。

    把"策略"从"机制"里抽出来：同一套 Combiner 逻辑，不同租户可用不同 policy。
    """

    # 灰区兜底动作：软信号存在、但 LLM Judge 不可用或判定不确定时如何处置。
    # 可选 allow / review / block；默认 review（fail-closed 的温和档，不直接拦截也不放任）。
    gray_zone_default: GuardrailAction = Field(
        default=GuardrailAction.REVIEW,
        description="灰区无法自动判定安全时的兜底动作。",
    )
    # PII 命中时是否拦截；默认 False，即脱敏续跑（MASK）而非 BLOCK。
    block_on_pii: bool = Field(default=False, description="PII 命中是否直接拦截。")


def combine(
    text: str,
    signals: list[GuardrailSignal],
    *,
    policy: InputGuardrailPolicy | None = None,
) -> GuardrailDecision:
    """按严格优先级把所有信号裁决成唯一 GuardrailDecision。

    Args:
        text: 原始输入，用于 MASK 时生成脱敏文本。
        signals: 三层产出的全部证据信号。
        policy: 租户策略；缺省用默认 policy。
    """
    # 使用传入策略或默认策略。
    policy = policy or InputGuardrailPolicy()

    # 无任何信号：直接放行，风险 LOW。这是最常见的正常路径。
    if not signals:
        return GuardrailDecision(
            action=GuardrailAction.ALLOW,
            risk_level=RiskLevel.LOW,
            triggered=False,
            reason="未命中任何风控信号。",
            signals=[],
            sanitized_text=None,
        )

    # 预先分类信号，按"来源 + 建议动作"精确区分，避免灰区软信号被误当成确定性 REVIEW。
    # block_signals：HIGH 且建议 BLOCK 的确定性信号（硬闸注入 或 LLM 判 malicious）。
    block_signals = [s for s in signals if s.severity == RiskLevel.HIGH and s.suggested_action == GuardrailAction.BLOCK]
    # llm_review_signals：LLM Judge 明确判 suspicious（建议 REVIEW），这是真正需要人工复核的语义结论。
    llm_review_signals = [
        s for s in signals if s.source == SignalSource.LLM_JUDGE and s.suggested_action == GuardrailAction.REVIEW
    ]
    # pii_signals：PII 扫描信号。
    pii_signals = [s for s in signals if s.source == SignalSource.PII_SCAN]
    # soft_gray_signals：硬闸软可疑信号，仅作"灰区触发器"，处置由 LLM/策略决定，本身不直接 REVIEW。
    soft_gray_signals = [
        s for s in signals if s.source == SignalSource.HARD_RULE and s.suggested_action == GuardrailAction.REVIEW
    ]
    # llm_safe：LLM Judge 明确判定 safe（用于灰区放行）。
    llm_safe = any(s.source == SignalSource.LLM_JUDGE and s.suggested_action == GuardrailAction.ALLOW for s in signals)

    # 优先级 1：确定性拦截。任何 HIGH+BLOCK 信号一票否决。
    if block_signals:
        return GuardrailDecision(
            action=GuardrailAction.BLOCK,
            risk_level=RiskLevel.HIGH,
            triggered=True,
            reason="；".join(s.detail for s in block_signals) or "命中确定性高风险规则。",
            signals=signals,
            sanitized_text=None,
        )

    # 优先级 2：人工复核。仅当 LLM Judge 明确判 suspicious 时才进入 REVIEW（确定性的语义结论）。
    if llm_review_signals:
        return GuardrailDecision(
            action=GuardrailAction.REVIEW,
            risk_level=RiskLevel.MEDIUM,
            triggered=True,
            reason="；".join(s.detail for s in llm_review_signals) or "LLM 判定为可疑，需人工复核。",
            signals=signals,
            sanitized_text=None,
        )

    # 优先级 3：PII 处置。到这里说明没有确定性拦截也无需人工，PII 走脱敏续跑（除非策略要求拦截）。
    if pii_signals:
        # 策略要求 PII 直接拦截时走 BLOCK。
        if policy.block_on_pii:
            return GuardrailDecision(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.MEDIUM,
                triggered=True,
                reason="命中 PII 且租户策略要求拦截。",
                signals=signals,
                sanitized_text=None,
            )
        # 默认对 PII 脱敏后继续。
        return GuardrailDecision(
            action=GuardrailAction.MASK,
            risk_level=RiskLevel.MEDIUM,
            triggered=True,
            reason="；".join(s.detail for s in pii_signals) or "命中 PII，已脱敏后继续。",
            signals=signals,
            # 生成脱敏文本，供节点替换 input_text 后安全续跑。
            sanitized_text=redact_pii(text),
        )

    # 优先级 4：仅剩硬闸灰区软信号。LLM 判 safe → 放行；LLM 不可用/未覆盖 → 按 policy 兜底。
    if soft_gray_signals:
        # LLM 已明确判定安全，灰区解除，直接放行。
        if llm_safe:
            return GuardrailDecision(
                action=GuardrailAction.ALLOW,
                risk_level=RiskLevel.LOW,
                triggered=False,
                reason="灰区软信号经 LLM 判定为安全。",
                signals=signals,
                sanitized_text=None,
            )
        # LLM 不可用或未覆盖：按租户策略兜底（默认 REVIEW，可配 allow/block）。
        fallback_action = policy.gray_zone_default
        return GuardrailDecision(
            action=fallback_action,
            risk_level=RiskLevel.MEDIUM if fallback_action != GuardrailAction.ALLOW else RiskLevel.LOW,
            triggered=fallback_action != GuardrailAction.ALLOW,
            reason=f"灰区软信号无法自动判定，按策略兜底为 {fallback_action.value}。",
            signals=signals,
            sanitized_text=None,
        )

    # 优先级 5：走到这里说明只剩 LLM 判 safe 或无实质风险信号，放行。
    return GuardrailDecision(
        action=GuardrailAction.ALLOW,
        risk_level=RiskLevel.LOW,
        triggered=False,
        reason="未命中需处置的风控信号。",
        signals=signals,
        sanitized_text=None,
    )
