"""第一层硬闸子模块：PII 正则扫描与脱敏。

# 文件说明：
# - 属于输入 Guardrail 第一层（规则 / 正则 / PII 硬闸）。
# - 保险场景高频出现手机号、身份证、银行卡、邮箱等个人敏感信息。
# - 设计原则：PII 默认"脱敏后继续"（MASK），而不是直接拦截，避免误伤正常业务问询。
"""

from __future__ import annotations

import re

from agent_core.guardrails.schemas import GuardrailAction, GuardrailSignal, RiskLevel, SignalSource


# PII 正则表：每项 = (类别, 已编译正则, 遮蔽占位符)。
# 说明：正则以"够用且低误报"为目标，生产可替换为专业 PII 识别服务（如 Presidio）。
_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # 中国大陆手机号：1 开头，第二位 3-9，共 11 位，且不被更长数字串包裹。
    ("pii_phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号已脱敏]"),
    # 18 位身份证号：17 位数字 + 校验位（数字或 X）。
    ("pii_id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[身份证号已脱敏]"),
    # 银行卡号：连续 16-19 位数字。
    ("pii_bank_card", re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[银行卡号已脱敏]"),
    # 邮箱地址。
    ("pii_email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[邮箱已脱敏]"),
]


def scan_pii(text: str) -> list[GuardrailSignal]:
    """扫描文本中的 PII，返回信号列表；不修改文本，只产出证据。

    注意：signal.matched 不落 PII 明文，改用"类别 + 命中次数"表示，避免审计日志二次泄露。
    """
    # signals 收集本次命中的所有 PII 类别信号。
    signals: list[GuardrailSignal] = []
    # 逐条正则扫描；同一类别多次命中只产出一条信号，用 count 记录次数。
    for category, pattern, _placeholder in _PII_PATTERNS:
        # findall 命中次数决定该类别是否触发。
        matches = pattern.findall(text)
        # 未命中则跳过该类别。
        if not matches:
            # 当前类别无命中时继续下一条 PII 规则。
            continue
        # 命中则产出一条 PII 信号；severity 定为 MEDIUM，建议动作 MASK（脱敏续跑）。
        signals.append(
            GuardrailSignal(
                source=SignalSource.PII_SCAN,
                category=category,
                severity=RiskLevel.MEDIUM,
                # 只记录命中次数，绝不把 PII 明文写入信号，防止审计日志泄露。
                matched=f"count={len(matches)}",
                detail=f"检测到疑似 {category}，命中 {len(matches)} 处。",
                suggested_action=GuardrailAction.MASK,
            )
        )
    # 返回全部 PII 信号，交给 PolicyCombiner 统一裁决。
    return signals


def redact_pii(text: str) -> str:
    """把文本中的 PII 替换为占位符，用于 MASK 动作下的脱敏续跑。"""
    # redacted 是逐条正则替换后的结果。
    redacted = text
    # 按 PII 正则表依次替换，替换顺序不影响结果（各类别互不重叠）。
    for _category, pattern, placeholder in _PII_PATTERNS:
        # 用占位符替换全部命中片段。
        redacted = pattern.sub(placeholder, redacted)
    # 返回脱敏后的安全文本。
    return redacted
