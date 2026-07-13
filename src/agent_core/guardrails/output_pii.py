"""输出侧 PII 二次扫描与脱敏。

# 文件说明：
# - 输入 PII 扫描发生在用户输入进入记忆、工具和模型之前。
# - 本文件负责最终答案返回前的二次扫描，防止生成内容、工具摘要或模板误带敏感信息。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputPIIPattern:
    """输出 PII 规则项，包含类型、正则、占位符和敏感级别。"""

    # pii_type 是公开 trace 中可记录的类别名，不包含原始敏感文本。
    pii_type: str
    # pattern 是用于扫描和替换的正则表达式。
    pattern: re.Pattern[str]
    # placeholder 是命中后写回 answer 的脱敏占位符。
    placeholder: str
    # high_sensitivity 标记身份证、银行卡等高敏 PII。
    high_sensitivity: bool = False


OUTPUT_PII_PATTERNS: list[OutputPIIPattern] = [
    # 中国大陆手机号：11 位，且不被更长数字串包裹。
    OutputPIIPattern("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号已脱敏]"),
    # 18 位身份证号属于高敏内容，脱敏后仍会把风险提升为 high。
    OutputPIIPattern(
        "id_card",
        re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
        "[身份证号已脱敏]",
        high_sensitivity=True,
    ),
    # 银行卡号属于高敏内容，默认脱敏并提升风险。
    OutputPIIPattern(
        "bank_card",
        re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
        "[银行卡号已脱敏]",
        high_sensitivity=True,
    ),
    # 邮箱地址属于常见联系信息，默认脱敏后继续。
    OutputPIIPattern(
        "email",
        re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
        "[邮箱已脱敏]",
    ),
    # 微信号模式保持保守，避免把普通中文误判成微信。
    OutputPIIPattern(
        "wechat",
        re.compile(r"(?:微信|微信号|wx|wechat)[:：\s]*[A-Za-z][A-Za-z0-9_-]{5,19}", re.I),
        "[微信号已脱敏]",
    ),
    # 精确地址模式只覆盖常见“省市区 + 路/街/号”结构，生产可替换专业 PII 服务。
    OutputPIIPattern(
        "address",
        re.compile(r"[\u4e00-\u9fa5]{2,}(?:省|市|区|县)[\u4e00-\u9fa5A-Za-z0-9号路街道弄室单元\-]{4,}"),
        "[地址已脱敏]",
    ),
]


def scan_and_redact_output_pii(text: str) -> tuple[str, dict[str, Any]]:
    """扫描输出文本并返回脱敏文本与公开安全的扫描结果。"""
    # redacted 保存逐条正则替换后的文本，最终会写回 state.answer。
    redacted = text or ""
    # findings 只记录 PII 类型和位置，不记录原始命中内容。
    findings: list[dict[str, Any]] = []
    # high_sensitivity 只要命中身份证或银行卡即置为 True。
    high_sensitivity = False
    # 逐条规则扫描；同一类别多次命中会记录多个位置摘要。
    for item in OUTPUT_PII_PATTERNS:
        # matches 先基于当前 redacted 文本扫描，位置用于公开 trace 摘要。
        matches = list(item.pattern.finditer(redacted))
        # 未命中则跳过该类别。
        if not matches:
            # 当前规则没有发现 PII，继续扫描下一类别。
            continue
        # 高敏标记向上聚合，用于节点提升 risk_level。
        high_sensitivity = high_sensitivity or item.high_sensitivity
        # 写入不含原文的位置摘要，方便审计但不泄露敏感值。
        findings.extend(
            {
                "pii_type": item.pii_type,
                "start": match.start(),
                "end": match.end(),
                "length": match.end() - match.start(),
            }
            for match in matches
        )
        # 把全部命中替换为占位符，默认脱敏后继续。
        redacted = item.pattern.sub(item.placeholder, redacted)
    # 组装输出侧扫描结果；该结构可以安全写入 guardrail_results 和 response_package warnings。
    result = {
        "guardrail_name": "output_pii_scan",
        "triggered": bool(findings),
        "action": "mask" if findings else "pass",
        "redacted": redacted != (text or ""),
        "pii_types": sorted({item["pii_type"] for item in findings}),
        "findings": findings,
        "high_sensitivity": high_sensitivity,
    }
    # 返回脱敏文本与公开安全扫描结果。
    return redacted, result


def redact_pii_in_public_payload(value: Any) -> Any:
    """递归脱敏 trace/stream payload，避免公开事件里残留原始 PII。"""
    # 字符串直接复用输出 PII 规则替换。
    if isinstance(value, str):
        # 获取脱敏文本并忽略仅供内部判断的扫描详情。
        redacted, _result = scan_and_redact_output_pii(value)
        # 公共 Payload 只返回脱敏文本。
        return redacted
    # 列表逐项递归，保持原有顺序。
    if isinstance(value, list):
        # 返回逐项脱敏后的新列表，不修改调用方容器。
        return [redact_pii_in_public_payload(item) for item in value]
    # 字典逐值递归，保留 key 以免破坏事件结构。
    if isinstance(value, dict):
        # 返回保留键结构但递归清理每个值的新字典。
        return {key: redact_pii_in_public_payload(item) for key, item in value.items()}
    # 其他类型原样返回。
    return value
