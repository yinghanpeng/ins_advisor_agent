"""Guardrail 分层架构的共享数据契约。

# 文件说明：
# - 本文件是输入 Guardrail 三层架构（硬闸 / LLM Judge / PolicyCombiner）的公共语言。
# - 三层之间只通过这里定义的 GuardrailSignal 传递"证据"，通过 GuardrailDecision 传递"裁决"，
#   从而把"检测"和"动作裁决"彻底解耦：扫描器只负责产出信号，不做动作；Combiner 只做动作，不做检测。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class GuardrailAction(StrEnum):
    """最终动作枚举，取代过去只有 block/pass 的二元判断。

    动作优先级（严格从高到低）：BLOCK > REVIEW > MASK > ALLOW。
    """

    # 放行：输入安全，可继续进入主链路。
    ALLOW = "allow"
    # 脱敏续跑：命中 PII 等敏感信息，先做遮蔽再继续，而不是直接拦截。
    MASK = "mask"
    # 人工复核：灰区语义无法自动判定安全，交人工审批（fail-closed 的中间档）。
    REVIEW = "review"
    # 拦截：确定性越权 / 注入 / 违规，直接终止请求。
    BLOCK = "block"


class RiskLevel(StrEnum):
    """统一风险分级，供工具权限、人审和输出策略复用。"""

    # 无明显风险。
    LOW = "low"
    # 可疑但不确定，通常触发 LLM Judge 或人工复核。
    MEDIUM = "medium"
    # 确定性高风险，直接拦截。
    HIGH = "high"


# 信号来源枚举：标明一条证据由哪一层产出，便于审计"是谁判的"。
class SignalSource(StrEnum):
    """信号来源层。"""

    # 第一层：规则 / 正则 / 关键词硬闸。
    HARD_RULE = "hard_rule"
    # 第一层子类：PII 扫描（独立标注，动作偏向 MASK）。
    PII_SCAN = "pii_scan"
    # 第二层：LLM 语义判定。
    LLM_JUDGE = "llm_judge"


class GuardrailSignal(BaseModel):
    """一条 Guardrail 证据。扫描器只产出信号，绝不直接决定最终动作。"""

    # 该信号由哪一层产出。
    source: SignalSource = Field(..., description="信号来源层。")
    # 信号类别，例如 prompt_injection / jailbreak / pii_phone / soft_suspicious。
    category: str = Field(..., description="信号类别。")
    # 该信号自身的风险等级。
    severity: RiskLevel = Field(..., description="信号严重度。")
    # 命中的原始片段或模式，便于回放和取证（PII 片段本身不落明文，用类别代替）。
    matched: str = Field(default="", description="命中的模式或片段摘要。")
    # 人类可读的说明。
    detail: str = Field(default="", description="信号说明。")
    # 该信号"建议"的动作；最终动作仍由 PolicyCombiner 统一裁决。
    suggested_action: GuardrailAction = Field(
        default=GuardrailAction.ALLOW,
        description="该信号建议的动作，仅作参考，不是最终裁决。",
    )


class GuardrailDecision(BaseModel):
    """PolicyCombiner 输出的最终裁决，是三层架构的唯一结论。"""

    # 最终动作。
    action: GuardrailAction = Field(..., description="最终裁决动作。")
    # 综合风险等级。
    risk_level: RiskLevel = Field(..., description="综合风险等级。")
    # 是否触发了任何非放行动作，供审计快速过滤。
    triggered: bool = Field(..., description="是否命中任何风控动作（非 ALLOW）。")
    # 裁决理由，聚合关键信号说明。
    reason: str = Field(default="", description="裁决理由。")
    # 参与裁决的全部信号，形成完整证据链。
    signals: list[GuardrailSignal] = Field(default_factory=list, description="全部证据信号。")
    # 脱敏后的文本；仅 MASK 时给出，其余为 None。
    sanitized_text: str | None = Field(default=None, description="MASK 动作下的脱敏文本。")
