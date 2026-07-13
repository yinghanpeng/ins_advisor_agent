"""输入 Guardrail 编排器（三层架构入口）。

# 文件说明：
# - 本文件把三层拼装成一条清晰流水线：
#     第一层 硬闸（规则/正则/PII）  →  第二层 LLM Judge（仅灰区）  →  第三层 PolicyCombiner（最终裁决）。
# - 职责单一：只做"编排 + 结果格式化"，不写检测规则、不写动作优先级（分别在 prompt_injection/pii 与 combiner 中）。
# - review() 返回带 action/triggered/guardrail_name/reason 的 dict；
#   其中 action 保留 pass / safe_fallback / block 三种可执行语义；
#   同时附带 decision_action/risk_level/masked/sanitized_text/signals 等更细粒度字段供观测与新逻辑使用。
"""

from __future__ import annotations

from agent_core.guardrails.combiner import InputGuardrailPolicy, combine
from agent_core.guardrails.insurance_input import scan_insurance_input_risk
from agent_core.guardrails.llm_judge import judge_input_safety
from agent_core.guardrails.pii import scan_pii
from agent_core.guardrails.prompt_injection import scan_prompt_injection
from agent_core.guardrails.schemas import GuardrailAction, GuardrailDecision, GuardrailSignal, RiskLevel


# 执行动作映射：allow/mask 都继续流转；safe_fallback/block 保留独立语义。
_ACTION_MAP: dict[GuardrailAction, str] = {
    GuardrailAction.ALLOW: "pass",
    GuardrailAction.MASK: "pass",
    GuardrailAction.SAFE_FALLBACK: "safe_fallback",
    GuardrailAction.BLOCK: "block",
}


class InputGuardrail:
    """输入安全检查器：硬闸 → LLM Judge（灰区）→ PolicyCombiner。"""

    def __init__(self, policy: InputGuardrailPolicy | None = None, *, config_dir: str = "configs") -> None:
        """注入租户策略与配置目录；缺省用默认策略。"""
        # 租户级策略：控制灰区兜底与 PII 处置。
        self.policy = policy or InputGuardrailPolicy()
        # 配置目录：透传给 LLM Judge 以加载模型端点。
        self.config_dir = config_dir

    def evaluate(self, text: str) -> GuardrailDecision:
        """执行三层评估，返回结构化最终裁决（新代码推荐使用）。"""
        # ---------- 第一层：硬闸（规则/正则/PII），永远执行，产出确定性证据 ----------
        # 注入/越权模式扫描，产出 HARD(建议 BLOCK) 与 SOFT(建议 SAFE_FALLBACK) 两类信号。
        signals: list[GuardrailSignal] = scan_prompt_injection(text)
        # 保险业务风险与 Prompt Injection 分开扫描，明确区分违规协助和需要确认的代操作请求。
        signals += scan_insurance_input_risk(text)
        # PII 正则扫描，产出建议 MASK 的敏感信息信号。
        signals += scan_pii(text)

        # 是否已存在确定性拦截信号（HIGH + BLOCK）。命中则短路，绝不浪费 LLM 调用。
        has_hard_block = any(s.severity == RiskLevel.HIGH and s.suggested_action == GuardrailAction.BLOCK for s in signals)
        # 是否存在灰区信号（软可疑，建议 SAFE_FALLBACK）。只有灰区才需要 LLM 语义判定。
        has_gray_zone = any(s.suggested_action == GuardrailAction.SAFE_FALLBACK for s in signals)

        # ---------- 第二层：LLM Judge，仅在"非确定性拦截 且 命中灰区"时调用 ----------
        # 干净输入或已确定拦截都不调用模型：前者省 token，后者结论已定。
        if not has_hard_block and has_gray_zone:
            # LLM 判定成功返回一条语义信号；不可用时返回 None，由 Combiner 走确定性兜底。
            llm_signal = judge_input_safety(text, config_dir=self.config_dir)
            # 只有通过 Schema 校验的有效模型信号才追加到最终证据链。
            if llm_signal is not None:
                # 将语义 Judge 信号加入规则/PII 信号列表，由 Combiner 统一裁决。
                signals.append(llm_signal)

        # ---------- 第三层：PolicyCombiner，按严格优先级做唯一裁决 ----------
        return combine(text, signals, policy=self.policy)

    def review(self, text: str) -> dict:
        """[兼容入口] 返回下游节点可直接消费的 dict 结果。"""
        # 先拿到结构化裁决。
        decision = self.evaluate(text)
        # 映射成下游执行动作。
        action = _ACTION_MAP[decision.action]
        # 组装结果：旧字段保证兼容，新字段用于观测与 MASK 续跑。
        return {
            # 保留稳定 guardrail 名，供 workflow contract、eval 与审计引用。
            "guardrail_name": "input_prompt_injection",
            # triggered 表示是否命中任何非放行动作。
            "triggered": decision.triggered,
            # reason 汇总关键信号说明。
            "reason": decision.reason,
            # 执行动作：pass / safe_fallback / block。
            "action": action,
            # 新动作：allow / mask / safe_fallback / block，供精细化观测。
            "decision_action": decision.action.value,
            # 综合风险等级，节点会同步到 state.risk_level。
            "risk_level": decision.risk_level.value,
            # masked 标记：为真时节点需用 sanitized_text 替换输入后继续。
            "masked": decision.action == GuardrailAction.MASK,
            # 脱敏文本：仅 MASK 时非空。
            "sanitized_text": decision.sanitized_text,
            # 完整证据链，便于 trace 回放"是谁、因为什么、判了什么"。
            "signals": [signal.model_dump() for signal in decision.signals],
            # injection_score 只聚合 Prompt Injection 规则分，避免把保险业务动作错误解释为注入攻击。
            "injection_score": sum(
                signal.score
                for signal in decision.signals
                if signal.category in {"prompt_injection", "soft_suspicious", "suspicious_structure"}
            ),
            # input_risk_score 聚合全部确定性规则分，供 API、trace 和阈值监控观察输入总体风险。
            "input_risk_score": sum(
                signal.score
                for signal in decision.signals
                if signal.source.value == "hard_rule"
            ),
        }
