# Evaluator-Optimizer 有界闭环

本项目的 evaluator-optimizer 不是无限自我反思，也不是每次都重生成。它只在明显需要时触发，并且默认最多重生成一次。

## 评估维度

`evaluate_response_quality` 检查：

- 是否回答了用户问题；
- 是否有足够 grounding；
- 是否引用了工具或检索证据；
- 是否存在明显幻觉风险；
- 是否违反保险合规；
- 是否泄露 PII；
- 是否把未确认事实说成确定事实；
- 是否应该澄清却直接回答；
- 是否输出了不能执行的空话。

当前第一版是确定性规则评估，后续可以接入结构化模型评估器，但必须保持预算限制。

## 触发条件

不会每轮都重生成。触发条件包括：

- `grounding_result["grounded"] is False`；
- `risk_level in {"medium", "high"}`；
- 合规审查出现 warning；
- `output_pii_scan_result["triggered"] is True`；
- `answer` 太短；
- `context_needs["tool"] is True` 但没有成功工具结果；
- 工具结果存在但回答没有使用工具语义；
- `context_needs["clarify"] is True` 但没有走澄清分支。

## 重生成规则

`regenerate_response_if_needed` 只允许：

- 默认最多 1 次；
- 复用同一个 `compressed_context`；
- 复用同一个 `tool_results`；
- 不重新调用外部工具；
- 不编造未确认事实。

重生成后主链路会再次执行：

```text
output_pii_scan
grounding_verification
compliance_review
```

如果预算耗尽仍不合格，系统会在 `response_package["warnings"]` 中写入“证据不足/已降级”。

## 为什么受 budget 限制

生成闭环如果不设上限，很容易变成隐藏循环：

- 评估不通过；
- 再生成；
- 再评估；
- 再生成；
- 成本和延迟失控。

因此本项目把闭环放进显式状态字段：

- `evaluation_result`；
- `regeneration_attempts`；
- `metadata["response_warnings"]`。

测试会验证最多只重生成一次。
