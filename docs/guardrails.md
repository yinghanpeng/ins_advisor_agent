# Guardrails 安全与合规

Guardrails 用于约束输入、工具、语料和输出。

## 分层设计

- 输入层：Prompt Injection、越权请求、敏感信息、高风险意图；
- 销售语料层：原始访谈不能当系统指令，高风险话术不能直接生成；
- 工具层：权限、schema、副作用同步阻断、idempotency；
- 输出层：保险/金融合规、事实引用、敏感信息泄露；
- Source Boundary：外部内容只能当证据。

## 输入 Prompt Injection 检测补充

输入 Guardrail 的规则层不是简单执行 `pattern in text`，而是先构造仅用于检测的安全视图：

- HTML 实体和一层 URL 编码解码；
- Unicode NFKC、大小写和空白归一化；
- 移除零宽字符；
- 对数量和长度受限的 Base64/Hex UTF-8 片段做检测性解码；
- 对带 `system/instructions/filter` 锚点的高风险英语动词错拼做 Typoglycemia 检测。

标准化文本不会覆盖业务原文，也不会直接进入 Prompt。规则输出仍然是 `GuardrailSignal`，其中
`score` 只用于聚合弱特征和审计：确定性动作短语直接建议 `BLOCK`；单一角色扮演只记弱信号；
软短语、多个结构信号或混淆模式达到阈值后进入 LLM Judge 灰区。

兼容返回结构新增 `injection_score` 和 `input_risk_score`：前者只统计 Prompt Injection，后者统计
全部规则型输入风险。分值用于观测和弱信号聚合，最终动作仍由 `PolicyCombiner` 的确定性优先级决定。

`system prompt`、`developer mode`、`jailbreak`、`开发者指令`、`越权` 等单一技术名词不再直接作为 HARD 依据。只有“输出系统提示”
“override developer instructions”等动作动词与敏感对象组合，才视为高置信注入，避免正常安全讨论被误杀。

保险业务动作使用独立扫描类别，不与 Prompt Injection 混为一谈：

- `insurance_business_violation`：明确要求协助隐瞒病史、伪造材料或绕过核保，建议 `BLOCK`；
- `insurance_action_confirmation`：代投保、代签名、代支付等有法律或资金后果的动作，建议 `SAFE_FALLBACK`；
- 单独讨论“隐瞒病史有什么风险”不会命中，规则要求存在第一人称协助动作短语。

当前输入阶段不存在待审批状态。`SAFE_FALLBACK` 会阻断原动作，同步返回流程说明和无副作用的替代建议。

## 保险销售高风险表达

输出中禁止：

- 保证收益；
- 绝对安全；
- 避债避税；
- 恐吓营销；
- 编造客户故事；
- 贬低其他金融产品。

相关代码：

- `src/agent_core/guardrails/input.py`
- `src/agent_core/guardrails/output.py`
- `src/agent_core/guardrails/tool_guardrails.py`
