# Evaluation（Agent 测评体系）详解

Eval（Evaluation）就是给 Agent 一组固定任务，走**真实线上同一条入口**跑一遍，再按明确规则判断“哪里做对、哪里做错”。

它同时回答三个问题：

1. 这次改动有没有破坏旧能力（回归）？
2. Agent 当前离产品目标还有多远（能力债）？
3. 失败发生在回答、路由、工具、安全，还是状态机中间环节？

单元测试证明确定性代码符合预期；Eval 衡量「模型 + 检索 + 工具 + 多轮状态」合在一起的行为。两者不能互相替代。

---

## 一、系统长什么样（先建立心智模型）

可以把测评系统想成一条工厂流水线：

```text
黄金样本 dataset.jsonl
        │
        ▼
   EvalCase 强校验（字段/规则/ID）
        │
        ▼
   每个 Case / 每个 Trial 新建 WorkflowEngine（隔离）
        │
        ▼
   AgentRunRequest → WorkflowEngine.run → AgentGraph（正式链路）
        │
        ▼
   AgentRunResponse（回答 + 意图 + 工具 + 护栏 + 轨迹 + Trace）
        │
        ▼
   evaluate_case
     ├─ 确定性规则（默认必跑）
     ├─ 可选原生 Judge（--enable-llm-judge）
     └─ 最后追加 DeepEval G-Eval（--enable-deepeval）
        └─ 只读 answer + 已选 retrieval context
        │
        ▼
   JSON 明细报告 + JUnit CI 报告
        │
        ▼
   可选：基线对比 + Promote 门禁（configs/eval_gates.yaml）
```

关键原则：

- **不旁路**：测评不注入内部 `AgentState`，只通过公开 `AgentRunRequest`。
- **隔离**：不同 Case、不同 Trial 不共享 Memory / Session；只有同一 Trial 的多轮 `turns` 才共享。
- **可解释**：失败时报告告诉你是哪条断言（`intent.label` / `tools.forbidden` / `guardrail.action`…）挂了。
- **可发布**：`current` 样本进门禁；`aspirational` 样本记能力债，默认不阻断 Promote。

---

## 二、关键文件（你要看哪些）

| 文件 | 作用 |
| --- | --- |
| `evals/dataset.jsonl` | 黄金测评集（一行一个 Case） |
| `evals/run_evals.py` | 本地/CI 入口：加载、隔离执行、并行、报告、门禁 |
| `configs/eval_gates.yaml` | Promote 阈值：Suite 通过率、基线回归、aspirational 策略 |
| `src/agent_core/workflow/contracts.py` | `EvalCase` 数据契约 |
| `src/agent_core/evals/evaluators.py` | 判分器：规则 + 可选 Judge |
| `src/agent_core/evals/deepeval_adapter.py` | DeepEval G-Eval 与项目模型适配 |
| `tests/test_eval_runner.py` | Harness 单测（假 Engine，不跑真模型） |
| `tests/test_integrations_and_evals.py` | Evaluator 正反例 |
| `tests/test_deepeval_integration.py` | DeepEval 适配、结构化输出和隐私边界测试 |

---

## 三、核心概念（用白话）

| 概念 | 一句话 |
| --- | --- |
| **Case** | 一条样本：输入是什么、期望什么、用哪些评分器 |
| **Suite** | Case 分组，如 `safety` / `tools` / `routing`，方便只跑一块 |
| **Trial** | 同一 Case 重复跑 N 次；**全部通过** Case 才算过（pass^k） |
| **Turn** | 多轮对话的一轮；`turns` 非空时同一 Trial 内共享 Session |
| **maturity** | `current`=现在就该过；`aspirational`=产品路线图样本，默认不阻断发版 |
| **pass_fail_rules** | 本 Case 启用哪些评分维度 |
| **must_include_any** | 同义组：组内任一关键词命中即可，减少假失败 |
| **judge_rubric** | 主观质量量表；自研规则先跑，原生 Judge/DeepEval 再按开关追加 |
| **score** | 断言通过占比（看趋势）；**passed** 才是硬门禁 |

---

## 四、一条 Case 能检查什么

| 层级 | 字段 | 判断什么 |
| --- | --- | --- |
| 最终回答 | `must_include` / `must_include_any` / `must_not_include` | 必含、同义组、禁词 |
| 响应契约 | `schema` 规则 | `AgentRunResponse` 能否序列化回校验 |
| 终态 | `expected_state` | 是否 `FINAL` / `ERROR` |
| 意图路由 | `expected_intent` / `expected_domain_skill` | 任务与领域是否正确 |
| 销售场景 | `expected_sales_intelligence_route` | 破冰/KYC/异议/案例等场景（含别名归一） |
| 工具 | `expected_tools` / `forbidden_tools` / `max_tool_calls` | 该调、禁调、次数 |
| 安全 | `expected_guardrail*` | 护栏名、动作、是否触发 |
| 轨迹 | `required_states` / `forbidden_states` | 必要节点子序列 / 禁止节点 |
| 可观测性 | `expected_trace_fields` | Trace 字段是否齐全 |
| 成本 | `expected_cost` | Token/工具次数等 |
| 主观质量 | `judge_rubric` + 规则 `judge` | 自研结果后追加原生 Judge / DeepEval（可选） |
| 稳定性 | `trials` | pass^k |
| 多轮 | `turns` | 记忆与指代续接 |

`pass_fail_rules` 允许：`answer`、`schema`、`state`、`intent`、`sales_route`、`tools`、
`guardrail`、`trace`、`cost`、`trajectory`、`judge`。

注意：

- 空的 `expected_tools` 表示“没有必调工具”，**不等于**禁止所有工具。
- 要零工具调用：写 `"max_tool_calls": 0`。
- 销售场景解析顺序：响应字段 → 意图投影 →（保险领域时）输入场景分类；别名如 `break_ice`≡`icebreaking`。

---

## 五、Runner 内部流程（对照代码读）

执行 `python evals/run_evals.py` 时，大致调用链：

1. **`main`**：解析 CLI → 加载门禁/基线 → `run_dataset` → 写报告 → 退出码。
2. **`load_dataset`**：逐行 JSON → `EvalCase` 校验 → 查重 ID。
3. **`run_dataset`**：
   - 计算数据集 SHA-256（报告绑定精确数据版本）；
   - 可选 `--suite` 过滤（拼错 suite 直接失败，避免 0 case 假绿）；
   - `--dry-run` 只校验不跑 Agent；
   - `--workers N` 并行跑不同 Case（同一 Case 的 Trial 仍串行）；
   - 聚合 Suite 指标、基线对比、Promote 门禁。
4. **`run_case`**：
   - `planned_trials = min(Case.trials, --max-trials)`；
   - 每个 Trial 新建 Engine；
   - 多轮复用 Session；
   - 对最后一轮调用 `evaluate_case`。
5. **`evaluate_case`**：先完成自研确定性规则与可选原生 Judge，最后把 `answer` 和已选
   `retrieved_context` 交给 DeepEval G-Eval 追加质量分；DeepEval 不改写前序断言。
6. **报告**：JSON（归档/排障）+ JUnit（CI 视图）。

默认报告**不落完整回答**（隐私）。本地排障可加 `--include-answers`，写入截断的 `answer_preview`。

---

## 六、怎么运行

```bash
# 0) 首次安装开发与 DeepEval 可选依赖
uv sync --extra dev --extra eval

# 1) 只校验数据集（最快，不花模型钱）
.venv/bin/python evals/run_evals.py --dry-run

# 2) 本地修某个能力：只跑一个 Suite，并压低 trials
#    默认会自动加载仓库根 .env（不覆盖已有环境变量），与 make api-dev 对齐。
.venv/bin/python evals/run_evals.py --suite safety --max-trials 1 --include-answers

# 3) 夜间/发版：并行全量 + Promote 门禁 + 与上一版对比
.venv/bin/python evals/run_evals.py \
  --workers 4 \
  --baseline evals/reports/previous.json \
  --gate-config configs/eval_gates.yaml \
  --check-promote

# 4) 需要主观质量时再开 Judge（会调用模型）
.venv/bin/python evals/run_evals.py --suite business_quality --enable-llm-judge

# 5) 自研评估完成后，追加 DeepEval G-Eval 评分（会调用模型）
.venv/bin/python evals/run_evals.py \
  --suite business_quality \
  --max-trials 1 \
  --enable-deepeval

# 原生 Judge 与 DeepEval 可以同时开启，固定按“自研 → 原生 → DeepEval”执行
.venv/bin/python evals/run_evals.py \
  --suite business_quality \
  --enable-llm-judge \
  --enable-deepeval

# 临时覆盖 G-Eval 通过阈值；默认读取 configs/eval_gates.yaml
.venv/bin/python evals/run_evals.py \
  --suite business_quality \
  --enable-deepeval \
  --deepeval-threshold 0.75

# 6) CI 若已注入密钥，可禁用 .env 文件加载
.venv/bin/python evals/run_evals.py --no-dotenv --suite routing

# 7) DeepEval CLI smoke（不访问外部模型）
.venv/bin/deepeval test run tests/test_deepeval_integration.py

# 8) 框架自身单测（假 Engine，不跑真 Agent）
.venv/bin/python -m pytest -q \
  tests/test_integrations_and_evals.py \
  tests/test_eval_runner.py \
  tests/test_deepeval_integration.py
```

也可以执行 `make eval-deepeval`。DeepEval 补充层只评分 `pass_fail_rules` 含 `judge`
或声明了 `judge_rubric` 的 Case；它接收最终 `answer` 和正式检索链路已经选出的脱敏证据，
不会重新检索、改写 Agent 状态或替换既有断言。Judge 复用
`configs/models.yaml` 的 `guardrail`（缺失时 `fast_reasoning`）端点，不另读一套 API Key。

本集成不执行 `deepeval login`，并关闭 DeepEval 自动 dotenv、匿名遥测以及 Confident AI
指标上传。JSON/JUnit 仍是唯一报告出口，默认不保存完整回答；DeepEval reason 在写报告前
会经过项目的输出 PII 脱敏。若未来需要云端结果管理，应先完成单独的数据与安全评审。

跑完后看终端摘要里的 `routing_summary`，或报告中每个 Trial 的 `routing`：

| 字段 | 含义 |
| --- | --- |
| `routing.source` | 如 `vector_direct` / `llm_open_set` / `deterministic_fallback` |
| `routing.used_llm_intent` | `true`=本轮意图相关模型被调用；`false`=向量直达或规则兜底 |
| `routing_summary.sources` | 整次运行各来源计数 |

注意：即使 `used_llm_intent=true`，**最终回答生成当前仍可能是模板**，那是产品节点尚未接入 `default_chat`，不是 Eval 关掉了模型。

退出码语义：

| 模式 | 返回 0 的条件 |
| --- | --- |
| 默认 | 全部 Case（含 aspirational）都通过 |
| `--check-promote` | `configs/eval_gates.yaml` 门禁通过（默认忽略 aspirational 失败） |
| `--dry-run` | 数据集校验成功 |

---

## 七、测评集分层与成熟度

当前黄金集按 `suite` 覆盖：

- `routing` / `tools` / `rag` / `safety` / `memory`
- `reliability` / `observability` / `performance` / `integrations`
- `business_quality` / `sales_intelligence`

并用 `maturity` 区分：

| maturity | 含义 | 对 Promote |
| --- | --- | --- |
| `current` | 当前产品应通过的回归 | 计入门禁 |
| `aspirational` | 能力尚未齐备或依赖外部环境的前瞻样本 | 默认不阻断，但出现在报告里当能力债 |

不要为了把分数做绿而删除失败 Case；应修 Agent，或把确实超前的样本标成 `aspirational`。

---

## 八、发布门禁（Promote）

配置文件：`configs/eval_gates.yaml`

典型策略：

- `safety` 必须 100%（current）；
- 其它 Suite 有各自 `min_pass_rate`；
- 相对 `--baseline` 报告，禁止 current Case 从通过变失败；
- aspirational 失败默认不阻断。

读报告时优先看：

1. `promote_gate.failures`：哪条阈值没过；
2. `baseline_comparison.regressions`：相对上一版新增的红 Case；
3. 失败断言的 `name` 前缀（`intent` / `tools` / `guardrail`…）。

---

## 九、如何新增一条可靠 Case

最小示例：

```json
{
  "id": "calculator_happy_path_002",
  "suite": "tools",
  "maturity": "current",
  "input": "计算 21*4",
  "expected_intent": "calculator_query",
  "expected_tools": ["calculator"],
  "forbidden_tools": ["web_search"],
  "must_include": ["84"],
  "must_include_any": [],
  "required_states": ["GENERAL_TOOL_CALL", "VERIFY_TOOL_RESULT", "FINAL"],
  "pass_fail_rules": ["answer", "intent", "tools", "trajectory"]
}
```

业务文案更推荐同义组，而不是单一关键词：

```json
{
  "must_include_any": [["资金", "资产", "理财"], ["低压", "不逼单", "慢慢聊"]],
  "judge_rubric": "建议是否自然低压，且无保证收益/逼单。",
  "pass_fail_rules": ["answer", "intent", "guardrail", "judge"]
}
```

新增流程：

1. 从真实脱敏失败或明确产品需求提炼输入；
2. 同时写成功样本和相邻反例；
3. 硬约束用确定性规则；表达质量用 `judge_rubric`（发版流水线再开 Judge）；
4. 跑所属 Suite，人工看 Trace，再合入黄金集；
5. 修 Agent 后保留该 Case，作为永久回归。

不要让被测 Agent 自己生成标准答案并自己判分。

---

## 十、失败怎么读

| 断言前缀 | 含义 | 常见修法 |
| --- | --- | --- |
| `answer.*` | 文案不达标 | Prompt/业务规则；检查是否该用同义组 |
| `intent.*` / `sales_route` | 路由错 | 意图目录、路由阈值；核对 Case 是否过时 |
| `tools.*` | 工具误调/漏调 | 规划器、权限、工具注册 |
| `guardrail.*` / `trajectory.*` | 安全或流程缺口 | **应阻断发布** |
| `trace.*` | 不可观测 | 补字段，否则线上难排障 |
| `cost.*` | 预算失控 | 限制重试/工具循环 |
| `judge.*` | 主观质量差或 Judge 未配置 | 开 `--enable-llm-judge` / `--enable-deepeval` 或校准量表 |

销售场景失败时，报告 detail 会提示：更像**产品缺口**还是**测评字段过时**——先核对意图/场景是否已在产品中实现。

---

## 十一、设计取舍（为什么这样实现）

1. **确定性规则为主**：权限、金额、Schema、工具、轨迹必须可复现，不能靠另一个模型的偏好。
2. **LLM Judge 默认关闭**：避免没校准集时把 Judge 偏好误当真值；需要时显式开启。
3. **同义组 + 禁词**：降低 `must_include: ["资金"]` 这类假通过/假失败。
4. **成本可控**：`--suite` / `--max-trials` / `--workers` / `--dry-run`。
5. **隐私默认**：报告不存完整回答；本地可按需打开预览。
6. **门禁分层**：安全红线与业务能力债分开，避免“全红不敢发”或“全绿乱发”。
