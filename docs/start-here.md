# 我看不懂项目，无从下手怎么办？

如果你第一次打开这个项目，不要从所有目录一起看。按下面 5 步走就够了。

## 第 1 步：先跑起来

在项目根目录执行：

```bash
python3 main.py
```

你会看到 4 条示例输入，每条都会打印：

- 用户输入；
- 最终回答；
- `trace_id`；
- 意图识别结果；
- 命中的业务 Skill；
- 状态流转路径；
- Guardrail 结果；
- 检索到的销售洞察卡片；
- 完整状态迁移记录。

只测试一条输入：

```bash
python3 main.py --message "客户喜欢银行理财，我怎么破冰"
```

进入交互模式：

```bash
python3 main.py --interactive
```

## 第 2 步：只看 4 个文件

先不要看完整工程，只看这几个：

1. `main.py`：本地怎么运行。
2. `src/agent_core/workflow/engine.py`：请求从哪里进入 Agent Core。
3. `src/agent_core/graph/nodes.py`：每个状态节点做什么。
4. `src/agent_core/sales_intelligence/retriever.py`：销售洞察如何检索。

看懂这 4 个文件，你就能理解主链路。

## 第 3 步：理解一句话架构

这个项目可以先理解成：

```text
用户输入
→ WorkflowEngine
→ 状态机节点
→ 意图识别
→ 通用能力或保险顾问 Skill
→ Sales Intelligence 检索
→ Context Builder
→ 生成回答
→ Guardrail 审查
→ 最终输出和 trace
```

## 第 4 步：再看文档

推荐阅读顺序：

1. `docs/conversation-flows.md`
2. `docs/project-structure.md`
3. `docs/production-readiness-checklist.md`
4. `docs/sales-intelligence-layer.md`
5. `docs/state-machine.md`

## 第 5 步：再看测试

测试是最短的“用法说明”：

- `tests/test_workflow_engine.py`：主流程怎么跑；
- `tests/test_trace_and_security.py`：trace 和注入防护怎么验证；
- `tests/test_rag_hybrid_memory_approval.py`：RAG、Memory、审批怎么验证；
- `tests/test_sales_pipeline.py`：访谈语料怎么变成卡片。

## 你现在应该怎么改？

如果你想继续开发，建议按这个顺序：

1. 先把 `main.py` 跑通；
2. 在 `src/agent_core/graph/nodes.py` 里加一个新节点；
3. 在 `src/agent_core/workflow/steps.py` 里补 step contract；
4. 在 `tests/` 里写一个对应测试；
5. 再把文档补到 `docs/project-structure.md`。

