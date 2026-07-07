# Evaluation 评估体系

本项目把评估作为 Agent 工程的一等公民，而不是上线后靠人工感觉判断质量。

## 当前文件

- `evals/dataset.jsonl`：本地评估数据集；
- `evals/run_evals.py`：本地评估运行入口；
- `src/agent_core/evals/evaluators.py`：规则评估、schema 评估、LLM-as-judge adapter；
- `src/agent_core/evals/langsmith_dataset.py`：LangSmith dataset adapter；
- `src/agent_core/evals/langsmith_runner.py`：LangSmith eval runner adapter。

## 当前覆盖类型

数据集覆盖：

- 普通任务；
- 模糊输入；
- 信息缺失；
- 工具失败；
- RAG 无结果；
- Prompt Injection；
- 高风险请求；
- 成本压力；
- 多轮状态；
- Dify 调用；
- LangSmith trace；
- 销售破冰；
- KYC 追问；
- 异议处理；
- 案例讲述；
- 计划书收口；
- 销售语料高风险处理。

## 运行方式

```bash
python3 evals/run_evals.py
```

