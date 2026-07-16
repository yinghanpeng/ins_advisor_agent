# 第一次阅读项目

## 先运行

```bash
python3 main.py --message "客户喜欢银行理财，我怎么破冰"
python3 main.py --interactive
pytest -q
```

首轮保险请求通常不会直接给完整策略，而是建立 active intent 并提出一个温和 KYC 问题。交互模式下
继续回答，才能看到活跃意图续接、槽位合并和策略生成。

## 推荐阅读顺序

1. `src/agent_core/workflow/engine.py`：统一兼容入口；
2. `src/agent_core/graph/builder.py`：真实代码执行顺序；
3. `src/agent_core/intents/router.py`：向量 + LLM 双层意图路由；
4. `src/agent_core/graph/nodes.py`：状态节点；
5. `src/agent_core/agents/registry.py`：专业 Agent 如何被发现和选择；
6. `src/agent_core/agents/advisor_coach/agent.py`：现有保险顾问 Agent 的完整业务顺序；
7. `src/agent_core/agents/insurance_proposal/`：计划书契约和默认禁用占位；
8. `src/agent_core/skills/insurance_advisor/kyc.py`：保险领域槽位；
9. `src/agent_core/skills/insurance_advisor/knowledge.py`：双知识库；
10. `tests/test_intent_routing.py`：阈值、活跃意图和换题最小示例。

一句话架构：

```text
输入安全
→ Redis active intent
→ 向量意图库
→ 必要时 LLM 裁定
→ 置信度分发
→ DomainAgentRegistry
→ 通用工具路径 / AdvisorCoachAgent / 主动澄清
→ Grounding、PII、合规和记忆
```

保险逻辑不再运行 Dify Workflow，也不要求调用方指定 `workflow_name`。

## 修改方式

- 新增意图：更新 `configs/intent_catalog.yaml`、知识库语料和路由测试；
- 调整阈值：修改 `configs/intent_routing.yaml`，同时用标注集重新评估；
- 新增保险字段：修改 `InsuranceKycDelta`、意图字段优先级、问题模板和业务记忆测试；
- 接真实知识库：修改 `configs/insurance_handler.yaml` 和 Provider 配置；
- 新增工具：定义 ToolSpec、Schema、权限、Runner 和失败测试；
- 修改公共总控：修改 `builder.py`/`nodes.py`；修改保险领域顺序：修改 `agents/advisor_coach/agent.py`。

完整流程见 [request-lifecycle-flowchart.md](request-lifecycle-flowchart.md)。
