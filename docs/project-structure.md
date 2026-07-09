# 项目结构与文件职责说明

本文档按目录说明当前项目结构，并解释每个主要文件的作用。它适合用于项目交接、后续开发维护，也适合面试时讲清楚“这不是一个大 Prompt，而是一个生产级 Agent Framework”。

## 顶层结构

```text
ins_advisor_agent/
├── README.md
├── main.py
├── pyproject.toml
├── configs/
├── data/
├── dify/
├── docs/
├── evals/
├── scripts/
├── src/
└── tests/
```

## 根目录文件

| 文件 | 作用 |
| --- | --- |
| `README.md` | 中文项目入口文档，说明项目定位、架构、快速启动、核心目录、Sales Intelligence、Dify、LangSmith、测试和扩展方式。 |
| `main.py` | 本地命令行运行入口。新手可以直接执行 `python3 main.py` 查看完整对话、状态流转、Guardrail 和检索结果。 |
| `.env.example` | 环境变量模板，列出 Agent Gateway、模型、LangSmith、成本预算、Dify 等配置项。真实密钥不应提交到仓库。 |
| `.gitignore` | Git 忽略规则，避免 `.venv`、`.idea`、缓存、密钥文件进入版本管理。 |
| `pyproject.toml` | Python 项目配置，声明依赖、可选 API 依赖、pytest 配置、代码风格基础配置。 |

## configs：运行配置

`configs/` 用来把运行参数从代码里拆出来，避免硬编码。生产环境可以替换为配置中心。

| 文件 | 作用 |
| --- | --- |
| `configs/agent.yaml` | Agent 总配置，包含默认 workflow、默认 domain skill、网关接口、运行时 checkpoint、重试等设置。 |
| `configs/states.yaml` | 状态机枚举清单，列出 `CLASSIFY_INTENT`、`ROUTE_CAPABILITY`、`GENERATE_RESPONSE`、`RECOVERY`、`FINAL` 等状态。 |
| `configs/workflow.yaml` | Workflow 配置，定义通用 workflow 和 `break_ice_assistant_workflow` 的步骤。 |
| `configs/tools.yaml` | 工具配置，描述每个工具的名称、风险等级、是否有副作用、是否需要审批、超时、权限 scope 等。 |
| `configs/general_capabilities.yaml` | 通用能力层开关，列出 web search、weather、calculator、file parser、translation、summarizer 等能力。 |
| `configs/domain_skills.yaml` | Domain Skill 配置，当前启用 `insurance_advisor`，预留研究助手、文档分析、客服、数据分析等扩展位。 |
| `configs/sales_intelligence.yaml` | Sales Intelligence Layer 配置，包含原始访谈目录、卡片目录、是否强制脱敏、是否强制合规审查、库类型等。 |
| `configs/rag.yaml` | RAG 配置，包括 chunk size、overlap、top-k、rerank top-k、evidence digest 长度和 source boundary policy。 |
| `configs/guardrails.yaml` | Guardrails 配置，覆盖输入安全、销售语料安全、工具安全、输出合规和高风险处理策略。 |
| `configs/langsmith.yaml` | LangSmith adapter 配置，说明环境变量、是否 graceful degradation、trace 哪些对象。 |
| `configs/cost_budget.yaml` | 成本预算配置，包括请求 token 预算、每日预算、最大工具调用次数、预算压力下的降级策略。 |

## data：本地数据样例

| 文件 | 作用 |
| --- | --- |
| `data/sales_insight_cards/example_card.json` | 一条已审核的销售洞察卡片示例，用来展示 Sales Intelligence Card 的字段结构和检索样例。 |

预留目录：

| 目录 | 作用 |
| --- | --- |
| `data/raw_interviews/` | 原始销售采访语料归档目录，生产中应存放脱敏前或原始引用信息，并受权限控制。 |
| `data/processed_interviews/` | 清洗、切片后的访谈中间结果目录。 |
| `data/sales_insight_cards/` | 结构化销售洞察卡片目录。 |
| `data/eval_seed_cases/` | 从销售洞察中生成 eval seed 的目录。 |

## dify：Dify Control Plane

Dify 在本项目中不是生产流量主入口，而是 Prompt 管理、Workflow 配置和内部调试的 Control Plane。

| 文件 | 作用 |
| --- | --- |
| `dify/workflow.yml` | 推荐 Dify 工作流示意：通过 HTTP 调用 FastAPI Agent Core，而不是让 Dify 承担全部生产运行逻辑。 |
| `dify/README.md` | 说明 Dify 在项目中的定位、如何调用 Agent Core，以及为什么不作为大流量 data plane。 |

### dify/nodes：Dify 节点职责说明

| 文件 | 作用 |
| --- | --- |
| `dify/nodes/intent_router.md` | 说明意图识别节点职责：识别寒暄、通用工具请求、业务 Skill、销售语料处理、危险请求等。 |
| `dify/nodes/capability_router.md` | 说明能力路由节点职责：把请求分到通用能力层或 Domain Skill 层。 |
| `dify/nodes/state_updater.md` | 说明状态更新节点职责：记录会话状态和 trace id，但长期状态由 Agent Core 管理。 |
| `dify/nodes/workflow_router.md` | 说明业务 workflow 路由，如破冰、KYC 追问、异议处理、计划书成交等。 |
| `dify/nodes/sales_intelligence_router.md` | 说明销售智能路由，强调不能直接检索原始访谈。 |
| `dify/nodes/general_tool_router.md` | 说明通用工具路由，例如天气、时间、计算、搜索、文件解析等。 |
| `dify/nodes/domain_tool_router.md` | 说明业务工具路由，保险 Skill 通过边界调用 Sales Intelligence。 |
| `dify/nodes/tool_result_verifier.md` | 说明工具结果校验节点，负责 schema、来源、错误、延迟等检查。 |
| `dify/nodes/context_builder.md` | 说明上下文构建节点，整合用户状态、检索证据、销售洞察、新闻摘要、合规规则。 |
| `dify/nodes/strategy_generator.md` | 说明策略生成节点，只消费已压缩和审核的证据 digest。 |
| `dify/nodes/compliance_reviewer.md` | 说明合规审查节点，拦截收益承诺、避税避债、恐吓营销、编造案例等。 |
| `dify/nodes/final_response.md` | 说明最终响应节点，返回最终答案、trace id 和低风险下一步动作。 |

## docs：项目文档

| 文件 | 作用 |
| --- | --- |
| `docs/project-structure.md` | 本文档，逐层说明项目结构和文件作用。 |
| `docs/start-here.md` | 新手入门文档，告诉你先跑什么、先看哪些文件、怎么一步步理解项目。 |
| `docs/conversation-flows.md` | 完整对话链路文档，展示保险破冰、通用天气、Prompt Injection、异议处理和访谈加工如何流转。 |
| `docs/production-readiness-checklist.md` | 生产级 Agent 检查表逐项对照，说明每项能力对应的代码、文档和测试。 |
| `docs/logging-and-comments-policy.md` | 注释和结构化日志规范，说明每个 Python 文件、状态迁移、工具、RAG 和 Guardrail 的日志要求。 |
| `docs/architecture.md` | 总体架构文档，说明 Control Plane / Data Plane、Agent Core、Skill、Sales Intelligence 的关系。 |
| `docs/state-machine.md` | 状态机文档，包含 Mermaid 状态图和状态流转说明。 |
| `docs/workflow.md` | Workflow 文档，说明通用 workflow 和保险破冰 workflow 的步骤。 |
| `docs/tool-system.md` | 工具系统文档，说明 `ToolSpec` 字段、工具路由、权限和结果校验。 |
| `docs/general-capabilities.md` | 通用能力层文档，说明搜索、天气、时间、计算器、文件解析、翻译、总结等能力。 |
| `docs/sales-intelligence-layer.md` | Sales Intelligence Layer 核心文档，说明销售采访语料为什么不是普通 RAG，以及卡片、库、检索、评估关系。 |
| `docs/interview-processing.md` | 销售采访处理流水线文档，说明 ingest、anonymize、clean、segment、extract、review、index、eval。 |
| `docs/rag.md` | RAG 文档，说明通用 RAG 与 Sales Intelligence RAG 的边界。 |
| `docs/memory.md` | Memory 文档，说明 session memory、task memory、preference memory 和敏感信息策略。 |
| `docs/context-engineering.md` | Context Engineering 文档，说明上下文构建、压缩和 source boundary。 |
| `docs/guardrails.md` | Guardrails 文档，说明输入、销售语料、工具、输出和 source boundary 的安全策略。 |
| `docs/human-in-the-loop.md` | 人工审批文档，说明哪些场景需要 HITL，以及审批记录应该包含什么字段。 |
| `docs/observability.md` | 可观测性文档，说明本地结构化日志字段和 trace 目标。 |
| `docs/langsmith-integration.md` | LangSmith 集成文档，说明环境变量、降级策略、trace 内容。 |
| `docs/evaluation.md` | Evaluation 文档，说明 eval 数据集、本地 evaluator、LLM-as-judge adapter。 |
| `docs/retry-recovery.md` | 重试和恢复文档，说明工具失败、检索为空、JSON 错误、预算超限、高风险输出等处理策略。 |
| `docs/cost-control.md` | 成本控制文档，说明 token budget、降 top-k、跳过可选新闻、压缩上下文等策略。 |
| `docs/dify-integration.md` | Dify 集成文档，说明 Dify 如何通过 HTTP 节点调用 Agent Core。 |
| `docs/domain-skills.md` | Domain Skill 文档，说明业务 Skill 的职责边界和扩展方式。 |
| `docs/product-contract.md` | 产品接口契约文档，说明 Agent Gateway 输入输出字段。 |
| `docs/interview-guide.md` | 面试讲解文档，说明为什么这么拆、解决什么问题、如何讲 Sales Intelligence。 |
| `docs/known-limitations.md` | 已知限制文档，明确当前 mock、adapter、未接真实 provider 的地方。 |

## evals：评估数据和运行器

| 文件 | 作用 |
| --- | --- |
| `evals/dataset.jsonl` | 本地评估数据集，覆盖 normal task、vague input、tool failure、prompt injection、sales break ice、objection handling 等场景。 |
| `evals/run_evals.py` | 本地 eval runner，读取 `dataset.jsonl`，调用 `WorkflowEngine`，执行基本通过/失败检查。 |

## scripts：脚本入口

| 文件 | 作用 |
| --- | --- |
| `scripts/run_local.sh` | 本地测试入口，当前执行 `python3 -m pytest`。 |
| `scripts/run_evals.sh` | 本地评估入口，当前执行 `python3 evals/run_evals.py`。 |
| `scripts/ingest_interviews.sh` | 销售访谈导入脚本占位，提示使用 Sales Intelligence ingestion API。 |
| `scripts/export_dify.sh` | Dify 导出脚本占位，后续可接 Dify API。 |
| `scripts/import_dify.sh` | Dify 导入脚本占位，后续可接 Dify API 或 UI 导入流程。 |

## src/agent_core：核心代码

`src/agent_core/` 是项目主体，实现 Agent Core、状态机、工具、RAG、Memory、Guardrails、Sales Intelligence、Domain Skill、评估和外部集成。

### src/agent_core 根文件

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/__init__.py` | Python package 入口，定义包版本。 |

### api：Agent Gateway 适配层

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/api/__init__.py` | API package 标识。 |
| `src/agent_core/api/schemas.py` | API schema re-export，暴露 `AgentRunRequest` 和 `AgentRunResponse`。 |
| `src/agent_core/api/routes.py` | FastAPI route factory 和可直接调用的 `run_agent`。未安装 FastAPI 时仍可导入本地函数。 |
| `src/agent_core/api/server.py` | FastAPI app 创建入口；未安装 FastAPI 时 `app=None` 并在启动时给出明确错误。 |
| `src/agent_core/api/middleware.py` | Gateway middleware 辅助函数，当前包含 trace id 注入逻辑。 |

### graph：显式状态机层（线性执行器）

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/graph/__init__.py` | graph package 标识。 |
| `src/agent_core/graph/state.py` | 定义 `AgentNode` 状态枚举和 `AgentState` 全局状态对象。 |
| `src/agent_core/graph/nodes.py` | 状态机节点函数，包括意图识别、业务路由、Sales Intelligence 检索、上下文构建、生成、合规审查。 |
| `src/agent_core/graph/builder.py` | `AgentGraph` 线性执行器；`invoke()` 在公共前置后按 `workflow_name` 分叉到 `_run_universal` / `_run_kyc`。 |
| `src/agent_core/graph/intent_classifier.py` | 意图分类：模型优先（`classify_intent_via_model`）+ 关键词规则兜底。 |
| `src/agent_core/graph/checkpoints.py` | 内存 checkpoint store，用于本地开发和测试。 |
| `src/agent_core/graph/edges.py` | 遗留的状态转移策略函数（旧显式状态图产物），当前线性执行器已内联分支，暂未引用。 |

### workflow：Workflow 执行层

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/workflow/__init__.py` | workflow package 标识。 |
| `src/agent_core/workflow/contracts.py` | 定义 `AgentRunRequest`、`AgentRunResponse`、`EvalCase` 等输入输出契约。 |
| `src/agent_core/workflow/engine.py` | `WorkflowEngine`，统一调用 graph、记录日志、输出 API response。 |
| `src/agent_core/workflow/steps.py` | 业务 workflow step 清单，当前包含 `BREAK_ICE_ASSISTANT_STEPS`。 |

### planning：Planner / Executor / Verifier / Supervisor

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/planning/__init__.py` | planning package 标识。 |
| `src/agent_core/planning/planner.py` | 根据 intent 生成步骤计划。 |
| `src/agent_core/planning/executor.py` | 执行计划的 adapter 占位，当前返回 planned-only 状态。 |
| `src/agent_core/planning/verifier.py` | 执行结果校验 adapter。 |
| `src/agent_core/planning/supervisor.py` | Supervisor 决策逻辑，判断继续还是恢复。 |

### tools：工具系统

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/tools/__init__.py` | tools package 标识。 |
| `src/agent_core/tools/schemas.py` | 定义 `ToolSpec`、`ToolCall`、`ToolResult`，是工具系统的核心契约。 |
| `src/agent_core/tools/registry.py` | 工具注册表和默认工具列表。 |
| `src/agent_core/tools/router.py` | 简单工具路由器，根据文本选择天气、时间、计算器、新闻、搜索、总结等工具。 |
| `src/agent_core/tools/permissions.py` | 工具权限策略，按 permission scope 判断是否允许调用。 |

### capabilities：通用能力层

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/capabilities/__init__.py` | capabilities package 标识。 |
| `src/agent_core/capabilities/registry.py` | 构建通用能力注册表。 |
| `src/agent_core/capabilities/router.py` | 通用能力路由 facade，复用工具路由。 |
| `src/agent_core/capabilities/time_date.py` | 时间/日期能力，返回 UTC 时间。 |
| `src/agent_core/capabilities/calculator.py` | 安全计算器，使用 AST 限制可执行表达式。 |
| `src/agent_core/capabilities/unit_converter.py` | 单位换算能力，当前支持少量示例单位。 |
| `src/agent_core/capabilities/summarizer.py` | 本地摘要 adapter，当前按字符截断模拟摘要。 |
| `src/agent_core/capabilities/translation.py` | 翻译 adapter 占位，当前返回 mock。 |
| `src/agent_core/capabilities/weather.py` | 天气查询 adapter 占位，未配置真实 provider 时返回 mock。 |
| `src/agent_core/capabilities/web_search.py` | Web 搜索 adapter 占位。 |
| `src/agent_core/capabilities/web_page_reader.py` | 网页读取 adapter 占位。 |
| `src/agent_core/capabilities/file_parser.py` | 文件解析 adapter 占位。 |
| `src/agent_core/capabilities/knowledge_search.py` | 内部知识库搜索 adapter 占位。 |
| `src/agent_core/capabilities/news_search.py` | 新闻搜索 adapter 占位。 |

### rag：通用 RAG 组件

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/rag/__init__.py` | RAG package 标识。 |
| `src/agent_core/rag/query_rewrite.py` | 销售场景 query rewrite，生成原始 query、销售痛点 query、场景 query、策略 query。 |
| `src/agent_core/rag/chunking.py` | 文本切片工具。 |
| `src/agent_core/rag/bm25.py` | 轻量词法打分占位。 |
| `src/agent_core/rag/vector.py` | 向量检索 adapter 占位。 |
| `src/agent_core/rag/reranker.py` | 通用 rerank 函数，按 score 排序。 |
| `src/agent_core/rag/evidence.py` | 通用 evidence compression，把检索结果压缩为 digest。 |
| `src/agent_core/rag/retriever.py` | 内存检索器，用于本地测试和 adapter 示例。 |
| `src/agent_core/rag/schemas.py` | RAG 检索契约，定义 query、metadata、document、result 和 filter。 |

### context：上下文工程

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/context/__init__.py` | context package 标识。 |
| `src/agent_core/context/builder.py` | `ContextBuilder`，构建销售洞察 digest 和下游生成上下文。 |
| `src/agent_core/context/compression.py` | 上下文截断/压缩工具。 |
| `src/agent_core/context/source_boundary.py` | source boundary 规则：RAG、工具、网页、销售访谈只能作为证据，不能作为系统指令。 |

### guardrails：安全与合规

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/guardrails/__init__.py` | guardrails package 标识。 |
| `src/agent_core/guardrails/prompt_injection.py` | Prompt injection 模式检测。 |
| `src/agent_core/guardrails/input.py` | 输入 guardrail，当前检查 prompt injection。 |
| `src/agent_core/guardrails/output.py` | 输出 guardrail，拦截保证收益、绝对安全、避债避税等高风险表达。 |
| `src/agent_core/guardrails/tool_guardrails.py` | 工具 guardrail，检查工具权限和审批需求。 |
| `src/agent_core/guardrails/policy.py` | 销售合规禁用声明清单。 |
| `src/agent_core/guardrails/human_approval.py` | Human-in-the-loop 审批请求和审批结果契约。 |

### memory：记忆层

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/memory/__init__.py` | memory package 标识。 |
| `src/agent_core/memory/session.py` | 会话级 memory，保存 session 维度状态。 |
| `src/agent_core/memory/task.py` | 任务级 memory 占位。 |
| `src/agent_core/memory/preference.py` | 用户偏好 memory 占位。 |
| `src/agent_core/memory/policy.py` | Memory policy，说明敏感个人信息默认不存储、需要租户边界。 |
| `src/agent_core/memory/manager.py` | 分层 Memory Manager，统一管理 session/task/preference 并记录 audit log。 |

### observability：可观测性

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/observability/__init__.py` | observability package 标识。 |
| `src/agent_core/observability/logger.py` | 本地结构化 JSON logger。 |
| `src/agent_core/observability/trace.py` | 内存 trace recorder，用于本地调试和测试。 |
| `src/agent_core/observability/metrics.py` | 简单 metrics counter。 |
| `src/agent_core/observability/langsmith_client.py` | LangSmith adapter，支持环境变量开关和 graceful degradation。 |
| `src/agent_core/observability/langsmith_callbacks.py` | LangSmith callback 构建占位，后续接 LangChain callback。 |

### evals：评估模块

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/evals/__init__.py` | evals package 标识。 |
| `src/agent_core/evals/evaluators.py` | rule-based evaluator、schema evaluator、LLM-as-judge placeholder。 |
| `src/agent_core/evals/langsmith_dataset.py` | 将本地 eval case 转为 LangSmith examples 的 adapter。 |
| `src/agent_core/evals/langsmith_runner.py` | LangSmith remote eval runner 占位。 |
| `src/agent_core/evals/feedback.py` | 人工反馈 `HumanFeedback` schema。 |

### recovery：失败恢复

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/recovery/__init__.py` | recovery package 标识。 |
| `src/agent_core/recovery/retry.py` | 简单 retry helper。 |
| `src/agent_core/recovery/fallback.py` | 降级回答模板。 |
| `src/agent_core/recovery/json_repair.py` | 从文本中提取 JSON 对象的 repair helper。 |

### cost：成本控制

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/cost/__init__.py` | cost package 标识。 |
| `src/agent_core/cost/budget.py` | 请求级 token budget 追踪和超预算拦截。 |
| `src/agent_core/cost/model_router.py` | 模型选择策略，根据复杂度和预算压力选择模型。 |

### integrations：外部系统集成

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/integrations/__init__.py` | integrations package 标识。 |
| `src/agent_core/integrations/dify_client.py` | Dify client adapter 占位，后续可接 Dify API。 |
| `src/agent_core/integrations/dify_webhook.py` | Dify webhook payload 归一化，把 Dify 请求转为 AgentRunRequest 形态。 |
| `src/agent_core/integrations/langsmith_exporter.py` | LangSmith trace export adapter 占位。 |

### sales_intelligence：销售实战智能层

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/sales_intelligence/README.md` | Sales Intelligence 子模块说明。 |
| `src/agent_core/sales_intelligence/__init__.py` | sales_intelligence package 入口，导出 `SalesInsightCard`。 |
| `src/agent_core/sales_intelligence/schemas.py` | 核心 Pydantic model：`SalesInsightCard`、`CustomerKYC`、`SalesInsightDigest` 和示例卡片。 |
| `src/agent_core/sales_intelligence/sales_insight_card.schema.json` | 销售洞察卡片 JSON Schema，供跨语言校验和文档展示。 |
| `src/agent_core/sales_intelligence/ingestion.py` | 原始访谈接入，支持文本和本地文本文件。 |
| `src/agent_core/sales_intelligence/anonymizer.py` | 访谈脱敏，处理手机号、邮箱、金额、姓名称谓等。 |
| `src/agent_core/sales_intelligence/cleaner.py` | 访谈清洗，去除重复空白和部分口语填充。 |
| `src/agent_core/sales_intelligence/segmenter.py` | 按销售场景切片，识别 KYC、破冰、宏观共鸣、案例、异议、计划书等场景。 |
| `src/agent_core/sales_intelligence/extractor.py` | 结构化洞察抽取，本地为确定性实现，生产应替换为 LLM + JSON Schema 校验。 |
| `src/agent_core/sales_intelligence/compliance_reviewer.py` | 销售卡片合规审查，标记 high/medium/low risk，并决定是否可用于生成。 |
| `src/agent_core/sales_intelligence/indexer.py` | 卡片文件索引器，负责保存和加载 `SalesInsightCard`。 |
| `src/agent_core/sales_intelligence/retriever.py` | Sales Intelligence 检索器，只返回可 RAG、低风险、已审核通过的卡片。 |
| `src/agent_core/sales_intelligence/reranker.py` | 销售卡片 rerank，优先已审核、低风险、和 query 匹配的卡片。 |
| `src/agent_core/sales_intelligence/evidence.py` | 将销售卡片压缩为 `SalesInsightDigest`。 |
| `src/agent_core/sales_intelligence/eval_generator.py` | 从销售卡片生成 eval case。 |
| `src/agent_core/sales_intelligence/capability_model.py` | 销售能力短板识别，例如破冰、KYC、问资产、异议处理、成交收口等。 |

### skills：Domain Skill 层

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/skills/__init__.py` | skills package 标识。 |
| `src/agent_core/skills/insurance_advisor/__init__.py` | insurance advisor skill package 标识。 |
| `src/agent_core/skills/insurance_advisor/skill.yaml` | 保险顾问 Skill 元数据，定义入口 workflow、路由、依赖和合规禁区。 |
| `src/agent_core/skills/insurance_advisor/workflow.py` | 保险顾问业务 workflow，当前实现破冰助手流程，调用 Sales Intelligence 和 Guardrails。 |
| `src/agent_core/skills/insurance_advisor/prompts/strategy_generator.md` | 策略生成 prompt 文档，描述输入、输出和硬规则。 |
| `src/agent_core/skills/insurance_advisor/README.md` | 保险顾问 Skill 说明，强调业务 Skill 不拥有通用工具、memory、tracing、recovery、cost。 |

### utils：工具函数

| 文件 | 作用 |
| --- | --- |
| `src/agent_core/utils/ids.py` | ID 生成工具，包括 `new_id` 和 `new_trace_id`。 |
| `src/agent_core/utils/time.py` | UTC 时间工具。 |
| `src/agent_core/utils/json.py` | JSON 序列化和安全解析工具。 |

## tests：测试

| 文件 | 作用 |
| --- | --- |
| `tests/test_sales_insight_schema.py` | 测试销售洞察卡片 Pydantic model 和 JSON schema。 |
| `tests/test_sales_pipeline.py` | 测试访谈 ingest、脱敏、清洗、切片、抽取流程。 |
| `tests/test_retriever_and_guardrails.py` | 测试 Sales Intelligence 检索器和输出合规 guardrail。 |
| `tests/test_tools_and_cost.py` | 测试工具注册/路由、计算器和成本预算。 |
| `tests/test_workflow_engine.py` | 测试 workflow engine 对保险请求和通用请求的路由。 |
| `tests/test_integrations_and_evals.py` | 测试 Dify payload 归一化和 rule-based evaluator。 |
| `tests/test_production_contracts.py` | 测试 workflow step contract、trace fields 和 guardrail 声明。 |
| `tests/test_trace_and_security.py` | 测试结构化状态 trace 和 prompt injection 阻断。 |
| `tests/test_rag_hybrid_memory_approval.py` | 测试 hybrid RAG、分层 memory 和 human approval queue。 |
| `tests/test_main_entry.py` | 测试 `main.py` 可以直接运行，保证新手入口不会失效。 |

## 关于“每个文件都要注释”

当前可注释文件已经补充文件级中文说明：

- Python 文件：使用文件头部中文说明和 docstring；
- YAML / YML / TOML / Shell：使用 `# 文件说明`；
- Markdown：文档本身就是中文说明；
- JSON / JSONL：格式不允许注释，否则会破坏解析，因此通过本文档和相关 schema 文档解释字段。

JSON 文件说明位置：

- `data/sales_insight_cards/example_card.json`：字段含义见 `src/agent_core/sales_intelligence/schemas.py` 和 `sales_insight_card.schema.json`；
- `evals/dataset.jsonl`：字段含义见 `src/agent_core/workflow/contracts.py` 的 `EvalCase`。

## 请求运行路径示例

保险破冰问题的运行路径：

```text
AgentRunRequest
→ WorkflowEngine
→ AgentGraph.invoke → _run_universal
→ CLASSIFY_INTENT
→ ROUTE_CAPABILITY
→ DOMAIN_WORKFLOW_ROUTING
→ SALES_INTELLIGENCE_ROUTING
→ SALES_INSIGHT_RETRIEVAL
→ BUILD_CONTEXT
→ GENERATE_RESPONSE
→ COMPLIANCE_REVIEW
→ FINAL
```

销售访谈入库路径：

```text
Raw Interview
→ ingestion
→ anonymizer
→ cleaner
→ segmenter
→ extractor
→ compliance_reviewer
→ indexer
→ retriever
→ evidence digest
→ eval_generator
```

## 当前实现边界

当前项目已经具备生产级结构和本地可运行骨架，但以下能力仍是 adapter/mock：

- FastAPI 未安装时不能直接启动 API server；
- LLM 调用尚未接真实 provider；
- Web search、news search、weather、file parser 等外部工具尚未接真实 provider；
- LangSmith 远程 trace 需要 API Key 和网络；
- Sales Insight 抽取当前是确定性本地实现，生产应替换为 LLM + JSON Schema + repair + 人工审核；
- 持久化当前以文件/内存为主，生产应接数据库、对象存储、向量库和租户隔离。

这些限制集中记录在：[known-limitations.md](known-limitations.md)
