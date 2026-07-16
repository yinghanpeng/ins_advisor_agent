# 这个测试文件专门约束“文档和注释质量”，防止项目再次退化成无描述字段或模板化注释。
import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

from agent_core.evals.feedback import HumanFeedback
from agent_core.config.runtime import (
    ApiRuntimeConfig,
    DatabaseConfig,
    InsuranceKnowledgeConfig,
    IntentRoutingConfig,
    MemoryConfig,
    ModelEndpointConfig,
    RetrievalConfig,
    RuntimeSettings,
)
from agent_core.agentic_loop.schemas import (
    ToolLoopConfig,
    ToolLoopDecision,
    ToolLoopIteration,
    ToolLoopState,
    ToolObservation,
)
from agent_core.graph.state import AgentNode, AgentState
from agent_core.intents.schemas import (
    ActiveIntentState,
    IntentCatalogEntry,
    IntentMatch,
    IntentRoutingResult,
    IntentShiftDecision,
    LLMIntentAdjudication,
)
from agent_core.memory.business_schemas import (
    Advisor,
    AdvisorProfileFact,
    AgentSessionState,
    AnalysisRun,
    CaseOutcome,
    Conversation,
    Customer,
    CustomerProfileFact,
    DifyKYCAnalysisOutput,
    GeneratedOutput,
    KYCQuestion,
    MemoryEvent,
    OpportunityCase,
    Tenant,
)
from agent_core.memory.write_policy import MemoryWriteProposal, MemoryWriteValidationResult
from agent_core.memory.recall import MemoryRecallDecision, MemoryRecallItem, MemoryRecallResult
from agent_core.memory.privacy import ConsentRequest, MemoryPrivacyRequest
from agent_core.rag.schemas import (
    DocumentMetadata,
    MetadataFilter,
    RetrievalDocument,
    RetrievalQuery,
    RetrievalResult,
)
from agent_core.recovery.fallback import RecoveryPlan
from agent_core.sales_intelligence.ingestion import RawInterview
from agent_core.sales_intelligence.schemas import (
    CorpusBatch,
    CorpusCase,
    CorpusMessage,
    CustomerKYC,
    DialoguePattern,
    SalesInsightCard,
    SalesInsightDigest,
)
from agent_core.sales_intelligence.segmenter import InterviewSegment
from agent_core.skills.insurance_advisor.knowledge import (
    InsuranceKnowledgeBundle,
    InsuranceKnowledgeItem,
)
from agent_core.skills.insurance_advisor.kyc import InsuranceKycDelta, InsuranceKycEvidence
from agent_core.tools.schemas import ToolCall, ToolPermissionSpec, ToolResult, ToolSpec
from agent_core.workflow.contracts import (
    AgentRunRequest,
    AgentRunResponse,
    EvalCase,
    PublicAgentRunResponse,
    StepRetryPolicy,
    WorkflowContract,
    WorkflowStepContract,
)


# PYTHON_PATHS 覆盖 main、evals、src、tests，确保注释质量约束扫描整个项目 Python 文件。
PYTHON_PATHS = [Path("main.py"), Path("evals/run_evals.py"), *Path("src").rglob("*.py"), *Path("tests").rglob("*.py")]

# RUNTIME_COMMENT_PATHS 扫描入口、评估 Runner、生产代码和运维脚本；测试断言本身不递归扫描。
RUNTIME_COMMENT_PATHS = [
    Path("main.py"),
    Path("evals/run_evals.py"),
    *Path("src/agent_core").rglob("*.py"),
    *Path("scripts").glob("*.py"),
]

# CONTROL_FLOW_NODES 是必须紧邻解释的控制语句集合；import、括号和纯数据字面量不做机械注释。
CONTROL_FLOW_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Match,
)

# LINE_LEVEL_STATEMENT_NODES 覆盖会产生状态、输出或流程副作用的普通语句。
LINE_LEVEL_STATEMENT_NODES = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Delete,
    ast.Return,
    ast.Assert,
    ast.Break,
    ast.Continue,
    ast.Pass,
)

# CONTROL_BOUNDARY_PATTERN 补足 AST 不稳定或不会单独暴露行号的各类分支边界。
CONTROL_BOUNDARY_PATTERN = re.compile(
    r"^(?:elif\b.*:|else\s*:|except\b.*:|finally\s*:|case\b.*:)$"
)

# CHINESE_COMMENT_PATTERN 确保新增解释是团队主要使用的中文，而不是无意义的英文占位词。
CHINESE_COMMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def _assignment_target_names(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> list[str]:
    """提取赋值目标名称，用于识别模块级纯声明常量。"""

    # 普通赋值可能同时绑定多个目标；注解赋值和增量赋值只有一个目标。
    raw_targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    # names 保存递归展开 tuple/list 解包后的标识符。
    names: list[str] = []

    # visit_target 只提取 Name；属性和下标赋值属于运行时状态修改，不能作为常量豁免。
    def visit_target(target: ast.expr) -> None:
        """递归收集一个赋值目标中的简单变量名。"""

        # 简单变量名可以用于判断全大写常量或双下划线模块元数据。
        if isinstance(target, ast.Name):
            names.append(target.id)
            return
        # tuple/list 解包继续逐项展开；其它目标类型保持不收集。
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                visit_target(item)

    # 遍历所有原始目标，形成稳定的名称集合。
    for raw_target in raw_targets:
        visit_target(raw_target)
    # 返回提取结果；空列表表示不能证明它是声明式常量。
    return names


def _is_described_pydantic_field(node: ast.AST) -> bool:
    """判断赋值是否已由 Field(description=...) 在同一契约行完成解释。"""

    # Pydantic 字段通常使用注解赋值；也兼容普通赋值写法。
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    # 两种赋值节点都暴露 value；空值注解字段不满足豁免条件。
    value = node.value
    if not isinstance(value, ast.Call):
        return False
    # 同时识别 Field(...) 和 pydantic.Field(...)，避免绑定方式影响检查结果。
    function_name = value.func.id if isinstance(value.func, ast.Name) else ""
    attribute_name = value.func.attr if isinstance(value.func, ast.Attribute) else ""
    if function_name != "Field" and attribute_name != "Field":
        return False
    # 只有非空 description 才等价于逐行字段说明；缺描述仍交给门禁报错。
    for keyword in value.keywords:
        if keyword.arg == "description" and isinstance(keyword.value, ast.Constant):
            return isinstance(keyword.value.value, str) and bool(keyword.value.value.strip())
    # 没有 description 的 Field 不豁免。
    return False


def _inline_comments_by_line(source: str) -> dict[int, str]:
    """使用 Python tokenizer 提取真实注释，避免把字符串中的 # 误判为行尾注释。"""

    # comments 以一基行号索引注释正文，同一物理行通常只有一个 COMMENT token。
    comments: dict[int, str] = {}
    # tokenize 理解 Python 字符串和转义边界，只有 COMMENT token 才是真实源码注释。
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        # 其它 token 即使文本包含 # 也可能只是字符串内容，不能用于注释门禁。
        if token.type != tokenize.COMMENT:
            continue
        # 去掉开头 # 和首尾空白，保留中文正文供统一匹配。
        comments[token.start[0]] = token.string[1:].strip()
    # 返回按物理行索引的真实注释表。
    return comments

# PYDANTIC_MODELS 是所有必须具备 Field(description=...) 的结构化契约模型清单。
PYDANTIC_MODELS = [
    # AgentState 是主链路状态黑匣子，字段描述必须完整。
    AgentState,
    # Workflow contract 模型约束 step 输入、输出、风控、重试和 trace。
    StepRetryPolicy,
    WorkflowStepContract,
    WorkflowContract,
    AgentRunRequest,
    AgentRunResponse,
    PublicAgentRunResponse,
    EvalCase,
    # Intent Layer schema 约束向量候选、LLM 裁定、活跃状态和最终分发结果。
    IntentCatalogEntry,
    IntentMatch,
    LLMIntentAdjudication,
    IntentShiftDecision,
    ActiveIntentState,
    IntentRoutingResult,
    # Insurance Handler schema 约束本轮 KYC 增量和双知识库上下文。
    InsuranceKycEvidence,
    InsuranceKycDelta,
    InsuranceKnowledgeItem,
    InsuranceKnowledgeBundle,
    # Runtime Config schema 保证中间件、模型、向量库、Redis 和阈值配置都有可读说明。
    ModelEndpointConfig,
    DatabaseConfig,
    RetrievalConfig,
    IntentRoutingConfig,
    InsuranceKnowledgeConfig,
    MemoryConfig,
    ApiRuntimeConfig,
    RuntimeSettings,
    # 隐私请求 schema 约束用户导出、删除和用途级同意字段。
    MemoryPrivacyRequest,
    ConsentRequest,
    # Tool schema 模型约束工具权限、风险、调用和结果。
    ToolPermissionSpec,
    ToolSpec,
    ToolCall,
    ToolResult,
    # Agentic 工具循环 schema 约束 planner、observation 和循环预算。
    ToolLoopConfig,
    ToolLoopDecision,
    ToolObservation,
    ToolLoopIteration,
    ToolLoopState,
    # RAG schema 模型约束 query、metadata、document、result 和 filters。
    RetrievalQuery,
    DocumentMetadata,
    RetrievalDocument,
    RetrievalResult,
    MetadataFilter,
    # Sales Intelligence schema 模型约束 KYC、洞察卡片和摘要。
    CustomerKYC,
    SalesInsightCard,
    SalesInsightDigest,
    CorpusBatch,
    CorpusCase,
    CorpusMessage,
    DialoguePattern,
    # 业务记忆 schema 模型约束租户、顾问、客户、事实、case、会话、分析、输出和结果闭环。
    Tenant,
    Advisor,
    Customer,
    AdvisorProfileFact,
    CustomerProfileFact,
    OpportunityCase,
    Conversation,
    AgentSessionState,
    KYCQuestion,
    DifyKYCAnalysisOutput,
    AnalysisRun,
    GeneratedOutput,
    MemoryEvent,
    CaseOutcome,
    MemoryWriteProposal,
    MemoryWriteValidationResult,
    MemoryRecallDecision,
    MemoryRecallItem,
    MemoryRecallResult,
    # 访谈导入和分段模型约束销售语料资产化入口。
    RawInterview,
    InterviewSegment,
    # 反馈和恢复模型约束评估与 recovery 策略。
    HumanFeedback,
    RecoveryPlan,
]


def test_all_pydantic_fields_have_business_descriptions() -> None:
    """所有 Pydantic 字段都必须写业务含义，避免 API schema 变成只有字段名的空壳。"""
    # missing 收集缺少 description 的字段名，最后统一报错方便一次性修复。
    missing: list[str] = []
    # 遍历所有核心 Pydantic 模型。
    for model in PYDANTIC_MODELS:
        # model_fields 是 Pydantic v2 暴露的字段定义集合。
        for field_name, field in model.model_fields.items():
            # description 为空说明该字段没有业务解释。
            if not field.description or not field.description.strip():
                missing.append(f"{model.__name__}.{field_name}")
    # 任何字段缺描述都直接失败，防止契约文档空心化。
    assert not missing, "缺少 Field(description=...) 的字段：" + ", ".join(missing)


def test_no_template_style_generated_comments_remain() -> None:
    """禁止重新引入模板化坏注释。"""
    # forbidden 故意拆字符串，避免测试文件自身被搜索命中。
    forbidden = ["方法" + "说明：", "类" + "说明：", "职责说明；" + "具体输入输出"]
    # offenders 收集包含模板化注释的文件路径。
    offenders: list[str] = []
    # 扫描项目 Python 文件，防止机械生成注释重新进入代码库。
    for path in PYTHON_PATHS:
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden):
            offenders.append(str(path))
    # 发现模板化注释时直接失败，并列出文件。
    assert not offenders, "存在模板化注释：" + ", ".join(offenders)


def test_runtime_control_flow_has_adjacent_line_level_comments() -> None:
    """生产控制语句必须有紧邻中文解释，防止只写文件头而关键分支无注释。"""

    # offenders 保存文件、语句行和缺少注释的源码摘要，失败时可以直接定位到行。
    offenders: list[str] = []
    # 逐个解析生产 Python 文件；AST 能跳过括号续行和数据字面量，避免基于文本误报。
    for path in RUNTIME_COMMENT_PATHS:
        # 已删除的兼容文件可能仍出现在 Git 差异中，磁盘不存在时不参与当前代码审计。
        if not path.exists():
            continue
        # 保留原始行用于查找语句前最近的非空注释和生成精确错误位置。
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        # tokenizer 提取真实行内注释，字符串中的 # 不会覆盖上一行中文说明。
        inline_comments = _inline_comments_by_line(source)
        # ast.parse 失败会由 compile/test 更早暴露；这里专注检查成功解析后的控制语句。
        tree = ast.parse(source, filename=str(path))
        # ast.walk 覆盖顶层、函数、方法和嵌套分支，不能只检查公开函数第一层。
        for node in ast.walk(tree):
            # 赋值、import 和 return 不属于分支控制；它们由字段 description、docstring 和代码契约解释。
            if not isinstance(node, CONTROL_FLOW_NODES):
                continue
            # 控制语句行内注释也属于精确到行的解释，优先检查同一物理行。
            statement_line = lines[node.lineno - 1]
            inline_comment = inline_comments.get(node.lineno, "")
            # 没有行内注释时向上跳过空行，只接受最近非空行的独立注释。
            previous_index = node.lineno - 2
            while previous_index >= 0 and not lines[previous_index].strip():
                previous_index -= 1
            # 独立注释必须以 # 开头；上一行代码上的更早泛化注释不能覆盖当前控制语句。
            previous_comment = ""
            if previous_index >= 0 and lines[previous_index].lstrip().startswith("#"):
                previous_comment = lines[previous_index].lstrip()[1:].strip()
            # 接受行内或紧邻上一行的中文解释；两者都没有时记录精确文件和行号。
            explanatory_comment = inline_comment or previous_comment
            if not CHINESE_COMMENT_PATTERN.search(explanatory_comment):
                offenders.append(
                    f"{path}:{node.lineno} {statement_line.strip()}"
                )
    # 一次列出全部遗漏，修改者可以按行补齐而不是反复运行只看到第一个错误。
    assert not offenders, "控制语句缺少紧邻中文注释：\n" + "\n".join(offenders)


def test_runtime_effectful_statements_have_adjacent_line_level_comments() -> None:
    """生产赋值、调用和返回必须有逐行中文意图说明。"""

    # offenders 汇总所有有业务效果但没有相邻解释的语句，便于一次性按行修完。
    offenders: list[str] = []
    # 遍历与控制流门禁相同的生产文件集合，避免测试代码扫描测试自身造成递归约束。
    for path in RUNTIME_COMMENT_PATHS:
        # 已删除的兼容文件不属于当前可执行代码，直接跳过。
        if not path.exists():
            continue
        # 同时保留源码行和 AST，以便判断语句类型并输出精确物理行。
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        # 使用 tokenizer 区分真实注释与字符串字面量内的 # 字符。
        inline_comments = _inline_comments_by_line(source)
        tree = ast.parse(source, filename=str(path))
        # 模块直属赋值中的全大写常量和 __all__/__version__ 是声明式数据，不要求机械复述。
        module_statement_ids = {id(statement) for statement in tree.body}
        # 遍历所有嵌套语句，覆盖私有 helper、异常恢复分支和工厂函数内部闭包。
        for node in ast.walk(tree):
            # 普通赋值/返回等语句直接纳入；独立表达式只检查调用、await 和 yield 副作用。
            is_effectful_expression = isinstance(node, ast.Expr) and isinstance(
                node.value,
                (ast.Call, ast.Await, ast.Yield, ast.YieldFrom),
            )
            # 不产生运行效果的数据字面量、import 和声明行不机械增加噪声注释。
            if not isinstance(node, LINE_LEVEL_STATEMENT_NODES) and not is_effectful_expression:
                continue
            # Field(description=...) 已在该契约行精确描述业务语义，不再重复写同义行注释。
            if _is_described_pydantic_field(node):
                continue
            # 模块级全大写常量和双下划线元数据属于纯声明数据，文件附近的分组注释即可。
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and id(node) in module_statement_ids:
                target_names = _assignment_target_names(node)
                if target_names and all(
                    name.isupper() or (name.startswith("__") and name.endswith("__"))
                    for name in target_names
                ):
                    continue
            # 先读取同一行尾注释；它可以精确解释简单的一行状态更新。
            statement_line = lines[node.lineno - 1]
            inline_comment = inline_comments.get(node.lineno, "")
            # 没有行尾注释时，只接受向上跳过空行后的最近一行独立注释。
            previous_index = node.lineno - 2
            while previous_index >= 0 and not lines[previous_index].strip():
                previous_index -= 1
            # 上一非空行必须本身就是注释，防止一个远处的函数总述覆盖多条执行语句。
            previous_comment = ""
            if previous_index >= 0 and lines[previous_index].lstrip().startswith("#"):
                previous_comment = lines[previous_index].lstrip()[1:].strip()
            # 行内或紧邻注释至少包含中文，排除单纯 noqa/type-ignore 等机器标记。
            if not CHINESE_COMMENT_PATTERN.search(inline_comment or previous_comment):
                offenders.append(f"{path}:{node.lineno} {statement_line.strip()}")
        # elif/else/except/finally/case 没有统一的独立 statement 节点，因此额外按源码边界检查。
        for line_number, statement_line in enumerate(lines, start=1):
            # 只识别去掉缩进后的分支边界，注释和字符串内容不会以这些完整语法行出现。
            if not CONTROL_BOUNDARY_PATTERN.match(statement_line.strip()):
                continue
            # 分支边界同样要求最近的上一非空行是中文解释。
            previous_index = line_number - 2
            while previous_index >= 0 and not lines[previous_index].strip():
                previous_index -= 1
            # 提取独立注释正文；不存在相邻注释时保留空字符串并统一报错。
            previous_comment = ""
            if previous_index >= 0 and lines[previous_index].lstrip().startswith("#"):
                previous_comment = lines[previous_index].lstrip()[1:].strip()
            # 缺少中文说明时记录具体分支行，避免异常路径成为注释盲区。
            if not CHINESE_COMMENT_PATTERN.search(previous_comment):
                offenders.append(f"{path}:{line_number} {statement_line.strip()}")
    # 聚合报告所有遗漏，后续提交也必须维持逐行注释基线。
    assert not offenders, "有效执行语句缺少紧邻中文注释：\n" + "\n".join(offenders)


def test_runtime_classes_and_functions_have_docstrings() -> None:
    """生产类和函数必须说明职责，不能只依赖调用方猜测名称。"""

    # offenders 记录所有缺少 docstring 的类、同步函数和异步函数及其准确声明行。
    offenders: list[str] = []
    # 逐文件解析 AST，覆盖公开 API、私有 helper、嵌套 FastAPI endpoint 和运维脚本入口。
    for path in RUNTIME_COMMENT_PATHS:
        # 跳过磁盘上已经删除的兼容文件。
        if not path.exists():
            continue
        # AST docstring 判断只接受声明体第一条字符串，普通行注释不能冒充函数契约。
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # 每个生产模块都要先说明文件职责，空 __init__.py 也不能依赖目录名猜测用途。
        if ast.get_docstring(tree, clean=True) is None:
            offenders.append(f"{path}:1 <module>")
        # 遍历嵌套声明，确保工厂函数内部定义的 HTTP endpoint 也有接口说明。
        for node in ast.walk(tree):
            # 只检查具备独立职责边界的类和函数声明。
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # ast.get_docstring 能正确识别普通和多行 docstring；缺失时记录声明位置。
            if ast.get_docstring(node, clean=True) is None:
                offenders.append(f"{path}:{node.lineno} {node.name}")
    # 一次报告所有遗漏，方便按模块补全而不是只修首个函数。
    assert not offenders, "类或函数缺少 docstring：\n" + "\n".join(offenders)


def test_agent_state_move_to_uses_allowed_transitions_guard() -> None:
    """move_to 是状态切换入口，并已预留 allowed_transitions 合法跳转校验。"""
    # 构造一个只允许 IDLE -> CLASSIFY_INTENT -> ROUTE_CAPABILITY 的状态白名单。
    state = AgentState(
        allowed_transitions={
            AgentNode.IDLE.value: [AgentNode.CLASSIFY_INTENT.value],
            AgentNode.CLASSIFY_INTENT.value: [AgentNode.ROUTE_CAPABILITY.value],
        }
    )
    # 合法跳转应成功，并写入 state_transitions。
    state.move_to(AgentNode.CLASSIFY_INTENT, reason="allowed_by_contract")
    assert state.current_state == AgentNode.CLASSIFY_INTENT
    assert state.state_transitions[-1]["to_state"] == AgentNode.CLASSIFY_INTENT.value

    # 非法跳转应被 move_to 阻断，证明状态切换唯一入口具备校验预留。
    with pytest.raises(ValueError):
        state.move_to(AgentNode.FINAL, reason="not_allowed_by_contract")
