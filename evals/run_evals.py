"""Run isolated local Agent evaluations and emit machine-readable reports."""

# 文件说明：
# - 本文件是本地 Eval Harness，始终通过 WorkflowEngine.run 执行正式 Agent 链路。
# - 普通 Case 和独立 Trial 互相隔离；同一个多轮 Trial 才复用 Engine、Session 和 Memory。
# - 支持并行 Case、干跑、可选原生/DeepEval Judge、报告、基线对比与 Promote 门禁。
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import yaml

# ROOT 指向项目根目录，保证从 evals/ 子目录或仓库根目录运行时使用同一数据路径。
ROOT = Path(__file__).resolve().parents[1]
# 将 src 加入导入路径，避免要求调用者先执行 pip install -e .。
sys.path.insert(0, str(ROOT / "src"))

# LLM 路由来源前缀：intent_routing_result.source 以这些开头时表示本轮调用了意图相关模型。
_LLM_ROUTING_SOURCE_PREFIXES = (
    "llm_candidate_adjudication",
    "llm_open_set",
    "llm_open_set_after_vector_error",
    # 活跃意图切换/取消判断也会调用独立小模型。
    "active_intent_shift_model",
)
# 确定性/向量直达来源：明确表示本轮意图裁定没有调用 LLM。
_NON_LLM_ROUTING_SOURCE_PREFIXES = (
    "vector_direct",
    "deterministic_fallback",
    "deterministic_fallback_after_vector_error",
    "multi_intent_execution_plan",
    # 纯续接活跃意图、未触发 shift 模型时不算意图 LLM。
    "active_intent",
)

# 以下导入依赖 ROOT 路径初始化，因此放在 sys.path 调整之后。
from agent_core.evals.evaluators import evaluate_case  # noqa: E402
from agent_core.observability.langsmith_client import LangSmithAdapter  # noqa: E402
from agent_core.observability.logger import StructuredLogger  # noqa: E402
from agent_core.utils.time import utc_now_iso  # noqa: E402
from agent_core.workflow.contracts import (  # noqa: E402
    AgentRunExecutionContext,
    AgentRunRequest,
    EvalCase,
)
from agent_core.workflow.engine import WorkflowEngine  # noqa: E402


# EngineFactory 允许测试注入轻量假 Engine，同时生产 Runner 默认构造真实 WorkflowEngine。
EngineFactory = Callable[[], WorkflowEngine]
# JudgeClientFactory 允许测试注入假 Judge，生产环境按需从配置创建真实客户端。
JudgeClientFactory = Callable[[], Any]


class _SilentLogger(StructuredLogger):
    """Eval 默认不输出逐节点日志，完整 Trace 仍保留在 AgentRunResponse。"""

    def event(self, event: str, **fields: Any) -> None:
        """丢弃普通控制台事件，避免数十个 Case 产生难以阅读的海量日志。"""

        # 显式消费参数，表明静默是 Eval Harness 的预期策略而不是遗漏实现。
        _ = event, fields

    def warning(self, event: str, **fields: Any) -> None:
        """丢弃逐节点告警；Trial 异常会作为结构化失败写入报告。"""

        # 显式消费参数，保持与 StructuredLogger 的调用契约一致。
        _ = event, fields


def load_dotenv(path: Path | None = None, *, override: bool = True) -> dict[str, Any]:
    """把仓库根目录 .env 加载进进程环境，供 models.yaml 的 ${ENV} 插值使用。

    Eval 本地默认 override=True：以 .env 文件为准，避免 IDE/Shell 里残留的旧
    OPENAI_BASE_URL 覆盖你真正配置的网关。CI 可用 --no-dotenv-override 保留进程环境优先。
    只解析 KEY=VALUE / export KEY=VALUE，不执行 shell 插值，避免 .env 变成脚本。
    """

    # env_path 默认指向仓库根 .env，与 Makefile 约定一致。
    env_path = path or (ROOT / ".env")
    # 文件不存在时返回空结果，本地无 .env 的纯 CI 环境仍可继续跑。
    if not env_path.exists():
        # loaded=False 让摘要明确区分“没找到文件”和“文件存在但无新键”。
        return {
            "loaded": False,
            "path": str(env_path),
            "set_count": 0,
            "skipped_existing": 0,
            "overridden_count": 0,
            "override": override,
        }
    # set_count / overridden_count / skipped_existing 只统计数量，绝不回显密钥值。
    set_count = 0
    overridden_count = 0
    skipped_existing = 0
    # 逐行解析，保持与常见 dotenv 工具相近的可读错误边界。
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        # line 去掉首尾空白；空行与注释不参与加载。
        line = raw_line.strip()
        # 空行与 # 注释是人工分组，直接跳过。
        if not line or line.startswith("#"):
            # 当前行没有可加载内容。
            continue
        # 兼容 `export KEY=VALUE` 写法，与 bash source 习惯一致。
        if line.startswith("export "):
            # 去掉 export 前缀后再按 KEY=VALUE 解析。
            line = line[len("export ") :].strip()
        # 没有等号的行无法形成环境变量，属于 .env 格式错误。
        if "=" not in line:
            # 带行号抛出，方便用户定位坏行；不回显可能含密钥的整行。
            raise ValueError(f"无效 .env 行（缺少 '='）: {env_path}:{line_number}")
        # key/value 只按第一个 '=' 分割，允许值里继续出现 '='。
        key, value = line.split("=", 1)
        # key 必须是非空标识符。
        key = key.strip()
        # 空键名无法写入 os.environ。
        if not key:
            # 与缺少 '=' 一样视为格式错误。
            raise ValueError(f"无效 .env 行（空键名）: {env_path}:{line_number}")
        # value 去掉包裹引号，兼容 KEY="secret" / KEY='secret'。
        value = value.strip()
        if (len(value) >= 2) and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            # 去掉成对引号，保留内部原文。
            value = value[1:-1]
        # 已存在且不允许覆盖时跳过，保护 CI/shell 预先 export 的值。
        if (not override) and key in os.environ:
            # 计数后继续下一键。
            skipped_existing += 1
            continue
        # 覆盖模式下统计“旧值被 .env 替换”的次数，便于确认读到了文件而不是残留环境。
        if override and key in os.environ and os.environ.get(key) != value:
            # 仅计数，不记录旧值或新值内容。
            overridden_count += 1
        # 写入进程环境，供后续 load_runtime_settings 的 ${ENV} 插值使用。
        os.environ[key] = value
        # 成功写入计数 +1。
        set_count += 1
    # models.yaml 使用 LLM_*；.env 若只配了 OPENAI_*，在此显式同步，避免插值拿到空串。
    alias_synced = _sync_llm_aliases_from_openai()
    # 返回加载摘要；调用方只应打印计数，不要打印具体键值。
    return {
        "loaded": True,
        "path": str(env_path),
        "set_count": set_count,
        "skipped_existing": skipped_existing,
        "overridden_count": overridden_count,
        "override": override,
        "alias_synced": alias_synced,
    }


def _sync_llm_aliases_from_openai() -> list[str]:
    """当 LLM_* 未配置时，用 OPENAI_* 填入，对齐 models.yaml 占位符。"""

    # synced 记录实际补齐的目标变量名，供摘要展示（不含值）。
    synced: list[str] = []
    # pairs 与 runtime.ENV_FALLBACK_ALIASES 方向相反：这里是把别名写入主名。
    pairs = (
        ("LLM_BASE_URL", "OPENAI_BASE_URL"),
        ("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    # 逐对检查：主名为空且别名有值时才写入，不覆盖用户显式 LLM_*。
    for primary, alias in pairs:
        # primary_value 为空或仅空白时视为未配置。
        primary_value = (os.environ.get(primary) or "").strip()
        # alias_value 来自 .env 中的 OpenAI 兼容命名。
        alias_value = (os.environ.get(alias) or "").strip()
        # 只有主名缺失、别名存在时才同步。
        if (not primary_value) and alias_value:
            # 写入主名，使 ${LLM_BASE_URL}/${LLM_API_KEY} 插值直接命中。
            os.environ[primary] = alias_value
            # 记录同步过的主名。
            synced.append(primary)
    # 返回同步列表；空列表表示无需补齐。
    return synced


def resolve_model_endpoints_summary() -> dict[str, Any]:
    """读取插值后的模型端点摘要（脱敏），确认 Eval 用的是 .env 里的配置。"""

    # 延迟导入，避免未加载 dotenv 前就触发配置解析。
    from agent_core.config.runtime import load_runtime_settings

    # settings 此时应已能看到 .env 注入后的环境变量。
    settings = load_runtime_settings(ROOT / "configs")
    # endpoints 按职责列出模型名与 base_url 主机，永不输出 api_key。
    endpoints: dict[str, Any] = {}
    # 关注 Eval 最常用的几个端点；缺失时标记 unavailable。
    for name in (
        "default_chat",
        "fast_reasoning",
        "intent_classifier",
        "guardrail",
        "insurance_kyc_extractor",
    ):
        # model_cfg 可能因环境未配齐而不存在。
        model_cfg = settings.models.get(name)
        if model_cfg is None:
            # 明确标 unavailable，方便用户对照 .env。
            endpoints[name] = {"status": "unavailable"}
            continue
        # base_url 只保留 scheme+host，去掉 path/query，避免泄露内部路由细节。
        base_url = str(model_cfg.base_url or "").strip()
        host = base_url
        if "://" in base_url:
            # 粗粒度截取 host：去掉 scheme 后取第一段路径前内容。
            host = base_url.split("://", 1)[1].split("/", 1)[0]
        # api_key_configured 只报告是否非空，不回显任何字符。
        endpoints[name] = {
            "status": "ok" if base_url and model_cfg.api_key and model_cfg.model else "incomplete",
            "model": model_cfg.model or None,
            "base_url_host": host or None,
            "api_key_configured": bool(str(model_cfg.api_key or "").strip()),
        }
    # 返回可直接写入报告摘要的结构。
    return {"config_dir": str(ROOT / "configs"), "endpoints": endpoints}


def extract_routing_diagnostics(response: Any) -> dict[str, Any]:
    """从 AgentRunResponse 提取意图路由诊断，标明本轮是否真正调用了意图 LLM。"""

    # routing 可能是 dict（正式响应）或空；假 Engine 测试也可能不填。
    routing = getattr(response, "intent_routing_result", None) or {}
    if not isinstance(routing, dict):
        # 非字典时降级为空诊断，避免报告序列化失败。
        routing = {}
    # source 是 Router 写入的稳定来源码，例如 vector_direct / llm_open_set / deterministic_fallback。
    source = str(routing.get("source") or "")
    # reason_code 补充短原因，例如 high_similarity_fixed_intent / model_unavailable_rule_fallback。
    reason_code = str(routing.get("reason_code") or "")
    # used_llm_intent：source 命中 LLM 前缀即为真；其余已知非 LLM 来源为假；未知则 None。
    used_llm_intent: bool | None
    if any(source.startswith(prefix) for prefix in _LLM_ROUTING_SOURCE_PREFIXES):
        # 明确调用了意图裁定模型。
        used_llm_intent = True
    elif any(source.startswith(prefix) for prefix in _NON_LLM_ROUTING_SOURCE_PREFIXES) or not source:
        # 向量直达、规则兜底、或尚未分类时都视为未调用意图 LLM。
        used_llm_intent = False
    else:
        # 新产品来源码尚未纳入表时保持未知，避免误报。
        used_llm_intent = None
    # 只返回控制面字段，不回显 slots 等可能含客户事实的内容。
    return {
        "source": source or None,
        "reason_code": reason_code or None,
        "vector_score": routing.get("vector_score"),
        "confidence": routing.get("confidence"),
        "dispatch_action": routing.get("dispatch_action"),
        "used_llm_intent": used_llm_intent,
    }


def _routing_summary(case_results: list[dict]) -> dict[str, Any]:
    """汇总整次 Eval 中意图路由是否真正打到 LLM。"""

    # source_counts 统计各 source 出现次数，便于一眼看出全是 fallback 还是有 llm_*。
    source_counts: dict[str, int] = {}
    # llm_trials / non_llm_trials / unknown_trials 按 Trial 计数。
    llm_trials = 0
    non_llm_trials = 0
    unknown_trials = 0
    # 遍历全部 Case 的全部 Trial。
    for case_result in case_results:
        # dry_run 没有 trials 时跳过。
        for trial in case_result.get("trials") or []:
            # routing 由 run_case 写入；异常 Trial 可能只有空诊断。
            routing = trial.get("routing") or {}
            # source_key 把空来源记为 unknown，避免 Counter 键为 None。
            source_key = str(routing.get("source") or "unknown")
            # 累加该来源出现次数。
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            # used 是布尔或 None。
            used = routing.get("used_llm_intent")
            if used is True:
                # 意图 LLM 实际被调用。
                llm_trials += 1
            elif used is False:
                # 向量直达或规则兜底。
                non_llm_trials += 1
            else:
                # 未知来源码。
                unknown_trials += 1
    # 返回机器可读汇总，写入报告顶层方便 CI grep。
    return {
        "llm_intent_trials": llm_trials,
        "non_llm_intent_trials": non_llm_trials,
        "unknown_routing_trials": unknown_trials,
        "sources": dict(sorted(source_counts.items())),
    }


def load_dataset(path: Path) -> list[EvalCase]:
    """读取 JSONL 数据集并在执行前校验 Schema、规则 ID 和 Case ID 唯一性。"""

    # cases 按文件顺序保存，报告和失败复现使用同一稳定顺序。
    cases: list[EvalCase] = []
    # 逐行解析可以把错误精确定位到 JSONL 行号。
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        # 空行只用于人工分组，不创建空 Case。
        if not line.strip():
            # 当前循环已经达到停止条件，立即跳过空行。
            continue
        # JSON 和 Pydantic 错误都补充数据集路径与行号后继续抛出，避免静默跳过脏数据。
        try:
            # case 使用 EvalCase 完整校验字段类型、受控 initial_state 和评分器 ID。
            case = EvalCase.model_validate(json.loads(line))
        # 任一解析异常都属于数据集错误，不应被记成 Agent 能力失败。
        except Exception as exc:
            # 重新抛出带精确行号的 ValueError，同时保留原始异常链供调试。
            raise ValueError(f"无效 Eval Case: {path}:{line_number}: {exc}") from exc
        # 将已校验 Case 加入稳定顺序列表。
        cases.append(case)
    # ids 收集全部稳定主键，用于发现报告覆盖和基线比较会产生歧义的重复项。
    ids = [case.id for case in cases]
    # duplicate_ids 只保留重复 ID 并排序，错误信息在本地和 CI 中保持一致。
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    # 数据集不允许重复主键，否则失败报告无法唯一关联原始 Case。
    if duplicate_ids:
        # 抛出结构化可读错误，不执行任何 Agent Trial。
        raise ValueError(f"Eval Case ID 重复: {', '.join(duplicate_ids)}")
    # 返回完整校验后的 Case 列表。
    return cases


def load_gate_config(path: Path | None) -> dict[str, Any]:
    """加载 Promote 门禁配置；缺省使用仓库内 configs/eval_gates.yaml。"""

    # config_path 优先使用调用方指定路径，否则回落到仓库默认门禁文件。
    config_path = path or (ROOT / "configs" / "eval_gates.yaml")
    # 文件不存在时返回空配置，门禁检查会使用内置保守默认值。
    if not config_path.exists():
        # 空字典表示“未提供显式阈值”。
        return {}
    # 使用 YAML 安全加载，禁止任意对象反序列化。
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # 非字典配置视为无效，避免把列表误当成门禁树。
    if not isinstance(loaded, dict):
        # 明确失败，防止 CI 用坏配置放行。
        raise ValueError(f"Eval 门禁配置必须是映射对象: {config_path}")
    # 返回原始字典，后续读取时再取默认值。
    return loaded


def _default_engine_factory() -> WorkflowEngine:
    """创建一个不上传远程 Trace、也不输出逐节点日志的真实 Eval Engine。"""

    # logger 保持 WorkflowEngine 日志接口，但把控制台噪音留给结构化 Eval 报告替代。
    logger = _SilentLogger()
    # langsmith 明确关闭，避免离线回归集把合成或脱敏样本上传到远程项目。
    langsmith = LangSmithAdapter(enabled=False)
    # 每次调用都返回新 Engine，从而隔离 Memory、Business Store、Registry Cache 和会话状态。
    return WorkflowEngine(log=logger, langsmith=langsmith)


def _default_judge_client_factory() -> Any:
    """按运行时配置创建真实 Judge 客户端；测试可注入替身覆盖此工厂。"""

    # 延迟导入，避免未启用 Judge 时强依赖模型配置模块。
    from agent_core.config.runtime import load_runtime_settings
    from agent_core.models.client import OpenAICompatibleChatClient

    # settings 读取仓库 configs，优先使用 guardrail 端点，缺失时回退 fast_reasoning。
    settings = load_runtime_settings("configs")
    try:
        # 优先专用风控模型，与线上 LLM Judge 护栏保持一致。
        config = settings.require_model("guardrail")
    except Exception:
        # 没有专用端点时退回轻量快速推理模型。
        config = settings.require_model("fast_reasoning")
    # 返回 OpenAI Compatible 客户端供 complete_json 调用。
    return OpenAICompatibleChatClient(config)


def _turn_inputs(case: EvalCase) -> list[str]:
    """返回 Case 实际执行的用户输入序列。"""

    # turns 非空时代表完整多轮场景；否则把兼容 input 包装成单轮列表。
    return list(case.turns) if case.turns else [case.input]


def _request_for_turn(case: EvalCase, input_text: str, trial_number: int) -> AgentRunRequest:
    """把受控 Case 夹具映射成正式 AgentRunRequest，不绕过公开请求校验。"""

    # initial 保存经过 EvalCase allowlist 校验的请求级夹具副本。
    initial = dict(case.initial_state)
    # metadata 从数据集显式字段复制，后续仍由 AgentRunRequest 执行 default-deny 校验。
    metadata = dict(initial.get("metadata") or {})
    # source 是常用观测标签的简写，映射到公开允许的 metadata.source。
    if initial.get("source") is not None:
        # 只写入字符串化来源标签，不允许通过简写注入其它 metadata 键。
        metadata["source"] = str(initial["source"])
    # eval_id 关联本次运行与稳定 Case 主键；调用方自定义值不能覆盖它。
    metadata["eval_id"] = case.id
    # 默认 Session 同时包含 Case 和 Trial，确保独立重复运行不会共享短期记忆。
    default_session_id = f"eval:{case.id}:trial:{trial_number}"
    # 默认 Tenant 按 Case 隔离，避免不同 Case 的 Artifact 或业务记忆互相可见。
    default_tenant_id = f"eval:{case.id}"
    # 使用正式请求模型构造输入，任何受保护 metadata 都会在这里立即失败。
    return AgentRunRequest(
        input=input_text,
        session_id=str(initial.get("session_id") or default_session_id),
        user_id=(str(initial["user_id"]) if initial.get("user_id") is not None else None),
        tenant_id=str(initial.get("tenant_id") or default_tenant_id),
        workflow_name=str(initial.get("workflow_name") or "universal_agent_workflow"),
        domain_skill=(
            str(initial["domain_skill"])
            if initial.get("domain_skill") is not None
            else None
        ),
        metadata=metadata,
    )


def _execution_context(case: EvalCase) -> AgentRunExecutionContext:
    """把受控请求预算夹具转换为代码侧执行上下文。"""

    # raw_budget 只可能来自 EvalCase allowlist 中的 request_token_budget。
    raw_budget = case.initial_state.get("request_token_budget")
    # Pydantic 负责验证预算为正整数；None 表示继续使用正式运行默认值。
    return AgentRunExecutionContext(request_token_budget=raw_budget)


def _answer_preview(answer: str, *, include_answers: bool, limit: int = 500) -> str | None:
    """按开关决定是否把回答摘要写入本地报告。"""

    # 关闭时返回 None，保持默认隐私友好策略。
    if not include_answers:
        # 调用方不应序列化缺失字段为完整正文。
        return None
    # 截断过长回答，避免报告文件膨胀。
    text = answer.strip()
    # 空回答也显式记录，便于区分“没生成”和“未落盘”。
    if len(text) <= limit:
        # 短回答原样返回。
        return text
    # 超长时追加省略标记。
    return text[:limit] + "…(truncated)"


def run_case(
    case: EvalCase,
    engine_factory: EngineFactory = _default_engine_factory,
    *,
    max_trials: int | None = None,
    include_answers: bool = False,
    enable_llm_judge: bool = False,
    enable_deepeval: bool = False,
    judge_client_factory: JudgeClientFactory | None = None,
    judge_required: bool = False,
    deepeval_threshold: float = 0.7,
) -> dict:
    """隔离运行一个 Case 的全部 Trial，并采用所有 Trial 均通过的稳定性门槛。"""

    # planned_trials 允许 CLI 用 --max-trials 压低本地/CI 成本，但不抬高数据集声明值。
    planned_trials = case.trials if max_trials is None else min(case.trials, max_trials)
    # 防御性保证至少跑 1 次。
    planned_trials = max(1, planned_trials)
    # trial_results 保存每次独立运行的耗时、响应摘要和逐断言诊断。
    trial_results: list[dict] = []
    # judge_client 惰性创建，同一 Case 的多次 Trial 复用，避免重复握手。
    judge_client = None
    # 每次 Trial 使用全新 Engine；同一 Trial 的多轮输入复用这个 Engine。
    for trial_number in range(1, planned_trials + 1):
        # engine_factory 在测试中可注入替身，在真实运行中创建完整 WorkflowEngine。
        engine = engine_factory()
        # started_at 使用高精度单调时钟，仅测量 Trial 本身的墙钟耗时。
        started_at = perf_counter()
        # responses 保存多轮运行的每轮结果，最终评分默认针对最后一轮完整状态。
        responses = []
        # Trial 异常必须转成 Case 失败，不能中断其余数据集并丢失已有结果。
        try:
            # execution_context 在同一 Trial 的每一轮保持一致，确保预算行为可复现。
            execution_context = _execution_context(case)
            # 按声明顺序执行全部用户 Turn，并复用相同 Session、Memory 和 Engine。
            for input_text in _turn_inputs(case):
                # request 仍使用正式 AgentRunRequest，Eval 不获得内部状态旁路。
                request = _request_for_turn(case, input_text, trial_number)
                # response 来自唯一正式入口 WorkflowEngine.run。
                response = engine.run(request, execution_context=execution_context)
                # 保存本轮响应供最终评分和多轮 Trace ID 汇总。
                responses.append(response)
            # final_response 是多轮场景的最后一轮，单轮场景则是唯一响应。
            final_response = responses[-1]
            # 任一主观评分层需要 Judge 且尚未创建客户端时，按工厂构造一次。
            if (
                (enable_llm_judge or enable_deepeval)
                and judge_client is None
                and judge_client_factory is not None
            ):
                # 工厂失败会在 evaluate_case 中变成 judge.error 断言。
                judge_client = judge_client_factory()
            # evaluation 固定先合并自研断言，再按开关追加原生 Judge 与 DeepEval。
            evaluation = evaluate_case(
                case,
                final_response,
                enable_llm_judge=enable_llm_judge,
                enable_deepeval=enable_deepeval,
                judge_client=judge_client,
                judge_required=judge_required,
                deepeval_threshold=deepeval_threshold,
            )
            # duration_ms 在评分完成后记录，覆盖完整 Agent 运行和本地评分成本。
            duration_ms = int((perf_counter() - started_at) * 1000)
            # routing 标明本轮意图是 LLM 裁定还是向量/规则兜底，避免误以为“完全没调模型”。
            routing = extract_routing_diagnostics(final_response)
            # Trial 报告默认不保存完整回答；--include-answers 时写入截断预览供本地排障。
            trial_results.append(
                {
                    "trial": trial_number,
                    "passed": evaluation["passed"],
                    "score": evaluation["score"],
                    "duration_ms": duration_ms,
                    "turn_count": len(responses),
                    "trace_ids": [item.trace_id for item in responses],
                    "final_state": final_response.final_state,
                    "intent": final_response.intent,
                    "domain_skill": final_response.domain_skill,
                    "routing": routing,
                    "tool_names": [
                        str(item.get("tool_name") or "") for item in final_response.tool_calls
                    ],
                    "answer_preview": _answer_preview(
                        final_response.answer, include_answers=include_answers
                    ),
                    "assertions": evaluation["assertions"],
                    "error": None,
                }
            )
        # 捕获单个 Trial 的运行、请求或评分异常，继续执行后续 Trial 和 Case。
        except Exception as exc:
            # duration_ms 即使异常也记录已消耗时间，便于识别环境或 Provider 故障。
            duration_ms = int((perf_counter() - started_at) * 1000)
            # 异常路径尽量保留已完成轮次的路由诊断，方便区分“模型挂了”还是“根本没分类”。
            routing = (
                extract_routing_diagnostics(responses[-1])
                if responses
                else {
                    "source": None,
                    "reason_code": None,
                    "vector_score": None,
                    "confidence": None,
                    "dispatch_action": None,
                    "used_llm_intent": False,
                }
            )
            # 异常结果只保存类型和简短文本；正式日志仍不得包含密钥或认证 Header。
            trial_results.append(
                {
                    "trial": trial_number,
                    "passed": False,
                    "score": 0.0,
                    "duration_ms": duration_ms,
                    "turn_count": len(responses),
                    "trace_ids": [item.trace_id for item in responses],
                    "final_state": responses[-1].final_state if responses else None,
                    "intent": responses[-1].intent if responses else None,
                    "domain_skill": responses[-1].domain_skill if responses else None,
                    "routing": routing,
                    "tool_names": [],
                    "answer_preview": _answer_preview(
                        responses[-1].answer if responses else "",
                        include_answers=include_answers,
                    ),
                    "assertions": [],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
    # case_passed 使用 pass^k 语义：所有独立 Trial 均通过时 Case 才通过。
    case_passed = all(result["passed"] for result in trial_results)
    # case_score 取 Trial 断言分平均值，用于趋势比较但不覆盖硬失败结论。
    case_score = sum(float(result["score"]) for result in trial_results) / len(trial_results)
    # 返回 Case 级汇总和完整 Trial 诊断。
    return {
        "id": case.id,
        "type": case.type,
        "suite": case.suite,
        "maturity": case.maturity,
        "pass_fail_rules": list(case.pass_fail_rules),
        "planned_trials": planned_trials,
        "passed": case_passed,
        "score": case_score,
        "trials": trial_results,
    }


def _suite_summaries(case_results: list[dict]) -> dict[str, dict]:
    """按 Suite 聚合数量、通过率和平均断言分。"""

    # summaries 以 Suite 名为键，方便 CI 对安全、路由、工具等套件设置独立门槛。
    summaries: dict[str, dict] = {}
    # 逐个 Suite 建立稳定聚合结果。
    for suite in sorted({str(result["suite"]) for result in case_results}):
        # suite_results 只保留当前套件 Case。
        suite_results = [result for result in case_results if result["suite"] == suite]
        # current_results 仅统计计入发布门禁的样本。
        current_results = [result for result in suite_results if result.get("maturity") != "aspirational"]
        # passed_count 统计完整通过的 Case 数，而不是通过 Trial 数。
        passed_count = sum(1 for result in suite_results if result["passed"])
        # current_passed 只统计 current 成熟度。
        current_passed = sum(1 for result in current_results if result["passed"])
        # total_count 是当前 Suite 的 Case 总数。
        total_count = len(suite_results)
        # current_total 是门禁分母。
        current_total = len(current_results)
        # summaries 保存机器可读子指标。
        summaries[suite] = {
            "total": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate": passed_count / total_count if total_count else 0.0,
            "current_total": current_total,
            "current_passed": current_passed,
            "current_pass_rate": current_passed / current_total if current_total else 1.0,
            "score": (
                sum(float(result["score"]) for result in suite_results) / total_count
                if total_count
                else 0.0
            ),
        }
    # 返回全部 Suite 汇总。
    return summaries


def compare_with_baseline(report: dict, baseline: dict) -> dict[str, Any]:
    """对比当前报告与基线，找出 current Case 回归与修复。"""

    # baseline_by_id 以 Case ID 索引旧结果，便于 O(1) 查找。
    baseline_by_id = {
        str(item["id"]): item
        for item in baseline.get("cases", [])
        if isinstance(item, dict) and item.get("id")
    }
    # regressions 保存“基线通过、当前失败”的 current Case。
    regressions: list[str] = []
    # fixed 保存“基线失败、当前通过”的 Case，便于发布说明。
    fixed: list[str] = []
    # new_failures 与 regressions 相同语义，保留别名方便门禁配置阅读。
    for case_result in report.get("cases", []):
        # 只对 current 成熟度做回归门禁；aspirational 变化记入报告但不算回归。
        if case_result.get("maturity") == "aspirational":
            # 跳过前瞻样本。
            continue
        # previous 可能不存在（新增 Case）。
        previous = baseline_by_id.get(str(case_result["id"]))
        # 没有基线记录时不算回归，只是新增覆盖。
        if previous is None:
            # 继续检查下一条。
            continue
        # 基线通过且当前失败 = 回归。
        if previous.get("passed") and not case_result.get("passed"):
            # 记录稳定 Case ID。
            regressions.append(str(case_result["id"]))
        # 基线失败且当前通过 = 修复。
        if (not previous.get("passed")) and case_result.get("passed"):
            # 记录修复 ID。
            fixed.append(str(case_result["id"]))
    # 返回结构化对比结果。
    return {
        "regressions": sorted(regressions),
        "fixed": sorted(fixed),
        "regression_count": len(regressions),
        "fixed_count": len(fixed),
    }


def evaluate_promote_gates(
    report: dict,
    gate_config: dict[str, Any],
    *,
    baseline_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按 configs/eval_gates.yaml 判断当前报告是否允许 Promote。"""

    # overall_cfg / suites_cfg 读取嵌套配置，缺省为空字典。
    overall_cfg = dict(gate_config.get("overall") or {})
    suites_cfg = dict(gate_config.get("suites") or {})
    aspirational_cfg = dict(gate_config.get("aspirational") or {})
    baseline_cfg = dict(gate_config.get("baseline") or {})
    # failures 收集人类可读的门禁失败原因。
    failures: list[str] = []
    # include_aspirational 决定 aspirational 是否计入门槛分母。
    include_aspirational = bool(aspirational_cfg.get("fail_promote", False))
    # gate_cases 按策略筛选参与门禁的 Case。
    gate_cases = [
        case
        for case in report.get("cases", [])
        if include_aspirational or case.get("maturity") != "aspirational"
    ]
    # overall_pass_rate 只看门禁集合。
    overall_total = len(gate_cases)
    overall_passed = sum(1 for case in gate_cases if case.get("passed"))
    overall_pass_rate = overall_passed / overall_total if overall_total else 1.0
    # min_pass_rate 默认 0.7，与仓库示例配置一致。
    min_overall = float(overall_cfg.get("min_pass_rate", 0.7))
    # 总通过率不达标时记录失败。
    if overall_pass_rate + 1e-12 < min_overall:
        # 原因包含实际值与阈值，方便 CI 日志阅读。
        failures.append(
            f"overall pass_rate {overall_pass_rate:.3f} < min_pass_rate {min_overall:.3f}"
        )
    # 按 Suite 检查独立阈值。
    suite_summaries = report.get("suites") or {}
    for suite_name, suite_gate in sorted(suites_cfg.items()):
        # suite_gate 允许写成纯数字或含 min_pass_rate 的映射。
        if isinstance(suite_gate, dict):
            # 映射形式读取 min_pass_rate。
            min_rate = float(suite_gate.get("min_pass_rate", 0.0))
        else:
            # 数字形式直接作为最低通过率。
            min_rate = float(suite_gate)
        # summary 可能不存在（本轮未跑该 Suite）。
        summary = suite_summaries.get(suite_name)
        # 未执行的 Suite 不因缺失而失败，避免 --suite 局部运行误杀 Promote 检查。
        if summary is None:
            # 跳过未覆盖 Suite。
            continue
        # rate 优先使用 current_pass_rate；严格 aspirational 时改用总 pass_rate。
        rate = (
            float(summary.get("pass_rate", 0.0))
            if include_aspirational
            else float(summary.get("current_pass_rate", summary.get("pass_rate", 0.0)))
        )
        # Suite 未达标时追加失败原因。
        if rate + 1e-12 < min_rate:
            # 写明 Suite 名与阈值。
            failures.append(
                f"suite {suite_name} pass_rate {rate:.3f} < min_pass_rate {min_rate:.3f}"
            )
    # 基线对比：禁止 current 回归。
    if baseline_comparison is not None and baseline_cfg.get("disallow_regressions", True):
        # regression_count 来自 compare_with_baseline。
        regression_count = int(baseline_comparison.get("regression_count") or 0)
        # max_new_failures 允许极少数波动；默认 0。
        max_new = int(overall_cfg.get("max_new_failures", 0))
        # 超过允许回归数则阻断 Promote。
        if regression_count > max_new:
            # 附带回归 Case ID，便于定位。
            ids = ", ".join(baseline_comparison.get("regressions") or [])
            failures.append(
                f"baseline regressions {regression_count} > max_new_failures {max_new}: {ids}"
            )
    # passed 表示可以 Promote。
    passed = not failures
    # 返回门禁结论。
    return {
        "passed": passed,
        "failures": failures,
        "overall_pass_rate": overall_pass_rate,
        "overall_total": overall_total,
        "overall_passed": overall_passed,
        "include_aspirational": include_aspirational,
    }


def run_dataset(
    dataset_path: Path,
    engine_factory: EngineFactory = _default_engine_factory,
    suite: str | None = None,
    *,
    workers: int = 1,
    max_trials: int | None = None,
    include_answers: bool = False,
    enable_llm_judge: bool = False,
    enable_deepeval: bool = False,
    judge_client_factory: JudgeClientFactory | None = None,
    judge_required: bool = False,
    deepeval_threshold: float = 0.7,
    dry_run: bool = False,
    gate_config: dict[str, Any] | None = None,
    baseline_report: dict[str, Any] | None = None,
) -> dict:
    """运行完整数据集或指定 Suite，并返回可直接序列化的 Eval Run 报告。"""

    # started_at 保存人类可读 UTC 时间，方便跨环境关联一次 Eval Run。
    started_at = utc_now_iso()
    # started_clock 使用单调时钟统计整体耗时，不受系统时钟调整影响。
    started_clock = perf_counter()
    # dataset_bytes 同时用于加载前哈希，确保报告绑定到精确数据版本。
    dataset_bytes = dataset_path.read_bytes()
    # dataset_hash 使用 SHA-256，便于 Registry、CI 和历史报告比较不可变输入。
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    # all_cases 在任何 Agent 执行前完成全量 Schema 与 ID 校验，Suite 过滤不能隐藏脏数据。
    all_cases = load_dataset(dataset_path)
    # cases 只保留调用方选择的 Suite；None 表示运行完整黄金集。
    cases = [case for case in all_cases if case.suite == suite] if suite else all_cases
    # 拼错或不存在的 Suite 必须明确失败，不能用零 Case 的绿色结果误导 CI。
    if not cases:
        # 错误只包含 Suite 名，不回显 Case 输入。
        raise ValueError(f"数据集中不存在 Eval Suite: {suite}")
    # dry_run 只做数据集与门禁配置校验，不调用模型或工具。
    if dry_run:
        # duration_ms 仅覆盖加载与校验。
        duration_ms = int((perf_counter() - started_clock) * 1000)
        # 返回轻量校验报告，status 固定为 dry_run。
        return {
            "status": "dry_run",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "duration_ms": duration_ms,
            "dataset": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "selected_suite": suite,
            "enable_llm_judge": enable_llm_judge,
            "enable_deepeval": enable_deepeval,
            "deepeval_threshold": deepeval_threshold if enable_deepeval else None,
            "total": len(cases),
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "score": 0.0,
            "suites": {},
            "cases": [
                {
                    "id": case.id,
                    "suite": case.suite,
                    "maturity": case.maturity,
                    "trials": case.trials,
                    "pass_fail_rules": list(case.pass_fail_rules),
                }
                for case in cases
            ],
            "dry_run": True,
        }
    # worker_count 至少为 1；并行只跨 Case，不跨同一 Case 的 Trial。
    worker_count = max(1, int(workers))
    # case_results 最终按数据集顺序排列，保证报告稳定。
    case_results: list[dict] = []
    # 单线程路径保持原有顺序执行，方便本地调试。
    if worker_count == 1:
        # 顺序执行每个 Case。
        case_results = [
            run_case(
                case,
                engine_factory=engine_factory,
                max_trials=max_trials,
                include_answers=include_answers,
                enable_llm_judge=enable_llm_judge,
                enable_deepeval=enable_deepeval,
                judge_client_factory=judge_client_factory,
                judge_required=judge_required,
                deepeval_threshold=deepeval_threshold,
            )
            for case in cases
        ]
    else:
        # results_by_id 用于并行完成后按原序重排。
        results_by_id: dict[str, dict] = {}
        # ThreadPool 适合 IO/模型等待密集的 Agent 调用；Engine 非线程共享。
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # future_map 保存 Future 到 Case ID 的映射。
            future_map = {
                executor.submit(
                    run_case,
                    case,
                    engine_factory,
                    max_trials=max_trials,
                    include_answers=include_answers,
                    enable_llm_judge=enable_llm_judge,
                    enable_deepeval=enable_deepeval,
                    judge_client_factory=judge_client_factory,
                    judge_required=judge_required,
                    deepeval_threshold=deepeval_threshold,
                ): case.id
                for case in cases
            }
            # 按完成顺序收集，稍后重排。
            for future in as_completed(future_map):
                # case_id 用于写入字典。
                case_id = future_map[future]
                # result 可能携带异常；这里让异常冒泡以快速暴露 Harness 缺陷。
                results_by_id[case_id] = future.result()
        # 按原始 cases 顺序重建列表。
        case_results = [results_by_id[case.id] for case in cases]
    # passed_count 统计完整通过的 Case 数。
    passed_count = sum(1 for result in case_results if result["passed"])
    # total_count 是数据集中实际执行的 Case 数。
    total_count = len(case_results)
    # duration_ms 覆盖数据加载、Engine 构造、Agent 执行和评分报告聚合。
    duration_ms = int((perf_counter() - started_clock) * 1000)
    # score 使用 Case 分数等权平均；硬门禁仍由 failed 数量决定。
    score = (
        sum(float(result["score"]) for result in case_results) / total_count
        if total_count
        else 0.0
    )
    # report 保存数据版本、总览、Suite 指标和逐 Case 诊断。
    report: dict[str, Any] = {
        "status": "passed" if passed_count == total_count else "failed",
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "duration_ms": duration_ms,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "selected_suite": suite,
        "workers": worker_count,
        "max_trials": max_trials,
        "include_answers": include_answers,
        "enable_llm_judge": enable_llm_judge,
        "enable_deepeval": enable_deepeval,
        "deepeval_threshold": deepeval_threshold if enable_deepeval else None,
        "total": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass_rate": passed_count / total_count if total_count else 0.0,
        "score": score,
        "suites": _suite_summaries(case_results),
        "routing_summary": _routing_summary(case_results),
        "cases": case_results,
    }
    # 基线对比可选；有基线时写入 regressions/fixed。
    if baseline_report is not None:
        # comparison 只看 current 成熟度回归。
        report["baseline_comparison"] = compare_with_baseline(report, baseline_report)
    # 门禁评估可选；提供配置时写入 promote_gate。
    if gate_config is not None:
        # promote_gate 供 --check-promote 决定退出码。
        report["promote_gate"] = evaluate_promote_gates(
            report,
            gate_config,
            baseline_comparison=report.get("baseline_comparison"),
        )
    # 返回完整结构化报告，写文件和 Registry 持久化由调用层决定。
    return report


def write_json_report(report: dict, path: Path) -> None:
    """把完整 Eval 报告写成便于归档和程序读取的格式。"""

    # 创建报告目录，首次运行无需人工准备 evals/reports。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 UTF-8、中文直出和缩进格式，便于代码审查与本地排障。
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _failure_message(case_result: dict) -> str:
    """把结构化失败压缩成 JUnit 可读文本。"""

    # failures 保存异常或失败断言的短摘要，不包含完整回答与 Trace 正文。
    failures: list[str] = []
    # 遍历每个 Trial，保留 Trial 序号便于复现非确定性问题。
    for trial in case_result["trials"]:
        # 运行异常优先记录类型和简短消息。
        if trial.get("error"):
            # error 是 Runner 构造的受控字典，不包含异常栈或环境 Secret。
            error = trial["error"]
            # 将异常摘要加入 JUnit Failure 文本。
            failures.append(
                f"trial {trial['trial']}: {error.get('type')}: {error.get('message')}"
            )
            # 当前 Trial 已由异常解释，不再追加空断言列表。
            continue
        # 逐项收集未通过断言。
        for assertion in trial["assertions"]:
            # 通过项不应增加 JUnit 噪音。
            if assertion["passed"]:
                # 当前循环已经达到停止条件，立即跳过通过断言。
                continue
            # detail 优先提供人类可读原因，同时附稳定断言名。
            failures.append(
                f"trial {trial['trial']} {assertion['name']}: {assertion.get('detail') or 'failed'}"
            )
    # 本地排障若开启了答案预览，附加截断回答帮助理解失败。
    preview = None
    # 取最后一个 Trial 的预览（若有）。
    if case_result.get("trials"):
        # preview 可能为 None。
        preview = case_result["trials"][-1].get("answer_preview")
    # 有预览时追加到失败文本末尾。
    if preview:
        # 明确标注这是本地调试字段。
        failures.append(f"answer_preview: {preview}")
    # 返回换行分隔摘要；理论上失败 Case 至少有一条内容，空时使用防御性兜底。
    return "\n".join(failures) or "case failed without diagnostic"


def write_junit_report(report: dict, path: Path) -> None:
    """写出 CI 平台可展示的 JUnit XML 报告。"""

    # 创建报告目录，保持与 JSON 输出行为一致。
    path.parent.mkdir(parents=True, exist_ok=True)
    # suite 是 JUnit 根节点，汇总总 Case 数、失败数和秒级耗时。
    suite = ET.Element(
        "testsuite",
        {
            "name": "agent-evals",
            "tests": str(report["total"]),
            "failures": str(report["failed"]),
            "errors": "0",
            "time": f"{report['duration_ms'] / 1000:.3f}",
        },
    )
    # 每个 Eval Case 对应一个 JUnit testcase，Suite 名作为 classname 方便 CI 分组。
    for case_result in report["cases"]:
        # dry_run 报告没有 trials 字段时跳过 failure 细节。
        if report.get("dry_run"):
            # 只写占位 testcase，表示校验通过到加载阶段。
            ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": str(case_result["suite"]),
                    "name": str(case_result["id"]),
                    "time": "0",
                },
            )
            # 继续下一条。
            continue
        # case_time 汇总该 Case 全部 Trial 耗时。
        case_time = sum(int(trial["duration_ms"]) for trial in case_result["trials"]) / 1000
        # testcase 保存稳定 ID、Suite 和耗时。
        testcase = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": str(case_result["suite"]),
                "name": str(case_result["id"]),
                "time": f"{case_time:.3f}",
            },
        )
        # 失败 Case 增加 failure 子节点，CI 页面可直接展开逐断言原因。
        if not case_result["passed"]:
            # failure 不区分模型质量和断言类型，详细类别保留在文本中的稳定断言名。
            failure = ET.SubElement(testcase, "failure", {"message": "agent eval failed"})
            # 写入不包含完整正文（除非开启 include_answers）的压缩诊断。
            failure.text = _failure_message(case_result)
    # tree 使用 ElementTree 序列化标准 XML 声明。
    tree = ET.ElementTree(suite)
    # 写入 UTF-8 XML，供 GitHub Actions、GitLab CI 或 Jenkins 直接消费。
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _parse_args() -> argparse.Namespace:
    """解析本地和 CI 共用的 Eval Runner 参数。"""

    # parser 提供显式数据集、Suite 和报告路径，支持本地单能力调试与完整 CI 门禁。
    parser = argparse.ArgumentParser(description="Run isolated Agent evaluation cases.")
    # dataset 默认指向仓库当前回归集。
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals" / "dataset.jsonl")
    # suite 可选过滤单个能力套件；省略时运行完整数据集。
    parser.add_argument("--suite", type=str, default=None)
    # workers 控制 Case 级并行度；默认 1 保证本地日志顺序稳定。
    parser.add_argument("--workers", type=int, default=1, help="并行执行的 Case 数（Trial 仍串行）。")
    # max_trials 压低每个 Case 的重复次数，适合本地快速迭代。
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="覆盖上限：实际 trials = min(Case.trials, max_trials)。",
    )
    # dry_run 只校验数据集，不跑 Agent。
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验数据集 Schema/ID，不调用 WorkflowEngine。",
    )
    # 默认自动加载仓库根 .env，并以文件为准覆盖进程残留环境（本地 Eval 推荐）。
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="不自动加载仓库根目录 .env（默认会加载并以 .env 覆盖同名变量）。",
    )
    # CI 若已在进程注入密钥，可用此开关避免被本地 .env 覆盖。
    parser.add_argument(
        "--no-dotenv-override",
        action="store_true",
        help="加载 .env 但不覆盖进程里已有环境变量（CI 注入密钥时使用）。",
    )
    # dotenv 路径可覆盖，便于临时指向脱敏样例环境。
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=ROOT / ".env",
        help="要加载的 .env 路径；大模型地址/密钥/模型名均从此文件注入。",
    )
    # include_answers 把截断回答写入本地报告，默认关闭以保护隐私。
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="在本地 JSON/JUnit 中写入截断回答预览，便于排障。",
    )
    # enable_llm_judge 开启项目原生结构化 Judge，兼容已有本地和 CI 命令。
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="对声明了 judge/judge_rubric 的 Case 调用项目原生 LLM Judge。",
    )
    # enable_deepeval 在自研评估之后追加 G-Eval，不替换任何已有规则或原生 Judge。
    parser.add_argument(
        "--enable-deepeval",
        action="store_true",
        help="自研评估完成后，用 answer/retrieval context 追加 DeepEval G-Eval 分数。",
    )
    # deepeval_threshold 可覆盖门禁配置中的连续分通过阈值。
    parser.add_argument(
        "--deepeval-threshold",
        type=float,
        default=None,
        help="DeepEval G-Eval 通过阈值（0~1）；默认读取 configs/eval_gates.yaml。",
    )
    # gate_config 指定 Promote 阈值文件。
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=ROOT / "configs" / "eval_gates.yaml",
        help="Promote 门禁配置路径。",
    )
    # baseline 指定上一版报告 JSON，用于回归对比。
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="上一版 evals/reports/*.json，用于检测 current Case 回归。",
    )
    # check_promote 用门禁结果决定退出码（即使部分 aspirational 失败也可能通过）。
    parser.add_argument(
        "--check-promote",
        action="store_true",
        help="按门禁配置决定退出码；适合发版/Promote 流水线。",
    )
    # report 支持简写和语义清晰的长别名，默认写入被目录级 .gitignore 忽略的最新 JSON 报告。
    parser.add_argument(
        "--report",
        "--json-report",
        dest="report",
        type=Path,
        default=ROOT / "evals" / "reports" / "latest.json",
    )
    # junit 同样保留兼容简写和完整别名，供 CI 测试报告界面消费。
    parser.add_argument(
        "--junit",
        "--junit-report",
        dest="junit",
        type=Path,
        default=ROOT / "evals" / "reports" / "latest.xml",
    )
    # 返回解析后的命名空间。
    return parser.parse_args()


def main() -> int:
    """运行数据集、写出 JSON/JUnit，并以退出码执行质量门禁。"""

    # args 保存用户或 CI 覆盖后的输入输出路径。
    args = _parse_args()
    # dotenv_info 记录是否加载了本地密钥文件；默认以 .env 覆盖，保证读到你文件里的大模型配置。
    dotenv_info: dict[str, Any]
    if args.no_dotenv:
        # CI 或显式禁用时跳过文件加载，只使用进程已有环境变量。
        dotenv_info = {
            "loaded": False,
            "path": str(args.dotenv),
            "set_count": 0,
            "skipped_existing": 0,
            "overridden_count": 0,
            "disabled": True,
            "override": False,
            "alias_synced": [],
        }
    else:
        # override 默认 True：IDE/Shell 残留的旧 OPENAI_* 不能压过 .env。
        dotenv_info = load_dotenv(
            args.dotenv,
            override=not args.no_dotenv_override,
        )
        # disabled=False 表示本次允许自动加载。
        dotenv_info["disabled"] = False
    # model_endpoints 在 dotenv 之后解析，摘要写入报告供确认“用的是哪套网关/模型”。
    model_endpoints = resolve_model_endpoints_summary()
    # incomplete 端点会让意图 LLM/Judge 静默降级；本地 Eval 启动时给出明确提示。
    incomplete = [
        name
        for name, item in model_endpoints["endpoints"].items()
        if item.get("status") != "ok"
    ]
    if incomplete and not args.dry_run:
        # 只打印端点名，不打印密钥；提醒用户检查 .env。
        print(
            json.dumps(
                {
                    "warning": "部分模型端点配置不完整，相关调用可能降级为规则",
                    "incomplete_endpoints": incomplete,
                    "dotenv": {
                        "path": dotenv_info.get("path"),
                        "loaded": dotenv_info.get("loaded"),
                        "override": dotenv_info.get("override"),
                        "overridden_count": dotenv_info.get("overridden_count"),
                        "alias_synced": dotenv_info.get("alias_synced"),
                    },
                    "model_endpoints": model_endpoints,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    # gate_config 始终加载，便于 dry-run 也能尽早发现 YAML 错误。
    gate_config = load_gate_config(args.gate_config)
    # judge_config 集中读取是否强制执行以及 DeepEval 默认阈值。
    judge_config = dict(gate_config.get("judge") or {})
    # judge_required 来自门禁配置：声明了 judge 却未开启时是否记失败。
    judge_required = bool(judge_config.get("required_when_declared", False))
    # needs_judge_client 表示至少一个主观评分层需要项目模型端点。
    needs_judge_client = bool(args.enable_llm_judge or args.enable_deepeval)
    # deepeval_threshold 优先采用 CLI，其次读取版本化门禁配置。
    deepeval_threshold = float(
        args.deepeval_threshold
        if args.deepeval_threshold is not None
        else judge_config.get("deepeval_threshold", 0.7)
    )
    # 即使当前数据集没有 Judge Case，也要在启动阶段拒绝无效阈值。
    if not 0.0 <= deepeval_threshold <= 1.0:
        # CLI 配置错误不能静默生成看似成功的报告。
        raise ValueError("--deepeval-threshold 必须位于 0 到 1 之间")
    # baseline_report 可选加载。
    baseline_report = None
    if args.baseline is not None:
        # 基线必须是可读 JSON。
        baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    # judge_factory 仅在开启原生 Judge 或 DeepEval 时传入，避免无谓导入模型配置。
    judge_factory = _default_judge_client_factory if needs_judge_client else None
    # report 执行完整隔离评测或 dry-run 校验。
    report = run_dataset(
        args.dataset,
        suite=args.suite,
        workers=args.workers,
        max_trials=args.max_trials,
        include_answers=args.include_answers,
        enable_llm_judge=args.enable_llm_judge,
        enable_deepeval=args.enable_deepeval,
        judge_client_factory=judge_factory,
        judge_required=judge_required,
        deepeval_threshold=deepeval_threshold,
        dry_run=args.dry_run,
        gate_config=None if args.dry_run else gate_config,
        baseline_report=None if args.dry_run else baseline_report,
    )
    # JSON 报告用于趋势比较、Registry 记录和本地排障。
    write_json_report(report, args.report)
    # JUnit 报告用于 CI 原生测试视图。
    write_junit_report(report, args.junit)
    # summary 只输出总览和报告位置，避免逐节点日志淹没终端。
    summary = {
        "status": report["status"],
        "total": report["total"],
        "passed": report.get("passed", 0),
        "failed": report.get("failed", 0),
        "pass_rate": report.get("pass_rate", 0.0),
        "score": report.get("score", 0.0),
        "report": str(args.report),
        "junit": str(args.junit),
        "dry_run": bool(report.get("dry_run")),
        "dotenv": {
            "loaded": dotenv_info.get("loaded"),
            "path": dotenv_info.get("path"),
            "set_count": dotenv_info.get("set_count"),
            "skipped_existing": dotenv_info.get("skipped_existing"),
            "overridden_count": dotenv_info.get("overridden_count"),
            "override": dotenv_info.get("override"),
            "alias_synced": dotenv_info.get("alias_synced"),
            "disabled": dotenv_info.get("disabled"),
        },
        "model_endpoints": model_endpoints,
        "routing_summary": report.get("routing_summary"),
        "enable_llm_judge": report.get("enable_llm_judge"),
        "enable_deepeval": report.get("enable_deepeval"),
        "deepeval_threshold": report.get("deepeval_threshold"),
        "promote_gate": report.get("promote_gate"),
        "baseline_comparison": report.get("baseline_comparison"),
    }
    # 输出单行机器可读 JSON，Shell 和 CI 可以直接解析。
    print(json.dumps(summary, ensure_ascii=False))
    # dry_run 成功即返回 0。
    if report.get("dry_run"):
        # 校验通过。
        return 0
    # Promote 模式：以门禁结论为准，而不是要求全部 aspirational 也绿。
    if args.check_promote:
        # promote_gate 在 run_dataset 中已写入。
        gate = report.get("promote_gate") or {}
        # 门禁通过返回 0。
        return 0 if gate.get("passed") else 1
    # 默认模式：任一 Case 失败（含 aspirational）返回 1，保持严格本地回归语义。
    return 0 if report["failed"] == 0 else 1


# 直接执行脚本时运行 CLI；被测试导入时只暴露可复用函数。
if __name__ == "__main__":
    # 将 main 返回值交给进程退出码。
    raise SystemExit(main())
