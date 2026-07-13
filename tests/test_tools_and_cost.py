# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
import time

from agent_core.capabilities.calculator import run as calculator_run
from agent_core.cost.budget import CostBudget
from agent_core.tools.executor import CAPABILITY_RUNNERS, execute_tool_call
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.router import ToolRouter
from agent_core.tools.schemas import ToolCall


def test_tool_registry_and_router():
    # 默认工具注册表应包含 calculator 等基础工具。
    registry = ToolRegistry.with_defaults()
    # ToolRouter 应根据“计算”和算术表达式路由到 calculator。
    spec = ToolRouter(registry).route("请帮我计算 12*8+3")
    # spec 不为空说明路由器找到了注册工具。
    assert spec is not None
    # 工具名必须是 calculator，不能误路由到 summarizer 或 web_search。
    assert spec.name == "calculator"


def test_calculator_safe_expression():
    # calculator 只执行清洗后的安全表达式，并返回结构化 result。
    assert calculator_run({"expression": "12*8+3"})["result"] == 99


def test_cost_budget_rejects_overrun():
    # 构造一个 10 token 的小预算，便于测试超预算行为。
    budget = CostBudget(request_token_budget=10)
    # 先消耗 5 token，应成功。
    budget.spend(5)
    assert budget.used_tokens == 5
    # 再消耗 6 token 会超过预算，应抛 ValueError。
    try:
        budget.spend(6)
    except ValueError as exc:
        # 错误信息应包含 budget，方便排障。
        assert "budget" in str(exc)
    else:
        # 如果没有抛错，说明预算保护失效。
        raise AssertionError("expected budget overrun")


def test_cost_budget_returns_structured_decision():
    # decide 不直接抛错，而是返回结构化决策，供 workflow 选择压缩或降级。
    budget = CostBudget(request_token_budget=10)
    # 预计消耗 12 token 超过预算。
    decision = budget.decide(12)
    # allowed=False 表示当前请求不应原样继续。
    assert decision.allowed is False
    # action=reduce_context 表示建议先压缩上下文。
    assert decision.action == "reduce_context"


def test_tool_spec_contains_permission_metadata():
    # 工具 registry 中的 news_search 必须带权限 metadata。
    registry = ToolRegistry.with_defaults()
    spec = registry.get("news_search")
    # spec 不为空说明默认工具已注册。
    assert spec is not None
    # news_search 需要 tenant 权限等级，因为它属于外部信息读取能力。
    assert spec.permission.level == "tenant"
    # internet.read scope 会被 ToolPermissionPolicy 用于白名单判断。
    assert spec.permission.scope == "internet.read"


def test_tool_timeout_returns_without_waiting_for_worker_shutdown(monkeypatch) -> None:
    """Runner 超时后当前请求必须立即降级，不能被线程池上下文再次阻塞。"""
    registry = ToolRegistry.with_defaults()
    base_spec = registry.get("summarizer")
    assert base_spec is not None
    # 20ms 工具预算配合 250ms 慢任务，足以暴露 with ThreadPoolExecutor 的隐式 wait=True。
    timeout_spec = base_spec.model_copy(
        update={"timeout_ms": 20, "retryable": False, "retry_policy": {"max_attempts": 1}}
    )

    def slow_runner(arguments):
        del arguments
        time.sleep(0.25)
        return {"summary": "late result"}

    monkeypatch.setitem(CAPABILITY_RUNNERS, "summarizer", slow_runner)
    started = time.perf_counter()
    result = execute_tool_call(
        ToolCall(name="summarizer", arguments={"text": "需要摘要的文本"}),
        timeout_spec,
    )
    elapsed = time.perf_counter() - started

    assert result.status == "error"
    assert "timeout" in (result.error or "")
    assert elapsed < 0.15
