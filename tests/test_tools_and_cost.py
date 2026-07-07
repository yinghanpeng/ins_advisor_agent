# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.capabilities.calculator import run as calculator_run
from agent_core.cost.budget import CostBudget
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.router import ToolRouter


def test_tool_registry_and_router():
    registry = ToolRegistry.with_defaults()
    spec = ToolRouter(registry).route("请帮我计算 12*8+3")
    assert spec is not None
    assert spec.name == "calculator"


def test_calculator_safe_expression():
    assert calculator_run({"expression": "12*8+3"})["result"] == 99


def test_cost_budget_rejects_overrun():
    budget = CostBudget(request_token_budget=10)
    budget.spend(5)
    assert budget.used_tokens == 5
    try:
        budget.spend(6)
    except ValueError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("expected budget overrun")


def test_cost_budget_returns_structured_decision():
    budget = CostBudget(request_token_budget=10)
    decision = budget.decide(12)
    assert decision.allowed is False
    assert decision.action == "reduce_context"


def test_tool_spec_contains_permission_metadata():
    registry = ToolRegistry.with_defaults()
    spec = registry.get("news_search")
    assert spec is not None
    assert spec.permission.level == "tenant"
    assert spec.permission.scope == "internet.read"
