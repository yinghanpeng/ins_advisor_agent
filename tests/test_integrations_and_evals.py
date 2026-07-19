# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.evals.evaluators import rule_based_evaluate
from agent_core.integrations.dify_webhook import normalize_dify_payload
from agent_core.workflow.contracts import AgentRunResponse, EvalCase


def test_dify_payload_normalization():
    payload = normalize_dify_payload({"query": "你好", "conversation_id": "c1"})
    assert payload["input"] == "你好"
    assert payload["session_id"] == "c1"
    assert payload["metadata"]["source"] == "dify"


def test_rule_based_evaluator():
    case = EvalCase(id="e1", type="normal", input="x", must_include=["低压"], must_not_include=["保证收益"])
    response = AgentRunResponse(trace_id="t1", session_id="s1", final_state="FINAL", answer="低压沟通")
    result = rule_based_evaluate(case, response)
    assert result["passed"] is True


def test_rule_based_evaluator_checks_route_tools_guardrail_trace_trajectory_and_cost():
    """丰富 Case 中每个结构化期望都必须真正影响最终评分。"""

    case = EvalCase(
        id="e2",
        type="tool",
        input="计算 1+1",
        expected_state="FINAL",
        expected_intent="calculator_query",
        expected_tools=["calculator"],
        forbidden_tools=["web_search"],
        expected_guardrail="tool_permission",
        expected_guardrail_action="pass",
        expected_guardrail_triggered=False,
        expected_trace_fields=["trace_id", "cost.tool_call_count", "tool_name"],
        required_states=["GENERAL_TOOL_CALL", "VERIFY_TOOL_RESULT", "FINAL"],
        forbidden_states=["RECOVERY"],
        expected_cost={"tool_call_count": 1},
        max_tool_calls=1,
        pass_fail_rules=["answer", "state", "intent", "tools", "guardrail", "trace", "cost", "trajectory"],
    )
    response = AgentRunResponse(
        trace_id="t2",
        session_id="s2",
        final_state="FINAL",
        answer="计算结果是 2",
        intent="calculator_query",
        guardrails=[{"guardrail_name": "tool_permission", "action": "pass", "triggered": False}],
        trace_events=[{"event": "tool_finished", "tool_name": "calculator"}],
        state_transitions=[
            {"to_state": "GENERAL_TOOL_CALL"},
            {"to_state": "VERIFY_TOOL_RESULT"},
            {"to_state": "FINAL"},
        ],
        tool_calls=[{"tool_name": "calculator", "status": "success"}],
        cost={"tool_call_count": 1},
    )

    result = rule_based_evaluate(case, response)

    assert result["passed"] is True
    assert all(assertion["passed"] for assertion in result["assertions"])


def test_rule_based_evaluator_reports_each_failed_dimension():
    """路由、工具、轨迹和预算失败必须返回独立且可定位的断言名。"""

    case = EvalCase(
        id="e3",
        type="tool",
        input="计算 1+1",
        expected_state="ERROR",
        expected_intent="calculator_query",
        expected_tools=["calculator"],
        required_states=["GENERAL_TOOL_CALL", "FINAL"],
        expected_cost={"tool_call_count": 1},
    )
    response = AgentRunResponse(
        trace_id="t3",
        session_id="s3",
        final_state="FINAL",
        answer="无法计算",
        intent="general_chat",
    )

    result = rule_based_evaluate(case, response)
    failed_names = {assertion["name"] for assertion in result["assertions"] if not assertion["passed"]}

    assert result["passed"] is False
    assert {"state.final", "intent.label", "tools.required", "trajectory.required_states", "cost.expected"} <= failed_names


def test_eval_case_rejects_untrusted_state_and_unknown_grader():
    """Eval 数据不能通过任意 initial_state 或自然语言规则绕过正式契约。"""

    import pytest

    with pytest.raises(ValueError, match="initial_state"):
        EvalCase(id="bad-state", type="security", input="x", initial_state={"admin": True})

    with pytest.raises(ValueError, match="未实现规则"):
        EvalCase(id="bad-rule", type="security", input="x", pass_fail_rules=["looks_good"])

    with pytest.raises(ValueError, match="未覆盖"):
        EvalCase(
            id="missing-rule",
            type="tool",
            input="x",
            expected_tools=["calculator"],
            pass_fail_rules=["answer"],
        )


def test_rule_based_evaluator_only_scores_explicit_rules():
    """显式 pass_fail_rules 必须真实控制评分维度，而不是无效说明字段。"""

    case = EvalCase(
        id="answer-only",
        type="normal",
        input="x",
        must_include=["完成"],
        pass_fail_rules=["answer"],
    )
    response = AgentRunResponse(
        trace_id="t4",
        session_id="s4",
        final_state="FINAL",
        answer="完成",
    )

    result = rule_based_evaluate(case, response)

    assert result["passed"] is True
    assert {assertion["name"] for assertion in result["assertions"]} == {
        "answer.non_empty",
        "answer.must_include",
        "answer.must_include_any",
        "answer.must_not_include",
    }


def test_rule_based_evaluator_accepts_must_include_any_synonym_groups():
    """同义组任一命中即通过，避免单一关键词造成假失败。"""

    case = EvalCase(
        id="synonym",
        type="business",
        input="x",
        must_include_any=[["资金", "资产", "理财"]],
        pass_fail_rules=["answer"],
    )
    response = AgentRunResponse(
        trace_id="t5",
        session_id="s5",
        final_state="FINAL",
        answer="可以先从客户现有理财配置聊起",
    )

    result = rule_based_evaluate(case, response)

    assert result["passed"] is True
    assert any(
        item["name"] == "answer.must_include_any" and item["passed"]
        for item in result["assertions"]
    )


def test_resolve_sales_route_normalizes_aliases_and_classifies_input():
    """销售场景应从意图投影或输入分类解析，并把历史别名归一化。"""

    from agent_core.evals.evaluators import normalize_sales_route, resolve_sales_route

    assert normalize_sales_route("break_ice") == "icebreaking"

    case = EvalCase(
        id="scene",
        type="sales",
        input="帮我讲一个匿名泛化案例增强说服力",
        expected_domain_skill="insurance_advisor",
        expected_sales_intelligence_route="case_evidence",
        pass_fail_rules=["intent", "sales_route"],
    )
    response = AgentRunResponse(
        trace_id="t6",
        session_id="s6",
        final_state="FINAL",
        answer="用泛化案例说明",
        domain_skill="insurance_advisor",
        intent=None,
    )

    assert resolve_sales_route(case, response) == "case_evidence"
    result = rule_based_evaluate(case, response)
    assert result["passed"] is True


def test_evaluate_case_skips_judge_unless_enabled():
    """未开启 --enable-llm-judge 时，声明了 judge 的 Case 默认跳过且不阻断。"""

    from agent_core.evals.evaluators import evaluate_case

    case = EvalCase(
        id="judge-skip",
        type="business",
        input="x",
        judge_rubric="表达是否得体",
        pass_fail_rules=["answer", "judge"],
    )
    response = AgentRunResponse(
        trace_id="t7",
        session_id="s7",
        final_state="FINAL",
        answer="得体回答",
    )

    skipped = evaluate_case(case, response, enable_llm_judge=False, judge_required=False)
    assert skipped["passed"] is True
    assert any(item["name"] == "judge.skipped" for item in skipped["assertions"])

    required = evaluate_case(case, response, enable_llm_judge=False, judge_required=True)
    assert required["passed"] is False
