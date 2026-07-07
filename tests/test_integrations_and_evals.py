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

