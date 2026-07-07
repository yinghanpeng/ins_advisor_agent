# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


def test_workflow_engine_routes_insurance_request():
    response = WorkflowEngine().run(AgentRunRequest(input="客户喜欢银行理财，我怎么破冰"))
    assert response.final_state == "FINAL"
    assert response.domain_skill == "insurance_advisor"
    assert response.intent == "insurance_advisor_help"
    assert response.retrieved_context


def test_workflow_engine_handles_general_request():
    response = WorkflowEngine().run(AgentRunRequest(input="今天上海天气"))
    assert response.final_state == "FINAL"
    assert response.intent == "weather_query"
    assert response.context_needs["tool"] is True
    assert response.tool_calls
    assert response.tool_results[0]["name"] == "weather_query"


def test_workflow_engine_executes_calculator_tool_chain():
    response = WorkflowEngine().run(AgentRunRequest(input="计算 12*8+3"))
    assert response.final_state == "FINAL"
    assert response.answer == "计算结果是：99。"
    assert response.tool_calls[0]["tool_name"] == "calculator"
    assert response.tool_results[0]["output"]["result"] == 99
    path = [item["to_state"] for item in response.state_transitions]
    assert "GENERAL_TOOL_ROUTING" in path
    assert "GENERAL_TOOL_CALL" in path
    assert "VERIFY_TOOL_RESULT" in path


def test_workflow_engine_resolves_pronoun_from_session_memory():
    engine = WorkflowEngine()
    engine.run(
        AgentRunRequest(
            input="上文讨论的是 Anthropic",
            session_id="memory_demo_session",
            user_id="memory_demo_user",
        )
    )
    response = engine.run(
        AgentRunRequest(
            input="帮我查一下它最近有没有融资，重点看过去三个月的英文报道",
            session_id="memory_demo_session",
            user_id="memory_demo_user",
        )
    )
    assert response.query_understanding["entity"] == "Anthropic"
    assert response.query_understanding["filters"]["language"] == "en"
    assert response.query_understanding["filters"]["source_type"] == "news"
    assert response.query_understanding["filters"]["date_range"]
