# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import WorkflowEngine


def test_workflow_engine_routes_insurance_request():
    # 保险沟通输入应进入 insurance_advisor Domain Skill，而不是通用聊天。
    response = WorkflowEngine().run(AgentRunRequest(input="客户喜欢银行理财，我怎么破冰"))
    # 工作流应正常结束。
    assert response.final_state == "FINAL"
    # domain_skill 证明领域路由命中保险顾问。
    assert response.domain_skill == "insurance_advisor"
    # intent 证明意图识别为保险顾问帮助。
    assert response.intent == "insurance_advisor_help"
    # retrieved_context 证明销售洞察检索链路被触发。
    assert response.retrieved_context


def test_workflow_engine_handles_general_request():
    # 天气输入应进入通用工具路径。
    response = WorkflowEngine().run(AgentRunRequest(input="今天上海天气"))
    # 工具路径也应正常结束。
    assert response.final_state == "FINAL"
    # 意图识别为 weather_query。
    assert response.intent == "weather_query"
    # Context Need 应明确 tool=True。
    assert response.context_needs["tool"] is True
    # tool_calls/tool_results 证明真实执行了工具链路。
    assert response.tool_calls
    assert response.tool_results[0]["name"] == "weather_query"


def test_workflow_engine_executes_calculator_tool_chain():
    # 计算输入应路由到 calculator，而不是让模型心算。
    response = WorkflowEngine().run(AgentRunRequest(input="计算 12*8+3"))
    # 工作流最终应正常结束。
    assert response.final_state == "FINAL"
    # 回答应来自 calculator 工具结果。
    assert response.answer == "计算结果是：99。"
    # tool_calls 记录调用过程，tool_results 记录可消费结果。
    assert response.tool_calls[0]["tool_name"] == "calculator"
    assert response.tool_results[0]["output"]["result"] == 99
    # 状态路径必须包含工具路由、工具调用和工具校验三个节点。
    path = [item["to_state"] for item in response.state_transitions]
    assert "GENERAL_TOOL_ROUTING" in path
    assert "GENERAL_TOOL_CALL" in path
    assert "VERIFY_TOOL_RESULT" in path


def test_workflow_engine_resolves_pronoun_from_session_memory():
    # 复用同一个 WorkflowEngine，确保两轮请求共享 MemoryManager。
    engine = WorkflowEngine()
    # 第一轮把 Anthropic 写入 session memory 的 last_entity。
    engine.run(
        AgentRunRequest(
            input="上文讨论的是 Anthropic",
            session_id="memory_demo_session",
            user_id="memory_demo_user",
        )
    )
    # 第二轮用户只说“它”，Query Understanding 应从短期记忆中消解为 Anthropic。
    response = engine.run(
        AgentRunRequest(
            input="帮我查一下它最近有没有融资，重点看过去三个月的英文报道",
            session_id="memory_demo_session",
            user_id="memory_demo_user",
        )
    )
    # entity 证明指代消解成功。
    assert response.query_understanding["entity"] == "Anthropic"
    # 英文报道应转成 language=en filter。
    assert response.query_understanding["filters"]["language"] == "en"
    # 报道/新闻请求应转成 source_type=news。
    assert response.query_understanding["filters"]["source_type"] == "news"
    # 过去三个月应解析成绝对日期范围。
    assert response.query_understanding["filters"]["date_range"]
