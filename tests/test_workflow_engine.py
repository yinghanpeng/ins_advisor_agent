# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
import pytest

from agent_core.graph.state import AgentNode
from agent_core.memory.manager import MemoryLayer
from agent_core.observability.langsmith_client import LangSmithAdapter
from agent_core.workflow.contracts import AgentRunRequest
from agent_core.workflow.engine import AGENT_STEP_LABELS, WorkflowEngine


class _RecordingLogger:
    """在内存中收集结构化日志，验证实时节点日志而不依赖控制台。"""

    def __init__(self) -> None:
        """初始化按发生顺序保存的日志列表。"""

        self.records: list[dict[str, object]] = []

    def event(self, event: str, **fields: object) -> None:
        """记录普通结构化事件。"""

        self.records.append({"event": event, **fields})

    def warning(self, event: str, **fields: object) -> None:
        """记录告警事件并保留 warning 级别。"""

        self.records.append({"event": event, "level": "warning", **fields})


class _FailingGraph:
    """在首个状态迁移后抛错，用于证明异常不会吞掉已完成步骤日志。"""

    def invoke(self, state):
        """记录一次状态迁移后模拟节点异常。"""

        state.move_to(AgentNode.INIT_CONTEXT, reason="test_failure")
        raise RuntimeError("simulated failure")


def test_every_agent_node_has_a_human_readable_flow_label():
    """新增状态节点时必须同步提供流程日志名称，避免终端退回难读的内部枚举。"""

    assert set(AGENT_STEP_LABELS) == {node.value for node in AgentNode}
    assert all(label.strip() for label in AGENT_STEP_LABELS.values())


def test_workflow_engine_logs_every_state_transition_in_real_time():
    """每个执行步骤的状态迁移都必须实时形成一条安全结构化日志。"""

    logger = _RecordingLogger()
    engine = WorkflowEngine(
        log=logger,
        langsmith=LangSmithAdapter(enabled=False),
    )

    response = engine.run(AgentRunRequest(input="计算 12*8+3"))

    logged_transitions = [
        (record.get("from_state"), record.get("to_state"))
        for record in logger.records
        if record.get("event") == "trace_event"
        and record.get("trace_event_name") == "state_transition"
    ]
    response_transitions = [
        (transition["from_state"], transition["to_state"])
        for transition in response.state_transitions
    ]
    assert logged_transitions == response_transitions
    flow_steps = [record for record in logger.records if record.get("event") == "agent_flow_step"]
    assert [record.get("step_index") for record in flow_steps] == list(range(1, len(flow_steps) + 1))
    assert flow_steps[0]["step_name"] == "初始化"
    assert any(record.get("step_name") == "输入安全拦截" for record in flow_steps)
    summary = next(record for record in logger.records if record.get("event") == "agent_flow_summary")
    assert summary["status"] == "completed"
    assert str(summary["flow"]).startswith("初始化 → 输入安全拦截 → 恢复记忆")
    assert logger.records[0]["event"] == "agent_run_started"
    assert logger.records[-1]["event"] == "agent_run_finished"
    assert all("input_text" not in record for record in logger.records)


def test_workflow_engine_logs_failure_state_before_reraising():
    """节点抛错时必须记录失败状态，同时保留异常供 API 错误边界处理。"""

    logger = _RecordingLogger()
    engine = WorkflowEngine(
        log=logger,
        langsmith=LangSmithAdapter(enabled=False),
    )
    engine.graph = _FailingGraph()

    with pytest.raises(RuntimeError, match="simulated failure"):
        engine.run(AgentRunRequest(input="触发测试异常"))

    failure = next(record for record in logger.records if record.get("event") == "agent_run_failed")
    assert failure["current_state"] == AgentNode.INIT_CONTEXT.value
    assert failure["exception_type"] == "RuntimeError"
    summary = next(record for record in logger.records if record.get("event") == "agent_flow_summary")
    assert summary["status"] == "failed"
    assert summary["flow"] == "初始化"
    assert any(
        record.get("trace_event_name") == "state_transition"
        for record in logger.records
    )


def test_workflow_engine_routes_insurance_request():
    # 保险沟通输入应进入 insurance_advisor Domain Skill，而不是通用聊天。
    response = WorkflowEngine().run(AgentRunRequest(input="客户喜欢银行理财，我怎么破冰"))
    # 工作流应正常结束。
    assert response.final_state == "FINAL"
    # domain_skill 证明领域路由命中保险顾问。
    assert response.domain_skill == "insurance_advisor"
    # intent 证明向量知识库命中了细分破冰意图。
    assert response.intent == "insurance_break_ice"
    # 首轮信息不足时先追问，不提前检索方法库或生成完整策略。
    assert response.insurance_kyc_status["information_status"] == "insufficient"
    # 活跃意图会写入 Session，下一轮优先判断是否在回答本问题。
    assert response.active_intent["intent"] == "insurance_break_ice"


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


def test_multi_intent_executes_in_priority_order_and_writes_one_conversation_turn() -> None:
    """复合请求逐步执行，但 Session 只能保存原始整句和一次聚合回答。"""
    engine = WorkflowEngine()
    original_input = "帮我计算 12*8，然后查上海天气，同时给客户一版保险破冰话术"

    response = engine.run(
        AgentRunRequest(
            input=original_input,
            tenant_id="tenant_multi",
            session_id="session_multi",
        )
    )

    plan = response.intent_routing_result["execution_plan"]
    assert [step["intent"] for step in plan] == [
        "calculator_query",
        "weather_query",
        "insurance_break_ice",
    ]
    assert [step["execution_priority"] for step in plan] == [20, 30, 70]
    assert [result["name"] for result in response.tool_results] == [
        "calculator",
        "weather_query",
    ]
    # 聚合章节顺序与实际执行 sequence 一致，不能按模型输出顺序或工具完成时间重排。
    assert response.answer.index("1. calculator_query") < response.answer.index("2. weather_query")
    assert response.answer.index("2. weather_query") < response.answer.index("3. insurance_break_ice")
    # 顶层主意图是通用计算，但实际执行过保险步骤时仍需暴露 KYC 控制摘要。
    assert response.insurance_kyc_status["information_status"] == "insufficient"

    session = engine.memory_manager.read(MemoryLayer.SESSION, "tenant_multi", "session_multi")
    user_messages = [
        item["content"]
        for item in session["recent_messages"]
        if item.get("role") == "user"
    ]
    assert user_messages == [original_input]
    assert len([item for item in session["recent_messages"] if item.get("role") == "assistant"]) == 1
    assert session["last_intents"] == [
        "calculator_query",
        "weather_query",
        "insurance_break_ice",
    ]


def test_session_reference_does_not_refresh_entity_anchor_ttl() -> None:
    """代词解析可以读取实体锚点，但不能把推断值伪装成用户再次明确提及并续期。"""
    engine = WorkflowEngine()
    engine.run(
        AgentRunRequest(
            input="上文讨论的是 Anthropic",
            tenant_id="tenant_entity",
            session_id="session_entity",
        )
    )
    first_session = engine.memory_manager.read(
        MemoryLayer.SESSION,
        "tenant_entity",
        "session_entity",
    )
    first_anchor = dict(first_session["last_entity_anchor"])

    response = engine.run(
        AgentRunRequest(
            input="帮我查一下它最近有没有融资",
            tenant_id="tenant_entity",
            session_id="session_entity",
        )
    )
    second_session = engine.memory_manager.read(
        MemoryLayer.SESSION,
        "tenant_entity",
        "session_entity",
    )

    assert response.query_understanding["entity"] == "Anthropic"
    assert response.query_understanding["entity_source"] == "session_reference"
    assert second_session["last_entity_anchor"] == first_anchor


def test_last_entity_uses_original_text_order_not_multi_intent_priority_order() -> None:
    """多意图会按 priority 重排执行，但实体锚点必须指向用户原文最后明确提到的公司。"""
    engine = WorkflowEngine()
    original_input = "查 Anthropic 的新闻，然后计算 2+2，同时查 OpenAI 的新闻"

    engine.run(
        AgentRunRequest(
            input=original_input,
            tenant_id="tenant_entity_order",
            session_id="session_entity_order",
        )
    )
    session = engine.memory_manager.read(
        MemoryLayer.SESSION,
        "tenant_entity_order",
        "session_entity_order",
    )

    # calculator priority=20 会先于 search priority=40，但不能把锚点选择变成执行顺序副作用。
    assert session["last_entity"] == "OpenAI"
    assert session["last_entity_anchor"]["value"] == "OpenAI"
    assert session["last_entity_anchor"]["source"] == "explicit_user_text"
