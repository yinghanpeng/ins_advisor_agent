"""Workflow step contracts for Agent Core and Dify mapping."""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

from agent_core.graph.state import AgentNode
from agent_core.workflow.contracts import WorkflowContract, WorkflowStepContract


# 保险破冰助手的逻辑步骤清单，用于文档、Dify 映射和面试讲解。
BREAK_ICE_ASSISTANT_STEPS = [
    # 识别用户是不是保险沟通/破冰/异议处理需求。
    "classify_intent",
    # 抽取客户 KYC，例如企业主、两个孩子、资产偏好。
    "extract_customer_kyc",
    # 抽取销售当前卡点，例如不会破冰、客户不信任、客户只看银行理财。
    "extract_sales_pain",
    # 判断具体沟通场景，例如饭局破冰、老客维护、计划书讲解。
    "classify_scene",
    # 检索已审核的销售实战洞察卡片。
    "retrieve_sales_intelligence",
    # 必要时检索外部上下文，例如热点新闻或宏观背景。
    "retrieve_external_context_if_needed",
    # 构建包含证据边界的上下文。
    "build_context",
    # 生成合规、低压、可执行的回答。
    "generate_response",
    # 输出前做保险合规审查。
    "compliance_review",
    # 返回最终响应。
    "final_response",
]


# BREAK_ICE_ASSISTANT_CONTRACT 是“破冰助手 workflow”的显式 step contract。
# 它不执行代码，而是声明每个 step 的输入、输出、允许下一状态、风控和 trace 字段。
BREAK_ICE_ASSISTANT_CONTRACT = WorkflowContract(
    # Dify、文档和测试都会引用这个稳定工作流名。
    name="break_ice_assistant_workflow",
    # 该工作流从意图识别开始，因为用户请求进入时还不知道是否命中保险顾问。
    entry_state=AgentNode.CLASSIFY_INTENT,
    # 允许正常结束、人工审批停住或错误终止。
    final_states=[AgentNode.FINAL, AgentNode.HUMAN_APPROVAL, AgentNode.ERROR],
    # steps 明确每个节点的契约，避免流程只藏在 prompt 或代码 if/else 中。
    steps=[
        # classify_intent step：识别意图并产出路由结果。
        WorkflowStepContract(
            name="classify_intent",
            state=AgentNode.CLASSIFY_INTENT,
            description="Classify user intent and choose general/domain capability route.",
            required_inputs=["input_text"],
            produced_outputs=["intent", "capability_route", "domain_skill"],
            allowed_next_states=[AgentNode.ROUTE_CAPABILITY],
            guardrails=["input_prompt_injection"],
            trace_fields=["trace_id", "intent", "capability_route"],
        ),
        # route_domain_workflow step：把保险顾问请求转入销售智能层。
        WorkflowStepContract(
            name="route_domain_workflow",
            state=AgentNode.DOMAIN_WORKFLOW_ROUTING,
            description="Route insurance advisor requests to the proper domain workflow.",
            required_inputs=["intent", "domain_skill"],
            produced_outputs=["sales_route"],
            allowed_next_states=[AgentNode.SALES_INTELLIGENCE_ROUTING, AgentNode.BUILD_CONTEXT],
            trace_fields=["trace_id", "domain_skill", "sales_route"],
        ),
        # retrieve_sales_intelligence step：只检索已审核销售卡片，不直接使用原始访谈。
        WorkflowStepContract(
            name="retrieve_sales_intelligence",
            state=AgentNode.SALES_INSIGHT_RETRIEVAL,
            description="Retrieve approved sales insight cards instead of raw interviews.",
            required_inputs=["input_text", "sales_route"],
            produced_outputs=["rewritten_queries", "retrieved_context"],
            allowed_next_states=[AgentNode.BUILD_CONTEXT, AgentNode.RECOVERY],
            guardrails=["sales_corpus_guardrail"],
            tools_allowed=["knowledge_search"],
            trace_fields=["trace_id", "rewritten_queries", "retrieved_context"],
        ),
        # build_context step：把检索证据压缩成带来源边界的 digest。
        WorkflowStepContract(
            name="build_context",
            state=AgentNode.BUILD_CONTEXT,
            description="Build compact context with source boundaries and evidence digest.",
            required_inputs=["retrieved_context"],
            produced_outputs=["sales_insight_digest"],
            allowed_next_states=[AgentNode.GENERATE_RESPONSE],
            trace_fields=["trace_id", "sales_insight_digest"],
        ),
        # generate_response step：基于压缩上下文生成候选回答。
        WorkflowStepContract(
            name="generate_response",
            state=AgentNode.GENERATE_RESPONSE,
            description="Generate a domain answer from compact context.",
            required_inputs=["input_text", "sales_insight_digest"],
            produced_outputs=["answer"],
            allowed_next_states=[AgentNode.COMPLIANCE_REVIEW],
            trace_fields=["trace_id", "answer"],
        ),
        # compliance_review step：输出前审查，必要时进入人工审批。
        WorkflowStepContract(
            name="compliance_review",
            state=AgentNode.COMPLIANCE_REVIEW,
            description="Review output and route unsafe responses to human approval.",
            required_inputs=["answer"],
            produced_outputs=["guardrail_results"],
            allowed_next_states=[AgentNode.FINAL, AgentNode.HUMAN_APPROVAL],
            guardrails=["insurance_output_compliance"],
            trace_fields=["trace_id", "guardrail_results", "final_state"],
        ),
    ],
)
