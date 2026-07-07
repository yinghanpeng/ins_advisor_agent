"""Workflow step contracts for Agent Core and Dify mapping."""

# 文件说明：
# - 本文件属于 Workflow 层，负责请求/响应契约、step contract 或执行引擎。
# - 模型输出进入下游逻辑前，应先通过这里定义的结构化契约。
from __future__ import annotations

from agent_core.graph.state import AgentNode
from agent_core.workflow.contracts import WorkflowContract, WorkflowStepContract


BREAK_ICE_ASSISTANT_STEPS = [
    "classify_intent",
    "extract_customer_kyc",
    "extract_sales_pain",
    "classify_scene",
    "retrieve_sales_intelligence",
    "retrieve_external_context_if_needed",
    "build_context",
    "generate_response",
    "compliance_review",
    "final_response",
]


BREAK_ICE_ASSISTANT_CONTRACT = WorkflowContract(
    name="break_ice_assistant_workflow",
    entry_state=AgentNode.CLASSIFY_INTENT,
    final_states=[AgentNode.FINAL, AgentNode.HUMAN_APPROVAL, AgentNode.ERROR],
    steps=[
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
        WorkflowStepContract(
            name="route_domain_workflow",
            state=AgentNode.DOMAIN_WORKFLOW_ROUTING,
            description="Route insurance advisor requests to the proper domain workflow.",
            required_inputs=["intent", "domain_skill"],
            produced_outputs=["sales_route"],
            allowed_next_states=[AgentNode.SALES_INTELLIGENCE_ROUTING, AgentNode.BUILD_CONTEXT],
            trace_fields=["trace_id", "domain_skill", "sales_route"],
        ),
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
        WorkflowStepContract(
            name="build_context",
            state=AgentNode.BUILD_CONTEXT,
            description="Build compact context with source boundaries and evidence digest.",
            required_inputs=["retrieved_context"],
            produced_outputs=["sales_insight_digest"],
            allowed_next_states=[AgentNode.GENERATE_RESPONSE],
            trace_fields=["trace_id", "sales_insight_digest"],
        ),
        WorkflowStepContract(
            name="generate_response",
            state=AgentNode.GENERATE_RESPONSE,
            description="Generate a domain answer from compact context.",
            required_inputs=["input_text", "sales_insight_digest"],
            produced_outputs=["answer"],
            allowed_next_states=[AgentNode.COMPLIANCE_REVIEW],
            trace_fields=["trace_id", "answer"],
        ),
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
