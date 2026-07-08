"""业务记忆存储接口与内存实现。

本模块是 Agent Core 的业务记忆 Data Plane：
- 保存从业者画像事实；
- 保存客户 KYC 事实；
- 保存 KYC 问题、分析运行、生成输出和事件记忆；
- 为本地 demo 和单元测试提供无需数据库的 InMemory 实现；
- 为后续 PostgreSQL / pgvector 落地保留同名接口。

内存版 store 不是生产数据库，但它先把生产约束固化进代码：
1. 所有读写都带 tenant_id，防止租户串数据；
2. 长期事实必须带 source_type 和 evidence_text；
3. 冲突事实不覆盖旧事实，而是关闭旧版本；
4. 审计日志只记录字段名、ID 和动作，不记录敏感正文。
"""

from __future__ import annotations

from typing import Protocol

from agent_core.memory.business_schemas import (
    AdvisorProfileFact,
    AgentSessionState,
    AnalysisRun,
    CustomerProfileFact,
    GeneratedOutput,
    KYCQuestion,
    MemoryEvent,
    OpportunityCase,
)
from agent_core.utils.time import utc_now_iso


class BusinessMemoryStore(Protocol):
    """业务记忆存储协议。

    后续 PostgreSQL 版本只要实现同一组方法，就可以替换内存版 store。
    Graph 节点不应该感知底层是内存、数据库还是远程服务。
    """

    audit_log: list[dict]

    def upsert_advisor_fact(self, fact: AdvisorProfileFact) -> AdvisorProfileFact:
        """写入或更新从业者画像事实。"""

    def upsert_customer_fact(self, fact: CustomerProfileFact) -> CustomerProfileFact:
        """写入或更新客户画像事实。"""

    def insert_memory_event(self, event: MemoryEvent) -> MemoryEvent:
        """写入一条事件记忆。"""

    def insert_analysis_run(self, run: AnalysisRun) -> AnalysisRun:
        """写入一次 KYC 分析运行记录。"""

    def insert_session_state(self, state: AgentSessionState) -> AgentSessionState:
        """写入一轮会话工作记忆快照。"""

    def insert_kyc_question(self, question: KYCQuestion) -> KYCQuestion:
        """写入一条 KYC 补问记录。"""

    def insert_generated_output(self, output: GeneratedOutput) -> GeneratedOutput:
        """写入一次生成输出记录。"""

    def upsert_opportunity_case(self, case: OpportunityCase) -> OpportunityCase:
        """写入或更新一个客户机会 case。"""

    def get_current_advisor_facts(self, tenant_id: str, advisor_id: str) -> list[AdvisorProfileFact]:
        """读取某租户下某从业者当前有效事实。"""

    def get_current_customer_facts(
        self,
        tenant_id: str,
        customer_id: str,
        *,
        certainty: str | None = None,
    ) -> list[CustomerProfileFact]:
        """读取某租户下某客户当前有效事实。"""

    def get_active_opportunity_case(
        self,
        tenant_id: str,
        advisor_id: str,
        customer_id: str,
    ) -> OpportunityCase | None:
        """读取某租户下某从业者和客户之间的 active case。"""

    def get_recent_events(
        self,
        tenant_id: str,
        *,
        opportunity_case_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        """读取最近事件记忆。"""

    def get_asked_focuses(self, tenant_id: str, opportunity_case_id: str) -> list[str]:
        """读取某 case 已经问过的 KYC 焦点。"""

    def get_latest_session_state(self, tenant_id: str, conversation_id: str) -> AgentSessionState | None:
        """读取某会话最近一次工作记忆快照。"""

    def resolve_conflict_and_close_old_fact(
        self,
        fact: CustomerProfileFact | AdvisorProfileFact,
    ) -> list[str]:
        """关闭与新事实冲突的旧事实，并返回被关闭的旧事实 ID。"""


class InMemoryBusinessMemoryStore:
    """可测试的内存版业务记忆 store。

    内存结构按 tenant_id 分隔查询范围，即使不同租户使用相同 advisor_id/customer_id，
    也不会读到对方的数据。生产版本可以把这些列表换成数据库表，但业务约束应保持一致。
    """

    def __init__(self) -> None:
        self.advisor_facts: list[AdvisorProfileFact] = []
        self.customer_facts: list[CustomerProfileFact] = []
        self.opportunity_cases: list[OpportunityCase] = []
        self.memory_events: list[MemoryEvent] = []
        self.analysis_runs: list[AnalysisRun] = []
        self.session_states: list[AgentSessionState] = []
        self.kyc_questions: list[KYCQuestion] = []
        self.generated_outputs: list[GeneratedOutput] = []
        self.audit_log: list[dict] = []

    def upsert_advisor_fact(self, fact: AdvisorProfileFact) -> AdvisorProfileFact:
        """写入从业者事实；同 key 冲突时关闭旧版本。"""
        self._require_fact_evidence(fact.source_type, fact.evidence_text)
        closed_ids = self.resolve_conflict_and_close_old_fact(fact)
        existing = self._find_same_current_advisor_fact(fact)
        if existing:
            existing.confidence = max(existing.confidence, fact.confidence)
            existing.evidence_text = fact.evidence_text
            existing.source_type = fact.source_type
            existing.updated_at = utc_now_iso()
            self._audit("advisor_fact_updated", fact.tenant_id, "advisor_profile_facts", existing.id, ["fact_key"])
            return existing
        self.advisor_facts.append(fact)
        self._audit(
            "advisor_fact_inserted",
            fact.tenant_id,
            "advisor_profile_facts",
            fact.id,
            ["fact_key", "confidence", *closed_ids],
        )
        return fact

    def upsert_customer_fact(self, fact: CustomerProfileFact) -> CustomerProfileFact:
        """写入客户事实；不确定事实仍保留 uncertain，不能混入 confirmed。"""
        self._require_fact_evidence(fact.source_type, fact.evidence_text)
        closed_ids = self.resolve_conflict_and_close_old_fact(fact)
        existing = self._find_same_current_customer_fact(fact)
        if existing:
            existing.confidence = max(existing.confidence, fact.confidence)
            existing.evidence_text = fact.evidence_text
            existing.source_type = fact.source_type
            existing.updated_at = utc_now_iso()
            self._audit("customer_fact_updated", fact.tenant_id, "customer_profile_facts", existing.id, ["fact_key"])
            return existing
        self.customer_facts.append(fact)
        self._audit(
            "customer_fact_inserted",
            fact.tenant_id,
            "customer_profile_facts",
            fact.id,
            ["fact_key", "certainty", "sensitivity_level", *closed_ids],
        )
        return fact

    def insert_memory_event(self, event: MemoryEvent) -> MemoryEvent:
        """写入事件记忆；审计只记录 event_type，不记录完整 payload。"""
        self.memory_events.append(event)
        self._audit("memory_event_inserted", event.tenant_id, "memory_events", event.id, ["event_type"])
        return event

    def insert_analysis_run(self, run: AnalysisRun) -> AnalysisRun:
        """写入 KYC 分析运行；output_json 留在业务表，审计只记关键字段。"""
        self.analysis_runs.append(run)
        self._audit("analysis_run_inserted", run.tenant_id, "analysis_runs", run.id, ["information_status", "scores"])
        return run

    def insert_session_state(self, state: AgentSessionState) -> AgentSessionState:
        """写入一轮工作记忆快照；不覆盖历史快照。"""
        self.session_states.append(state)
        self._audit("session_state_inserted", state.tenant_id, "agent_session_states", state.id, ["conversation_id"])
        return state

    def insert_kyc_question(self, question: KYCQuestion) -> KYCQuestion:
        """写入 KYC 补问；相同 case 下同一 focus 只保留第一条 asked 记录。"""
        for existing in self.kyc_questions:
            if (
                existing.tenant_id == question.tenant_id
                and existing.opportunity_case_id == question.opportunity_case_id
                and existing.focus_key == question.focus_key
            ):
                return existing
        self.kyc_questions.append(question)
        self._audit("kyc_question_inserted", question.tenant_id, "kyc_questions", question.id, ["focus_key", "round_no"])
        return question

    def insert_generated_output(self, output: GeneratedOutput) -> GeneratedOutput:
        """写入生成输出；用于追踪 compact_context 到最终话术的映射。"""
        self.generated_outputs.append(output)
        self._audit(
            "generated_output_inserted",
            output.tenant_id,
            "generated_outputs",
            output.id,
            ["output_type", "used_case_pattern_ids"],
        )
        return output

    def upsert_opportunity_case(self, case: OpportunityCase) -> OpportunityCase:
        """写入或更新 active case；按 tenant/advisor/customer/case_id 隔离。"""
        for index, existing in enumerate(self.opportunity_cases):
            if existing.tenant_id == case.tenant_id and existing.id == case.id:
                self.opportunity_cases[index] = case
                self._audit("opportunity_case_updated", case.tenant_id, "opportunity_cases", case.id, ["case_status"])
                return case
        self.opportunity_cases.append(case)
        self._audit("opportunity_case_inserted", case.tenant_id, "opportunity_cases", case.id, ["case_status"])
        return case

    def get_current_advisor_facts(self, tenant_id: str, advisor_id: str) -> list[AdvisorProfileFact]:
        """只返回同租户、同从业者、当前有效的画像事实。"""
        return [
            fact
            for fact in self.advisor_facts
            if fact.tenant_id == tenant_id and fact.advisor_id == advisor_id and fact.is_current
        ]

    def get_current_customer_facts(
        self,
        tenant_id: str,
        customer_id: str,
        *,
        certainty: str | None = None,
    ) -> list[CustomerProfileFact]:
        """只返回同租户、同客户、当前有效的 KYC 事实。"""
        facts = [
            fact
            for fact in self.customer_facts
            if fact.tenant_id == tenant_id and fact.customer_id == customer_id and fact.is_current
        ]
        if certainty is not None:
            facts = [fact for fact in facts if fact.certainty == certainty]
        return facts

    def get_active_opportunity_case(
        self,
        tenant_id: str,
        advisor_id: str,
        customer_id: str,
    ) -> OpportunityCase | None:
        """读取 active case；同名 ID 在不同租户下不会互相命中。"""
        for case in reversed(self.opportunity_cases):
            if (
                case.tenant_id == tenant_id
                and case.advisor_id == advisor_id
                and case.customer_id == customer_id
                and case.case_status == "active"
            ):
                return case
        return None

    def get_recent_events(
        self,
        tenant_id: str,
        *,
        opportunity_case_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        """按时间倒序返回最近事件，默认只看租户边界。"""
        events = [event for event in self.memory_events if event.tenant_id == tenant_id]
        if opportunity_case_id is not None:
            events = [event for event in events if event.opportunity_case_id == opportunity_case_id]
        return list(reversed(events))[:limit]

    def get_asked_focuses(self, tenant_id: str, opportunity_case_id: str) -> list[str]:
        """从 KYCQuestion 表读取已问焦点，避免靠字符串拼接判断重复追问。"""
        focuses: list[str] = []
        for question in self.kyc_questions:
            if question.tenant_id == tenant_id and question.opportunity_case_id == opportunity_case_id:
                if question.focus_key not in focuses:
                    focuses.append(question.focus_key)
        return focuses

    def get_latest_session_state(self, tenant_id: str, conversation_id: str) -> AgentSessionState | None:
        """返回某会话最近写入的工作记忆快照。"""
        for state in reversed(self.session_states):
            if state.tenant_id == tenant_id and state.conversation_id == conversation_id:
                return state
        return None

    def resolve_conflict_and_close_old_fact(
        self,
        fact: CustomerProfileFact | AdvisorProfileFact,
    ) -> list[str]:
        """关闭同租户、同主体、同 fact_key 但值不同的当前事实。"""
        closed_ids: list[str] = []
        now = utc_now_iso()
        if isinstance(fact, CustomerProfileFact):
            for old in self.customer_facts:
                if (
                    old.tenant_id == fact.tenant_id
                    and old.customer_id == fact.customer_id
                    and old.fact_key == fact.fact_key
                    and old.is_current
                    and not self._same_customer_fact_value(old, fact)
                ):
                    old.is_current = False
                    old.valid_to = now
                    old.updated_at = now
                    closed_ids.append(old.id)
        else:
            for old in self.advisor_facts:
                if (
                    old.tenant_id == fact.tenant_id
                    and old.advisor_id == fact.advisor_id
                    and old.fact_key == fact.fact_key
                    and old.is_current
                    and old.fact_value != fact.fact_value
                ):
                    old.is_current = False
                    old.valid_to = now
                    old.updated_at = now
                    closed_ids.append(old.id)
        if closed_ids:
            self._audit("fact_conflict_closed", fact.tenant_id, "profile_facts", ",".join(closed_ids), ["fact_key"])
        return closed_ids

    def _find_same_current_advisor_fact(self, fact: AdvisorProfileFact) -> AdvisorProfileFact | None:
        """查找同租户、同从业者、同 key、同 value 的当前事实。"""
        for existing in self.advisor_facts:
            if (
                existing.tenant_id == fact.tenant_id
                and existing.advisor_id == fact.advisor_id
                and existing.fact_key == fact.fact_key
                and existing.fact_value == fact.fact_value
                and existing.is_current
            ):
                return existing
        return None

    def _find_same_current_customer_fact(self, fact: CustomerProfileFact) -> CustomerProfileFact | None:
        """查找同租户、同客户、同 key、同 value、同 certainty 的当前事实。"""
        for existing in self.customer_facts:
            if (
                existing.tenant_id == fact.tenant_id
                and existing.customer_id == fact.customer_id
                and existing.fact_key == fact.fact_key
                and existing.certainty == fact.certainty
                and existing.is_current
                and self._same_customer_fact_value(existing, fact)
            ):
                return existing
        return None

    @staticmethod
    def _same_customer_fact_value(left: CustomerProfileFact, right: CustomerProfileFact) -> bool:
        """比较客户事实的业务值；normalized_value 存在时优先使用标准化值。"""
        left_value = left.normalized_value if left.normalized_value is not None else left.fact_value
        right_value = right.normalized_value if right.normalized_value is not None else right.fact_value
        return left_value == right_value and left.certainty == right.certainty

    @staticmethod
    def _require_fact_evidence(source_type: str, evidence_text: str) -> None:
        """长期事实必须能回到明确证据，避免模型把建议写成客户事实。"""
        if not source_type:
            raise ValueError("写入长期事实必须提供 source_type。")
        if not evidence_text or not evidence_text.strip():
            raise ValueError("写入长期事实必须提供 evidence_text。")

    def _audit(self, action: str, tenant_id: str, table_name: str, record_id: str, fields: list[str]) -> None:
        """写入低敏审计日志，只记录字段名和动作，不记录完整敏感内容。"""
        self.audit_log.append(
            {
                "ts": utc_now_iso(),
                "action": action,
                "tenant_id": tenant_id,
                "table_name": table_name,
                "record_id": record_id,
                "fields": fields,
            }
        )
