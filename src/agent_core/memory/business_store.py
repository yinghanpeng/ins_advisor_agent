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

from contextlib import contextmanager
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

    # audit_log 只保存动作、主体和字段名等低敏审计摘要。
    audit_log: list[dict]

    def transaction(self):
        """返回业务记忆 Unit of Work 上下文。"""

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
        """初始化按记录类型隔离的内存列表及低敏审计日志。"""

        # 分别保存顾问事实、客户事实和机会 Case；使用独立列表可避免不同记录类型混写。
        self.advisor_facts: list[AdvisorProfileFact] = []
        # 客户事实与顾问事实独立存储，避免主体类型混淆。
        self.customer_facts: list[CustomerProfileFact] = []
        # Opportunity Case 保存当前任务聚合状态。
        self.opportunity_cases: list[OpportunityCase] = []
        # 事件采用追加方式保存，便于测试回放完整业务时间线。
        self.memory_events: list[MemoryEvent] = []
        # 分析运行与业务事实分离，避免模型判断冒充客户事实。
        self.analysis_runs: list[AnalysisRun] = []
        # Session 快照、补问和生成结果同样保留历史版本，不在写入时覆盖旧值。
        self.session_states: list[AgentSessionState] = []
        # KYC 问题列表支持按 Case/focus 去重。
        self.kyc_questions: list[KYCQuestion] = []
        # 生成输出单独归档，便于与 Case Outcome 做效果闭环。
        self.generated_outputs: list[GeneratedOutput] = []
        # 审计日志只记录低敏元数据，由各写入方法通过 _audit 统一追加。
        self.audit_log: list[dict] = []

    @contextmanager
    def transaction(self):
        """内存实现提供同名 Unit of Work 接口，便于节点无差别调用。"""
        # 测试 Store 不具备数据库回滚，但同一调用栈内保持一致接口。
        yield self

    def upsert_advisor_fact(self, fact: AdvisorProfileFact) -> AdvisorProfileFact:
        """写入从业者事实；同 key 冲突时关闭旧版本。"""
        # 长期事实必须同时具备来源类型和非空证据，否则在进入 Store 前立即拒绝。
        self._require_fact_evidence(fact.source_type, fact.evidence_text)
        # 先关闭同一事实键下值不同的当前版本，保留可追溯的历史记录 ID。
        closed_ids = self.resolve_conflict_and_close_old_fact(fact)
        # 再判断是否已存在“同键同值”的当前事实，避免重复插入相同业务事实。
        existing = self._find_same_current_advisor_fact(fact)
        # 已找到同键同值事实时走就地证据增强分支，不创建重复版本。
        if existing:
            # 对重复证据采用较高置信度，避免新一轮低置信抽取削弱已有事实。
            existing.confidence = max(existing.confidence, fact.confidence)
            # 使用本轮最新证据和来源更新事实的可解释信息。
            existing.evidence_text = fact.evidence_text
            # 同步刷新最新事实来源类型。
            existing.source_type = fact.source_type
            # 记录本轮证据增强的更新时间。
            existing.updated_at = utc_now_iso()
            # 审计只记录 fact_key 字段发生更新，不写入事实值或证据正文。
            self._audit("advisor_fact_updated", fact.tenant_id, "advisor_profile_facts", existing.id, ["fact_key"])
            # 返回数据库语义上的已有事实对象。
            return existing
        # 没有同值事实时追加新版本，旧冲突版本已在上一步被关闭。
        self.advisor_facts.append(fact)
        # 审计中带上被关闭的记录 ID，便于定位一次事实版本切换。
        self._audit(
            "advisor_fact_inserted",
            fact.tenant_id,
            "advisor_profile_facts",
            fact.id,
            ["fact_key", "confidence", *closed_ids],
        )
        # 新事实写入路径返回传入对象。
        return fact

    def upsert_customer_fact(self, fact: CustomerProfileFact) -> CustomerProfileFact:
        """写入客户事实；不确定事实仍保留 uncertain，不能混入 confirmed。"""
        # 与顾问事实相同，客户长期事实必须提供可追溯的来源和原文证据。
        self._require_fact_evidence(fact.source_type, fact.evidence_text)
        # 关闭同一事实键下业务值不同的旧版本，保留版本历史而非物理删除。
        closed_ids = self.resolve_conflict_and_close_old_fact(fact)
        # certainty 也是客户事实等价性的一部分，confirmed 与 uncertain 不会被合并。
        existing = self._find_same_current_customer_fact(fact)
        # 已存在完整等价的当前客户事实时，只更新证据和置信度。
        if existing:
            # 相同事实重复命中时只提高或保持置信度，不允许置信度倒退。
            existing.confidence = max(existing.confidence, fact.confidence)
            # 刷新本轮证据、来源和更新时间，保留事实 ID 的稳定性。
            existing.evidence_text = fact.evidence_text
            # 同步刷新客户事实来源类型。
            existing.source_type = fact.source_type
            # 记录客户事实本轮更新时间。
            existing.updated_at = utc_now_iso()
            # 审计事件不记录 KYC 事实值，避免敏感信息进入普通日志。
            self._audit("customer_fact_updated", fact.tenant_id, "customer_profile_facts", existing.id, ["fact_key"])
            # 返回被证据增强的已有客户事实。
            return existing
        # 未发现同值当前事实时，将新事实作为当前版本追加。
        self.customer_facts.append(fact)
        # 记录客户事实插入及被关闭版本的低敏审计摘要。
        self._audit(
            "customer_fact_inserted",
            fact.tenant_id,
            "customer_profile_facts",
            fact.id,
            ["fact_key", "certainty", "sensitivity_level", *closed_ids],
        )
        # 新客户事实写入路径返回传入对象。
        return fact

    def insert_memory_event(self, event: MemoryEvent) -> MemoryEvent:
        """写入事件记忆；审计只记录 event_type，不记录完整 payload。"""
        # 事件采用 append-only 方式保存，确保业务时间线可回放。
        self.memory_events.append(event)
        # 只审计事件类型字段，故意不把可能含业务敏感信息的 payload 写入日志。
        self._audit("memory_event_inserted", event.tenant_id, "memory_events", event.id, ["event_type"])
        # 返回刚追加的事件记录。
        return event

    def insert_analysis_run(self, run: AnalysisRun) -> AnalysisRun:
        """写入 KYC 分析运行；output_json 留在业务表，审计只记关键字段。"""
        # 每次分析都是独立审计记录，不能覆盖此前模型/规则运行结果。
        self.analysis_runs.append(run)
        # 审计只标记状态和分数字段，具体分析 JSON 仅保留在受控业务记录中。
        self._audit("analysis_run_inserted", run.tenant_id, "analysis_runs", run.id, ["information_status", "scores"])
        # 返回刚追加的分析运行。
        return run

    def insert_session_state(self, state: AgentSessionState) -> AgentSessionState:
        """写入一轮工作记忆快照；不覆盖历史快照。"""
        # 每轮保存一个不可变语义的快照，读取时再选择最后写入的版本。
        self.session_states.append(state)
        # 审计只暴露关联会话字段名，不记录画像快照正文。
        self._audit("session_state_inserted", state.tenant_id, "agent_session_states", state.id, ["conversation_id"])
        # 返回刚追加的 Session Snapshot。
        return state

    def insert_kyc_question(self, question: KYCQuestion) -> KYCQuestion:
        """写入 KYC 补问；相同 case 下同一 focus 只保留第一条 asked 记录。"""
        # 顺序扫描当前补问，使用 tenant + case + focus 作为业务幂等键。
        for existing in self.kyc_questions:
            # 三个字段全部相同才表示同一 Case 已经问过该 KYC 焦点。
            if (
                existing.tenant_id == question.tenant_id
                and existing.opportunity_case_id == question.opportunity_case_id
                and existing.focus_key == question.focus_key
            ):
                # 已经问过同一焦点时返回原记录，防止一次重试被计为新的补问轮次。
                # 返回已有问题使调用方使用稳定 KYCQuestion ID。
                return existing
        # 只有首次出现的焦点才写入问题历史。
        self.kyc_questions.append(question)
        # 审计保留焦点和轮次字段名，问题正文不会进入低敏日志。
        self._audit("kyc_question_inserted", question.tenant_id, "kyc_questions", question.id, ["focus_key", "round_no"])
        # 首次问题写入返回传入对象。
        return question

    def insert_generated_output(self, output: GeneratedOutput) -> GeneratedOutput:
        """写入生成输出；用于追踪 compact_context 到最终话术的映射。"""
        # 生成结果按轮次追加，以便后续关联实际结果并做离线评测。
        self.generated_outputs.append(output)
        # 审计只记录输出类型和引用模式 ID，不复制最终话术正文。
        self._audit(
            "generated_output_inserted",
            output.tenant_id,
            "generated_outputs",
            output.id,
            ["output_type", "used_case_pattern_ids"],
        )
        # 返回刚追加的生成输出。
        return output

    def upsert_opportunity_case(self, case: OpportunityCase) -> OpportunityCase:
        """写入或更新 active case；按 tenant/advisor/customer/case_id 隔离。"""
        # 枚举列表并按 tenant + case_id 查找现有记录，禁止跨租户更新同名 ID。
        for index, existing in enumerate(self.opportunity_cases):
            # tenant 和记录 ID 必须同时相同才允许替换 Case 聚合状态。
            if existing.tenant_id == case.tenant_id and existing.id == case.id:
                # Case 是当前聚合状态，因此同 ID 写入采用整体替换而不是追加新记录。
                self.opportunity_cases[index] = case
                # 更新审计只标记状态字段，不记录完整客户机会画像。
                self._audit("opportunity_case_updated", case.tenant_id, "opportunity_cases", case.id, ["case_status"])
                # 更新路径返回已替换的 Case 对象。
                return case
        # 未命中同租户 Case 时创建新机会记录。
        self.opportunity_cases.append(case)
        # 记录新 Case 插入动作，不保存画像正文。
        self._audit("opportunity_case_inserted", case.tenant_id, "opportunity_cases", case.id, ["case_status"])
        # 创建路径返回新 Case。
        return case

    def get_current_advisor_facts(self, tenant_id: str, advisor_id: str) -> list[AdvisorProfileFact]:
        """只返回同租户、同从业者、当前有效的画像事实。"""
        # 三个条件共同形成读取边界：租户相同、主体相同且事实版本仍为 current。
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
        # 先按租户、客户和 current 标记收窄事实集合，杜绝跨客户记忆混入。
        facts = [
            fact
            for fact in self.customer_facts
            if fact.tenant_id == tenant_id and fact.customer_id == customer_id and fact.is_current
        ]
        # 调用方可选地只取 confirmed 或 uncertain；不传时保留两类事实供上层决策。
        if certainty is not None:
            # 按调用方指定 certainty 进一步过滤当前事实集合。
            facts = [fact for fact in facts if fact.certainty == certainty]
        # 返回应用全部强制和可选过滤后的客户事实列表。
        return facts

    def get_active_opportunity_case(
        self,
        tenant_id: str,
        advisor_id: str,
        customer_id: str,
    ) -> OpportunityCase | None:
        """读取 active case；同名 ID 在不同租户下不会互相命中。"""
        # 反向遍历使多个历史记录同时存在时优先选择最后更新/插入的 active Case。
        for case in reversed(self.opportunity_cases):
            # 同时校验租户、顾问、客户及 active 状态，形成完整读取边界。
            if (
                case.tenant_id == tenant_id
                and case.advisor_id == advisor_id
                and case.customer_id == customer_id
                and case.case_status == "active"
            ):
                # 只有租户、顾问、客户和 active 状态全部一致才允许返回该 Case。
                return case
        # 没有活动机会时显式返回 None，由上层决定是否创建新 Case。
        return None

    def get_recent_events(
        self,
        tenant_id: str,
        *,
        opportunity_case_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        """按时间倒序返回最近事件，默认只看租户边界。"""
        # 第一步始终按 tenant_id 过滤，这是任何可选 Case 条件之前的强制隔离边界。
        events = [event for event in self.memory_events if event.tenant_id == tenant_id]
        # 传入 Case ID 时进一步限制到当前机会；未传时允许返回租户级最近事件。
        if opportunity_case_id is not None:
            # 只保留与指定 Case 完全关联的事件。
            events = [event for event in events if event.opportunity_case_id == opportunity_case_id]
        # 内存列表按写入时间正序排列，因此反转后截取 limit 得到最近事件。
        return list(reversed(events))[:limit]

    def get_asked_focuses(self, tenant_id: str, opportunity_case_id: str) -> list[str]:
        """从 KYCQuestion 表读取已问焦点，避免靠字符串拼接判断重复追问。"""
        # 使用列表保持首次提问顺序，后续生成器可按顺序理解已经覆盖的 KYC 主题。
        focuses: list[str] = []
        # 扫描当前租户和 Case 的全部问题，不读取其他客户机会的提问记录。
        for question in self.kyc_questions:
            # 只有租户和 Case 都相同的问题才计入已问焦点。
            if question.tenant_id == tenant_id and question.opportunity_case_id == opportunity_case_id:
                # 同焦点即使历史数据重复，也只向上层返回一次。
                if question.focus_key not in focuses:
                    # 首次遇到该焦点时按提问顺序追加。
                    focuses.append(question.focus_key)
        # 返回保持首次提问顺序的唯一 focus 列表。
        return focuses

    def get_latest_session_state(self, tenant_id: str, conversation_id: str) -> AgentSessionState | None:
        """返回某会话最近写入的工作记忆快照。"""
        # 反向遍历 append-only 快照，首次匹配项即为该会话最新状态。
        for state in reversed(self.session_states):
            # tenant 与 conversation 同时匹配才能返回该快照。
            if state.tenant_id == tenant_id and state.conversation_id == conversation_id:
                # 反向扫描首次命中即为最新快照。
                return state
        # 会话尚未持久化快照时返回 None，避免伪造空状态与真实快照混淆。
        return None

    def resolve_conflict_and_close_old_fact(
        self,
        fact: CustomerProfileFact | AdvisorProfileFact,
    ) -> list[str]:
        """关闭同租户、同主体、同 fact_key 但值不同的当前事实。"""
        # 收集被关闭记录的 ID，供写入审计和调用方追踪版本替换关系。
        closed_ids: list[str] = []
        # 同一次冲突处理共用时间戳，保证 valid_to 与 updated_at 完全一致。
        now = utc_now_iso()
        # 客户事实和顾问事实使用不同主体键及值比较规则，先按具体模型分支。
        if isinstance(fact, CustomerProfileFact):
            # 查找同租户、同客户、同键且仍有效，但业务值已不同的旧事实。
            for old in self.customer_facts:
                # 仅完整匹配主体/键/current 且业务值变化时关闭旧客户事实。
                if (
                    old.tenant_id == fact.tenant_id
                    and old.customer_id == fact.customer_id
                    and old.fact_key == fact.fact_key
                    and old.is_current
                    and not self._same_customer_fact_value(old, fact)
                ):
                    # 逻辑关闭旧版本而不删除，使历史事实和修正轨迹仍可审计。
                    old.is_current = False
                    # 写入统一事实失效时间。
                    old.valid_to = now
                    # 更新时间与失效时间保持一致。
                    old.updated_at = now
                    # 收集被关闭旧事实 ID。
                    closed_ids.append(old.id)
        # 顾问事实使用原始值比较的独立冲突处理分支。
        else:
            # 顾问事实没有 normalized_value/certainty，直接比较原始 fact_value。
            for old in self.advisor_facts:
                # 仅完整匹配主体/键/current 且原始值变化时关闭旧顾问事实。
                if (
                    old.tenant_id == fact.tenant_id
                    and old.advisor_id == fact.advisor_id
                    and old.fact_key == fact.fact_key
                    and old.is_current
                    and old.fact_value != fact.fact_value
                ):
                    # 为冲突的顾问事实写入统一失效时间并从 current 集合中移除。
                    old.is_current = False
                    # 固化顾问事实失效时间。
                    old.valid_to = now
                    # 同步刷新顾问事实更新时间。
                    old.updated_at = now
                    # 收集被关闭顾问事实 ID。
                    closed_ids.append(old.id)
        # 只有真实关闭了旧记录才追加冲突审计，避免空操作制造噪声。
        if closed_ids:
            # 审计只记录旧 ID 和 fact_key 字段名，不记录冲突值。
            self._audit("fact_conflict_closed", fact.tenant_id, "profile_facts", ",".join(closed_ids), ["fact_key"])
        # 返回本次实际逻辑关闭的旧事实 ID。
        return closed_ids

    def _find_same_current_advisor_fact(self, fact: AdvisorProfileFact) -> AdvisorProfileFact | None:
        """查找同租户、同从业者、同 key、同 value 的当前事实。"""
        # 逐条匹配事实的完整业务等价键；任一条件不同都视为独立版本。
        for existing in self.advisor_facts:
            # 租户、主体、键、值和 current 五个条件共同定义顾问事实等价性。
            if (
                existing.tenant_id == fact.tenant_id
                and existing.advisor_id == fact.advisor_id
                and existing.fact_key == fact.fact_key
                and existing.fact_value == fact.fact_value
                and existing.is_current
            ):
                # 找到完整等价的当前顾问事实后立即返回。
                return existing
        # 未找到等价事实时返回 None，upsert 调用方会追加新记录。
        return None

    def _find_same_current_customer_fact(self, fact: CustomerProfileFact) -> CustomerProfileFact | None:
        """查找同租户、同客户、同 key、同 value、同 certainty 的当前事实。"""
        # 客户事实除主体和键外还比较 certainty 与标准化后的业务值。
        for existing in self.customer_facts:
            # 先匹配主体/键/certainty/current，再调用 helper 比较标准化业务值。
            if (
                existing.tenant_id == fact.tenant_id
                and existing.customer_id == fact.customer_id
                and existing.fact_key == fact.fact_key
                and existing.certainty == fact.certainty
                and existing.is_current
                and self._same_customer_fact_value(existing, fact)
            ):
                # 找到完整等价的当前客户事实后立即返回。
                return existing
        # 没有完整等价的当前事实时由调用方执行新版本插入。
        return None

    @staticmethod
    def _same_customer_fact_value(left: CustomerProfileFact, right: CustomerProfileFact) -> bool:
        """比较客户事实的业务值；normalized_value 存在时优先使用标准化值。"""
        # 标准化值消除了同义表达差异；没有标准化结果时才回退原始抽取值。
        left_value = left.normalized_value if left.normalized_value is not None else left.fact_value
        # 右侧事实采用相同标准化值优先规则。
        right_value = right.normalized_value if right.normalized_value is not None else right.fact_value
        # certainty 必须同时相同，避免把推测事实误判为已确认事实。
        return left_value == right_value and left.certainty == right.certainty

    @staticmethod
    def _require_fact_evidence(source_type: str, evidence_text: str) -> None:
        """长期事实必须能回到明确证据，避免模型把建议写成客户事实。"""
        # 空 source_type 无法说明事实来源，应在任何内存变更发生前阻断。
        if not source_type:
            # 显式异常阻止不可追溯事实进入长期记忆。
            raise ValueError("写入长期事实必须提供 source_type。")
        # 空白证据同样不可接受，strip 可拦截只包含空格或换行的字符串。
        if not evidence_text or not evidence_text.strip():
            # 显式异常阻止无原文依据的推测污染长期事实。
            raise ValueError("写入长期事实必须提供 evidence_text。")

    def _audit(self, action: str, tenant_id: str, table_name: str, record_id: str, fields: list[str]) -> None:
        """写入低敏审计日志，只记录字段名和动作，不记录完整敏感内容。"""
        # 统一生成 UTC 时间并追加结构化元数据；fields 只包含字段名而非字段值。
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
