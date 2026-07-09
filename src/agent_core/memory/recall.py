"""长期记忆按需召回、Hybrid Search 与 Rerank。

主流 Agent 的记忆设计通常把“是否召回”和“怎么召回”拆开：
1. 先判断当前请求是否真的需要长期记忆；
2. 需要时才生成 memory queries；
3. 使用关键词 + 向量近似 + metadata 的 hybrid search；
4. 再按当前性、确定性、业务层级和文本相关性 rerank；
5. 最后只把 TopK 摘要放入上下文，而不是把整份长期记忆塞给模型。

生产链路分两段：
1. MemoryRecallRuleEngine 用规则在毫秒级判断“必须召回 / 明确跳过 / 需要模型判断”；
2. 只有 ambiguous case 才调用 memory_recall_decision 模型，并用 Pydantic 校验 JSON；
3. should_recall=true 时，ProductionMemoryRetriever 通过真实 embedding + PostgreSQL pgvector
   + hybrid score + reranker 召回长期记忆。

这样设计是为了避免每轮都读取长期画像，既降低延迟和成本，也减少把无关历史事实带入回答的风险。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_core.config.runtime import MemoryConfig, RetrievalConfig
from agent_core.memory.business_schemas import (
    AdvisorProfileFact,
    CustomerProfileFact,
    MemoryEvent,
    OpportunityCase,
)
from agent_core.models.client import (
    OpenAICompatibleChatClient,
    OpenAICompatibleEmbeddingClient,
    RerankerClient,
)
from agent_core.persistence.postgres import PersistedMemoryHit, PostgresAgentRepository
from agent_core.rag.retriever import HybridRetriever
from agent_core.rag.schemas import (
    DocumentMetadata,
    MetadataFilter,
    RetrievalDocument,
    RetrievalQuery,
    RetrievalResult,
)


MemoryRecallLayer = Literal[
    "preference",
    "customer_profile",
    "advisor_profile",
    "case_state",
    "memory_event",
    "domain_fact",
]


RuleDecisionStatus = Literal[
    "must_recall",
    "skip_recall",
    "ambiguous",
]

MemoryRecallDecisionStatus = Literal[
    "rule_must_recall",
    "rule_skip_recall",
    "model_decision",
    "failed_schema_validation",
    "model_unavailable_safe_skip",
]


class MemoryRecallDecision(BaseModel):
    """长期记忆召回决策。"""

    should_recall: bool = Field(..., description="本轮是否需要召回长期记忆。")
    recall_layers: list[MemoryRecallLayer] = Field(
        default_factory=list,
        description="本轮允许召回的长期记忆层，例如 preference、customer_profile、advisor_profile。",
    )
    reason: str = Field(default="", description="触发或跳过长期记忆召回的原因。")
    queries: list[RetrievalQuery] = Field(default_factory=list, description="用于长期记忆检索的 query rewrite 列表。")
    top_k: int = Field(default=5, ge=1, le=20, description="长期记忆召回最多返回条数。")
    score_threshold: float = Field(default=0.08, ge=0, le=1, description="低于该最终分数的记忆不进入上下文。")
    status: MemoryRecallDecisionStatus = Field(
        default="rule_skip_recall",
        description="召回决策来源或失败状态，用于审计规则和模型各自的命中情况。",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="长期记忆检索 filters，例如 tenant_id、user_id、session_id、case_id、max_risk_level。",
    )
    confidence: float = Field(default=0.0, ge=0, le=1, description="召回决策置信度。规则强命中通常接近 1。")
    latency_budget_ms: int = Field(default=1200, description="召回决策允许消耗的延迟预算。")


class MemoryRecallItem(BaseModel):
    """一条经过 hybrid search 和 rerank 的长期记忆结果。"""

    layer: MemoryRecallLayer = Field(..., description="该记忆结果所属层级。")
    source_id: str = Field(..., description="记忆来源 ID，用于审计和回放。")
    chunk_id: str = Field(..., description="记忆片段 ID，用于定位具体事实或偏好项。")
    content: str = Field(..., description="进入上下文前的短摘要，不应包含 PII 或原始长对话。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="记忆结果的业务 metadata，例如 fact_key、certainty。")
    lexical_score: float = Field(default=0.0, description="关键词检索得分。")
    vector_score: float = Field(default=0.0, description="向量近似检索得分。")
    metadata_score: float = Field(default=0.0, description="metadata 匹配得分。")
    rerank_score: float = Field(default=0.0, description="二阶段 rerank 后的最终得分。")


class MemoryRecallResult(BaseModel):
    """长期记忆召回结果。"""

    decision: MemoryRecallDecision = Field(..., description="本次召回使用的决策对象。")
    items: list[MemoryRecallItem] = Field(default_factory=list, description="最终进入上下文候选的 TopK 记忆。")
    compact_summary: dict[str, Any] = Field(default_factory=dict, description="按层级压缩后的记忆摘要。")


class MemoryRecallRuleResult(BaseModel):
    """规则引擎对长期记忆是否召回的初步判断。"""

    status: RuleDecisionStatus = Field(..., description="规则判断结果：必须召回、跳过或交给模型判断。")
    recall_layers: list[MemoryRecallLayer] = Field(default_factory=list, description="规则命中的召回层级。")
    reason: str = Field(default="", description="规则命中原因，用于 trace。")
    queries: list[str] = Field(default_factory=list, description="规则生成的检索 query。")


class MemoryRecallDecisionModelOutput(BaseModel):
    """memory_recall_decision 模型必须返回的结构化 JSON。"""

    should_recall: bool = Field(..., description="模型判断是否召回长期记忆。")
    recall_scopes: list[MemoryRecallLayer] = Field(default_factory=list, description="允许召回的长期记忆 scope。")
    reason: str = Field(..., description="模型决策原因。")
    queries: list[str] = Field(default_factory=list, description="模型生成的长期记忆检索 query。")
    filters: dict[str, Any] = Field(default_factory=dict, description="模型建议的检索 filters。")
    confidence: float = Field(..., ge=0, le=1, description="模型决策置信度。")
    latency_budget_ms: int = Field(default=1200, description="模型认为本次召回可接受的延迟预算。")


# 记忆层级（MemoryRecallLayer）→ HybridRetriever / MetadataFilter 使用的 library 名。
# 决策层输出 recall_layers，检索层通过此映射转换为 libraries 过滤条件。
LAYER_TO_LIBRARY: dict[str, str] = {
    "preference": "preference_memory",
    "customer_profile": "customer_profile_memory",
    "advisor_profile": "advisor_profile_memory",
    "case_state": "case_memory",
    "memory_event": "event_memory",
    "domain_fact": "domain_fact_memory",
}


class MemoryRecallRuleEngine:
    """长期记忆召回的规则快速判断层。

    规则层目标延迟是毫秒级，专门处理确定性很强的情况：
    - 用户显式要求“按我之前说的”；
    - 多轮指代依赖 last_entity / last_case_id；
    - 保险 KYC 等领域任务必然依赖客户画像、从业者画像或 case 状态；
    - 天气、计算、一次性百科问题明确不需要长期记忆。

    只有规则无法确定时，才调用模型做结构化判断。这样能避免每轮都调用模型，也能避免
    把无关长期记忆召回到当前上下文。
    """

    explicit_history_terms = [
        "按我喜欢的风格",
        "按我之前说的",
        "还记得我上次说的吗",
        "继续上次那个客户",
        "这个客户之前的情况",
        "根据他的画像",
        "结合我的偏好",
        "基于之前的信息",
        "不要重复问",
        "你应该知道",
    ]
    pronoun_terms = ["他", "她", "它", "这个客户", "刚才那个", "上次那个", "之前那个产品"]
    skip_terms = ["天气", "weather", "计算", "calculator"]
    personalization_terms = ["风格", "偏好", "格式", "保险偏好", "销售偏好", "从业画像", "客户画像"]
    business_terms = ["insurance_advisor", "insurance_kyc_coach_workflow", "sales_intelligence"]
    missing_slot_terms = [
        "customer_type",
        "family_status",
        "risk_preference",
        "budget",
        "product_preference",
        "advisor_stage",
        "current_case_status",
    ]

    def decide(
        self,
        *,
        input_text: str,
        workflow_name: str,
        intent: str | None,
        domain_skill: str | None,
        session_memory: dict[str, Any],
        metadata: dict[str, Any],
    ) -> MemoryRecallRuleResult:
        """规则层快速判断：必须召回 / 明确跳过 / 交给模型（ambiguous）。

        本方法目标延迟毫秒级，不调用 LLM。按优先级依次检查：

        1. **skip_recall** — 用户禁止、工具类意图（天气/计算）、算术表达式；
        2. **must_recall** — 命中历史引用、多轮指代、个性化关键词、保险业务、缺失槽位；
        3. **ambiguous** — 以上均不满足，交给 ``decide_long_term_memory_recall`` 的模型层。

        参数:
            input_text: 用户本轮原始输入。
            workflow_name: 当前工作流名，如 ``insurance_kyc_coach_workflow``。
            intent: 意图分类结果（可能为 None，restore_memory 阶段尚未 classify）。
            domain_skill: 领域 Skill，如 ``insurance_advisor``。
            session_memory: 短期 SESSION 记忆，含 last_entity / last_intent / last_case_id。
            metadata: 请求 metadata，含 missing_slots、case_id、allow_long_term_memory 等。

        返回:
            MemoryRecallRuleResult，status 三选一：must_recall / skip_recall / ambiguous。
        """
        text = input_text.strip()
        lowered = text.lower()

        # ── 第一优先级：明确跳过 ──
        # 用户显式禁止，或 metadata 关闭长期记忆开关。
        if metadata.get("allow_long_term_memory") is False or "不要使用历史" in text:
            return MemoryRecallRuleResult(status="skip_recall", reason="本轮请求或用户显式禁止读取长期记忆。")
        # 工具类意图（天气/计算）不需要个性化或业务画像。
        if intent in {"weather_query", "calculator_query"} or any(term in lowered for term in self.skip_terms):
            return MemoryRecallRuleResult(status="skip_recall", reason="一次性工具类或通用事实请求不需要长期记忆。")
        # 含运算符和数字的表达式，判定为计算请求。
        if any(symbol in text for symbol in ["+", "-", "*", "/"]) and any(char.isdigit() for char in text):
            return MemoryRecallRuleResult(status="skip_recall", reason="计算类请求跳过长期记忆。")

        # ── 第二优先级：收集 recall_layers ──
        # 多条规则可叠加，最终去重合并。
        layers: list[MemoryRecallLayer] = []
        # 用户显式引用历史：「按我之前说的」「继续上次那个客户」等 → 全层级召回。
        if any(term in text for term in self.explicit_history_terms):
            layers.extend(["preference", "customer_profile", "advisor_profile", "case_state", "memory_event"])
        # 多轮指代：「他/这个客户」等代词命中时，是否升级为长期业务事实召回。
        #
        # ① 触发锚点只用真实体锚点 last_entity / last_case_id，不再看 last_intent。
        #    原因：last_intent 每轮都会被 update_short_term_memory 写入，从第 2 轮起几乎恒为真，
        #    会让条件退化成「有代词 且 非首轮」，与"是否存在可被指代的实体"无关。
        # ④ 区分 session 内指代与跨会话指代：
        #    - 被指实体已在短期 recent_messages 覆盖 → 指代消解由 query_understanding 的短期逻辑完成，
        #      无需再跑一次长期 hybrid 召回（省延迟/成本，避免冗余）；
        #    - 锚点存在但已滑出短期窗口（长对话被截断 / 来自更早会话）→ 才升级为长期业务事实召回。
        pronoun_hit = any(term in text for term in self.pronoun_terms)
        entity_anchor = session_memory.get("last_entity") or session_memory.get("last_case_id")
        if pronoun_hit and entity_anchor and not self._entity_in_short_term(entity_anchor, session_memory):
            layers.extend(["customer_profile", "case_state", "memory_event"])
        # 个性化关键词：「风格」「偏好」「客户画像」→ 偏好 + 画像层。
        if any(term in text for term in self.personalization_terms):
            layers.extend(["preference", "advisor_profile", "customer_profile"])
        # 保险领域工作流必然依赖客户/从业者/case 事实。
        if domain_skill == "insurance_advisor" or workflow_name in self.business_terms:
            layers.extend(["customer_profile", "advisor_profile", "case_state", "memory_event", "domain_fact"])
        # 关键槽位缺失时，尝试从长期记忆中补全（如 customer_type、budget）。
        missing_slots = metadata.get("missing_slots") or []
        if any(slot in missing_slots for slot in self.missing_slot_terms):
            layers.extend(["customer_profile", "advisor_profile", "case_state"])

        # ── 第三优先级：must_recall 或 ambiguous ──
        if layers:
            # 生成多条 retrieval query：原始输入 + last_entity 增强 + case_id 增强。
            queries = [text]
            if session_memory.get("last_entity"):
                queries.append(f"{session_memory['last_entity']} {text}")
            if metadata.get("case_id") or metadata.get("opportunity_case_id"):
                queries.append(f"case {metadata.get('case_id') or metadata.get('opportunity_case_id')} {text}")
            return MemoryRecallRuleResult(
                status="must_recall",
                recall_layers=list(dict.fromkeys(layers)),  # 去重并保持插入顺序
                reason="规则命中历史偏好、多轮指代、个性化或保险业务事实召回信号。",
                queries=queries,
            )

        # 规则无法确定 → 生产链路交给模型，本地链路安全降级为不召回。
        return MemoryRecallRuleResult(
            status="ambiguous",
            reason="规则无法确定是否需要长期记忆，交给 memory_recall_decision 模型判断。",
            queries=[text],
        )

    @staticmethod
    def _entity_in_short_term(entity_anchor: Any, session_memory: dict[str, Any]) -> bool:
        """判断被指代实体是否已在短期 recent_messages 里出现过。

        命中说明当前 session 内已有该实体的上下文，指代消解交给短期记忆即可，
        无需触发长期 hybrid 召回；未命中（实体已滑出短期窗口，或来自更早会话）才需要升级长期召回。

        参数:
            entity_anchor: session 中的 last_entity / last_case_id 锚点值。
            session_memory: 短期 SESSION 记忆，含 recent_messages。

        返回:
            True 表示实体已在短期上下文覆盖；False 表示需要长期召回补齐。
        """
        # 锚点为空时无法判定覆盖，交由上游按"无锚点"处理。
        anchor = str(entity_anchor).strip()
        if not anchor:
            return False
        # 逐条扫描最近消息，只要有一条文本包含该实体，即认为短期上下文已覆盖。
        for message in session_memory.get("recent_messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if content and anchor in str(content):
                return True
        return False


def plan_long_term_memory_recall(
    *,
    input_text: str,
    workflow_name: str,
    intent: str | None,
    domain_skill: str | None,
    risk_level: str,
    session_memory: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryRecallDecision:
    """规则优先判断本轮是否需要召回长期记忆。

    短期 session/task 记忆是工作流恢复和指代消解的一部分，可以每轮读取；
    本函数只决定跨会话长期记忆，例如 preference、客户画像、从业者画像和 case 事件。

    该函数保留给现有调用方使用。生产链路应优先调用 decide_long_term_memory_recall，
    因为 ambiguous case 需要继续走真实模型判断。
    """
    session_memory = session_memory or {}
    metadata = metadata or {}

    # 调用规则引擎做毫秒级判断；risk_level 参数保留供未来扩展，当前规则层未直接使用。
    rule = MemoryRecallRuleEngine().decide(
        input_text=input_text,
        workflow_name=workflow_name,
        intent=intent,
        domain_skill=domain_skill,
        session_memory=session_memory,
        metadata=metadata,
    )

    # 规则明确跳过：不召回，status=rule_skip_recall。
    if rule.status == "skip_recall":
        return MemoryRecallDecision(
            should_recall=False,
            reason=rule.reason,
            status="rule_skip_recall",
            confidence=0.95,
        )

    # 规则不确定 + 本地无模型：安全降级，不召回长期记忆（生产链路应走 decide_long_term_memory_recall）。
    if rule.status == "ambiguous":
        return MemoryRecallDecision(
            should_recall=False,
            reason=rule.reason,
            status="model_unavailable_safe_skip",
            queries=[RetrievalQuery(text=query, purpose="memory") for query in rule.queries],
            confidence=0.3,
        )

    # 规则明确召回：组装完整 MemoryRecallDecision，供 hybrid_recall_memory 使用。
    return MemoryRecallDecision(
        should_recall=True,
        recall_layers=rule.recall_layers,
        reason=rule.reason,
        queries=[RetrievalQuery(text=query, purpose="memory") for query in rule.queries],
        top_k=6 if domain_skill == "insurance_advisor" else 4,  # 保险领域允许更多召回条数
        status="rule_must_recall",
        confidence=0.95,
        filters=metadata,
    )


def decide_long_term_memory_recall(
    *,
    input_text: str,
    workflow_name: str,
    intent: str | None,
    domain_skill: str | None,
    tenant_id: str,
    user_id: str | None,
    session_id: str,
    session_memory: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    memory_config: MemoryConfig | None = None,
    model_client: OpenAICompatibleChatClient | None = None,
) -> MemoryRecallDecision:
    """生产级长期记忆召回决策：规则 + 模型两阶段。

    与 ``plan_long_term_memory_recall`` 的区别：
    - 本函数在 ambiguous case 会调用 ``memory_recall_decision`` 模型；
    - 增加租户策略校验（memory_config.enabled、tenant_id/user_id 必填）；
    - 输出 filters 含 decision_latency_ms，便于生产 trace 审计。

    决策流程:
        1. 租户/用户校验 → 关闭或缺 ID 则 skip；
        2. 规则引擎 → skip / must_recall 直接返回；
        3. ambiguous → 调用模型 JSON 决策；
        4. 模型失败或未配置 → 安全降级 skip。

    参数:
        tenant_id / user_id / session_id: 租户与用户隔离标识，长期记忆读取必填。
        memory_config: 租户级记忆策略（enabled、model_decision_enabled、max_recall_items）。
        model_client: OpenAI 兼容 Chat 客户端，用于 ambiguous case 结构化 JSON 决策。

    返回:
        MemoryRecallDecision，status 可能是 rule_skip_recall / rule_must_recall /
        model_decision / model_unavailable_safe_skip / failed_schema_validation。
    """
    started_at = time.perf_counter()
    session_memory = session_memory or {}
    # 合并 tenant/user/session 到 metadata，供后续检索 filters 和模型输入使用。
    metadata = {
        **(metadata or {}),
        "tenant_id": tenant_id,
        "user_id": user_id,
        "session_id": session_id,
    }
    memory_config = memory_config or MemoryConfig()

    # ── 租户策略层：全局关闭长期记忆 ──
    if not memory_config.enabled:
        return MemoryRecallDecision(
            should_recall=False,
            reason="当前租户策略关闭长期记忆。",
            status="rule_skip_recall",
            filters=metadata,
            confidence=1.0,
            latency_budget_ms=memory_config.decision_timeout_ms,
        )
    # ── 安全策略：缺少 tenant_id 或 user_id 禁止读取跨会话记忆 ──
    if not tenant_id or not user_id:
        return MemoryRecallDecision(
            should_recall=False,
            reason="缺少 tenant_id 或 user_id，安全策略禁止读取长期记忆。",
            status="rule_skip_recall",
            filters=metadata,
            confidence=1.0,
            latency_budget_ms=memory_config.decision_timeout_ms,
        )

    # ── 规则引擎（与本地版相同）──
    rule = MemoryRecallRuleEngine().decide(
        input_text=input_text,
        workflow_name=workflow_name,
        intent=intent,
        domain_skill=domain_skill,
        session_memory=session_memory,
        metadata=metadata,
    )
    if rule.status == "skip_recall":
        return MemoryRecallDecision(
            should_recall=False,
            reason=rule.reason,
            status="rule_skip_recall",
            filters=metadata,
            confidence=0.95,
            latency_budget_ms=memory_config.decision_timeout_ms,
        )
    if rule.status == "must_recall":
        return MemoryRecallDecision(
            should_recall=True,
            recall_layers=rule.recall_layers,
            reason=rule.reason,
            queries=[RetrievalQuery(text=query, purpose="memory") for query in rule.queries],
            top_k=memory_config.max_recall_items,
            status="rule_must_recall",
            filters={**metadata, "max_risk_level": "medium"},
            confidence=0.95,
            latency_budget_ms=memory_config.decision_timeout_ms,
        )

    # ── ambiguous：模型未启用或未注入 client → 安全降级 ──
    if not memory_config.model_decision_enabled or model_client is None:
        return MemoryRecallDecision(
            should_recall=False,
            reason="规则无法确定且未配置 memory_recall_decision 模型，安全降级为不召回长期记忆。",
            status="model_unavailable_safe_skip",
            queries=[RetrievalQuery(text=query, purpose="memory") for query in rule.queries],
            filters=metadata,
            confidence=0.2,
            latency_budget_ms=memory_config.decision_timeout_ms,
        )

    # ── ambiguous：调用 memory_recall_decision 模型 ──
    try:
        output, _model_result = model_client.complete_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是长期记忆召回决策节点。只判断是否需要读取长期记忆，"
                        "不得回答用户问题。输出 JSON 字段：should_recall、recall_scopes、"
                        "reason、queries、filters、confidence、latency_budget_ms。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户输入：{input_text}\n"
                        f"workflow_name：{workflow_name}\n"
                        f"intent：{intent}\n"
                        f"domain_skill：{domain_skill}\n"
                        f"session_memory：{session_memory}\n"
                        f"metadata：{metadata}"
                    ),
                },
            ],
            schema_model=MemoryRecallDecisionModelOutput,
        )
    except Exception as exc:
        # 模型 JSON 不合法或调用失败 → 不召回敏感长期记忆，保留短期 session。
        return MemoryRecallDecision(
            should_recall=False,
            reason=f"memory_recall_decision 模型输出不可用，安全降级：{exc}",
            status="failed_schema_validation",
            queries=[RetrievalQuery(text=query, purpose="memory") for query in rule.queries],
            filters=metadata,
            confidence=0.0,
            latency_budget_ms=memory_config.decision_timeout_ms,
        )

    # 合并模型 filters 与系统强制 filters，并记录决策耗时。
    filters = {
        **output.filters,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "session_id": session_id,
        "max_risk_level": output.filters.get("max_risk_level", "medium"),
        "decision_latency_ms": int((time.perf_counter() - started_at) * 1000),
    }
    queries = output.queries or rule.queries
    return MemoryRecallDecision(
        should_recall=output.should_recall,
        recall_layers=output.recall_scopes,
        reason=output.reason,
        queries=[RetrievalQuery(text=query, purpose="memory") for query in queries],
        top_k=memory_config.max_recall_items,
        status="model_decision",
        filters=filters,
        confidence=output.confidence,
        latency_budget_ms=output.latency_budget_ms,
    )


class ProductionMemoryRetriever:
    """长期记忆生产召回器。

    召回链路必须使用真实 embedding、PostgreSQL pgvector 和 reranker。外部返回的长期记忆
    仍然只作为上下文证据，不能覆盖用户本轮输入或系统合规约束。
    """

    def __init__(
        self,
        *,
        repository: PostgresAgentRepository,
        embedding_client: OpenAICompatibleEmbeddingClient,
        reranker_client: RerankerClient,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        """注入生产召回所需的三件套：PostgreSQL 仓储、Embedding 客户端、Reranker 客户端。"""
        self.repository = repository
        self.embedding_client = embedding_client
        self.reranker_client = reranker_client
        self.retrieval_config = retrieval_config or RetrievalConfig()

    def recall(
        self,
        *,
        decision: MemoryRecallDecision,
        tenant_id: str,
        user_id: str,
    ) -> MemoryRecallResult:
        """按决策对象从 PostgreSQL pgvector 召回长期记忆并压缩摘要。

        本方法是 ``hybrid_recall_memory`` 的生产版对应物，链路为：
        embedding → pgvector hybrid search → 多 query 去重合并 → reranker 精排 → TopK + 阈值过滤。

        与本地版的差异:
        - 使用真实 embedding 和 pgvector，而非 token_jaccard 近似；
        - 使用独立 reranker 模型，而非规则 bonus；
        - 数据来源是 PostgreSQL 持久化记忆，而非内存 dict 转 documents。

        参数:
            decision: 上游 ``decide_long_term_memory_recall`` 产出的召回决策。
            tenant_id: 租户 ID，传给 repository 做隔离检索。
            user_id: 用户 ID，长期记忆按用户维度存储和检索。

        返回:
            MemoryRecallResult，结构与 ``hybrid_recall_memory`` 一致。

        异常:
            ValueError: decision.should_recall=true 但 queries 为空。
        """
        # 决策层已判定不召回，直接返回空结果。
        if not decision.should_recall:
            return MemoryRecallResult(decision=decision, items=[], compact_summary={})
        if not decision.queries:
            raise ValueError("长期记忆召回需要至少一个 query")

        # ── Step 1: 批量 embedding + pgvector hybrid search ──
        query_texts = [query.text for query in decision.queries]
        embeddings = self.embedding_client.embed(query_texts)
        # merged 按 hit.id 去重，同一记忆被多条 query 命中时保留最高分。
        merged: dict[str, PersistedMemoryHit] = {}
        for query, embedding in zip(decision.queries, embeddings, strict=True):
            hits = self.repository.search_long_term_memory(
                tenant_id=tenant_id,
                user_id=user_id,
                query=query.text,
                query_embedding=embedding,
                scopes=decision.recall_layers,
                case_id=decision.filters.get("case_id") or decision.filters.get("opportunity_case_id"),
                max_risk_level=decision.filters.get("max_risk_level", "medium"),
                top_k=max(decision.top_k * 2, decision.top_k),  # 粗排取 2 倍候选供 reranker
                score_threshold=decision.score_threshold,
            )
            for hit in hits:
                previous = merged.get(hit.id)
                if previous is None or hit.final_score > previous.final_score:
                    merged[hit.id] = hit

        # ── Step 2: Reranker 精排 ──
        candidates = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)
        if candidates:
            ranking = self.reranker_client.rerank(
                query=" ".join(query_texts),
                documents=[hit.content for hit in candidates],
                top_k=decision.top_k,
            )
            ranked = [candidates[item.index] for item in ranking if item.index < len(candidates)]
        else:
            ranked = candidates

        # ── Step 3: 转换 + 阈值过滤 + 压缩摘要 ──
        items = [
            _memory_hit_to_recall_item(hit)
            for hit in ranked[: decision.top_k]
            if hit.final_score >= decision.score_threshold
        ]
        return MemoryRecallResult(
            decision=decision,
            items=items,
            compact_summary=_compact_memory_items(items),
        )


def preference_memory_to_documents(
    *,
    tenant_id: str,
    subject_id: str,
    preference_memory: dict[str, Any],
) -> list[RetrievalDocument]:
    """把 PREFERENCE 层原始 dict 转成 HybridRetriever 可检索的 RetrievalDocument 列表。

    MemoryManager.read(PREFERENCE) 返回的是键值对 dict，无法直接做 hybrid search。
    本函数将每个偏好字段拆成独立文档，补齐 library/source_id/chunk_id/tags/layer metadata，
    供 ``hybrid_recall_memory`` 按相关性召回 TopK，而非整包注入上下文。

    参数:
        tenant_id: 租户 ID，写入 DocumentMetadata.tenant_id。
        subject_id: 偏好主体 ID，通常为 user_id，匿名时退回 session_id。
        preference_memory: MemoryManager 读出的 PREFERENCE dict。

    返回:
        RetrievalDocument 列表；``memory_candidates`` 列表中每项单独成文档。

    典型输入::
        {"preferred_style": "喜欢结构化中文",
         "memory_candidates": [{"type": "preferred_style", "value": "..."}]}
    """
    documents: list[RetrievalDocument] = []
    for key, value in preference_memory.items():
        # memory_candidates 是 long_term_memory_candidate 节点写入的候选列表，需逐条展开。
        if key == "memory_candidates" and isinstance(value, list):
            for index, candidate in enumerate(value):
                candidate_type = str(candidate.get("type", "memory_candidate")) if isinstance(candidate, dict) else "memory_candidate"
                candidate_value = candidate.get("value") if isinstance(candidate, dict) else candidate
                documents.append(
                    _memory_document(
                        tenant_id=tenant_id,
                        library="preference_memory",
                        source_id=f"preference:{subject_id}",
                        chunk_id=f"memory_candidate:{index}",
                        text=f"preference {candidate_type}: {candidate_value}",
                        tags=["preference", candidate_type],
                        extra={"layer": "preference", "key": candidate_type, "value": candidate_value},
                    )
                )
        else:
            # 普通键值对：每个 key 对应一条独立检索文档。
            documents.append(
                _memory_document(
                    tenant_id=tenant_id,
                    library="preference_memory",
                    source_id=f"preference:{subject_id}",
                    chunk_id=str(key),
                    text=f"preference {key}: {value}",
                    tags=["preference", str(key)],
                    extra={"layer": "preference", "key": key, "value": value},
                )
            )
    return documents


def business_memory_to_documents(
    *,
    tenant_id: str,
    customer_facts: list[CustomerProfileFact] | None = None,
    advisor_facts: list[AdvisorProfileFact] | None = None,
    opportunity_case: OpportunityCase | None = None,
    events: list[MemoryEvent] | None = None,
) -> list[RetrievalDocument]:
    """把业务长期事实（客户画像/从业者/case/事件）转成可检索文档。

    供 ``load_business_memory`` 调用，与 ``preference_memory_to_documents`` 对称。
    转换规则:
    - 客户事实：跳过非 current 和 PII 敏感级；
    - 从业者事实：跳过非 current；
    - case：整条 case 状态合成一条文档；
    - 事件：每条 MemoryEvent 单独成文档。

    参数:
        tenant_id: 租户 ID。
        customer_facts: 客户 KYC 事实列表。
        advisor_facts: 保险从业者画像事实列表。
        opportunity_case: 当前 active 机会 case。
        events: 与 case/会话关联的事件记忆。

    返回:
        混合多 layer 的 RetrievalDocument 列表，供 ``hybrid_recall_memory`` 统一检索。
    """
    documents: list[RetrievalDocument] = []

    # ── 客户画像事实 ──
    for fact in customer_facts or []:
        # 过期事实和 PII 不进入检索候选集，避免隐私泄露和过时信息污染。
        if not fact.is_current or fact.sensitivity_level == "pii":
            continue
        value = fact.normalized_value if fact.normalized_value is not None else fact.fact_value
        documents.append(
            _memory_document(
                tenant_id=tenant_id,
                library="customer_profile_memory",
                source_id=f"customer_fact:{fact.customer_id}",
                chunk_id=fact.id,
                text=f"customer_profile {fact.fact_key}: {value} evidence: {fact.evidence_text}",
                tags=["customer_profile", fact.fact_key, fact.certainty],
                extra={
                    "layer": "customer_profile",
                    "fact_id": fact.id,
                    "fact_key": fact.fact_key,
                    "fact_value": value,
                    "certainty": fact.certainty,
                    "is_current": fact.is_current,
                    "sensitivity_level": fact.sensitivity_level,
                },
            )
        )

    # ── 从业者画像事实 ──
    for fact in advisor_facts or []:
        if not fact.is_current:
            continue
        documents.append(
            _memory_document(
                tenant_id=tenant_id,
                library="advisor_profile_memory",
                source_id=f"advisor_fact:{fact.advisor_id}",
                chunk_id=fact.id,
                text=f"advisor_profile {fact.fact_key}: {fact.fact_value} evidence: {fact.evidence_text}",
                tags=["advisor_profile", fact.fact_key],
                extra={
                    "layer": "advisor_profile",
                    "fact_id": fact.id,
                    "fact_key": fact.fact_key,
                    "fact_value": fact.fact_value,
                    "is_current": fact.is_current,
                },
            )
        )

    # ── 机会 case 状态（单条合成文档）──
    if opportunity_case is not None:
        documents.append(
            _memory_document(
                tenant_id=tenant_id,
                library="case_memory",
                source_id=f"case:{opportunity_case.id}",
                chunk_id="case_state",
                text=(
                    "case_state "
                    f"target_persona: {opportunity_case.target_persona} "
                    f"trigger_module: {opportunity_case.trigger_module} "
                    f"current_stage: {opportunity_case.current_stage} "
                    f"next_best_action: {opportunity_case.next_best_action}"
                ),
                tags=["case_state", opportunity_case.target_persona, opportunity_case.trigger_module],
                extra={
                    "layer": "case_state",
                    "case_id": opportunity_case.id,
                    "target_persona": opportunity_case.target_persona,
                    "trigger_module": opportunity_case.trigger_module,
                    "current_stage": opportunity_case.current_stage,
                },
            )
        )

    # ── 事件记忆 ──
    for event in events or []:
        documents.append(
            _memory_document(
                tenant_id=tenant_id,
                library="event_memory",
                source_id=f"memory_event:{event.opportunity_case_id or event.conversation_id or event.id}",
                chunk_id=event.id,
                text=f"memory_event {event.event_type}: {event.event_payload} evidence: {event.evidence_text}",
                tags=["memory_event", event.event_type],
                extra={"layer": "memory_event", "event_type": event.event_type, "event_id": event.id},
            )
        )
    return documents


def hybrid_recall_memory(
    *,
    decision: MemoryRecallDecision,
    documents: list[RetrievalDocument],
    tenant_id: str,
) -> MemoryRecallResult:
    """执行长期记忆 hybrid search + rerank，并返回压缩摘要。

    本函数是本地/内存版长期记忆召回的执行入口，对应生产链路中的
    ``ProductionMemoryRetriever.recall``。调用方需先把原始记忆（preference dict、
    业务事实等）通过 ``*_memory_to_documents`` 转成 ``RetrievalDocument`` 列表，
    再传入本函数完成「检索 → 重排 → 截断 → 压缩」四步流水线。

    参数:
        decision: 上游 ``plan_long_term_memory_recall`` / ``decide_long_term_memory_recall``
            产出的召回决策，包含 should_recall、recall_layers、queries、top_k、
            score_threshold 等控制字段。
        documents: 由 ``preference_memory_to_documents`` 或 ``business_memory_to_documents``
            转换后的可检索文档集合；每条文档携带 text 与 metadata（layer/library/tags）。
        tenant_id: 当前租户 ID，用于 metadata 过滤，防止跨租户记忆泄漏。

    返回:
        MemoryRecallResult，包含:
        - items: 经过 hybrid search + rerank + 阈值过滤后的 TopK 记忆条目；
        - compact_summary: 按 layer 压缩后的摘要 dict，可直接写入 memory_context。

    典型调用链::
        restore_memory / load_business_memory
            → *_memory_to_documents(documents)
            → hybrid_recall_memory(decision, documents, tenant_id)
            → memory_recall_results + memory_context.preference / profile_state
    """
    # ── 前置短路：决策层已判定不需要召回，或上游没有可检索文档 ──
    # 常见场景：规则 skip_recall、ambiguous 安全降级、preference dict 为空。
    if not decision.should_recall or not documents:
        return MemoryRecallResult(decision=decision, items=[], compact_summary={})

    # ── Step 1: 构建 metadata 过滤条件 ──
    # recall_layers（如 preference / customer_profile）映射到 HybridRetriever 使用的 library 名
    # （如 preference_memory / customer_profile_memory），确保只检索决策允许的层级。
    libraries = [LAYER_TO_LIBRARY[layer] for layer in decision.recall_layers]

    # ── Step 2: Hybrid Search（词法 + 向量近似 + metadata 融合打分）──
    # decision.queries 可能包含多条 rewrite query（原始输入 + last_entity + case_id 等），
    # HybridRetriever 会对每条 query 遍历 documents，按 source_id+chunk_id 去重并保留最高分。
    # top_k 取 decision.top_k 的 2 倍，为后续 rerank 和 score_threshold 过滤留出候选余量。
    results = HybridRetriever(documents).search(
        decision.queries,
        filters=MetadataFilter(
            tenant_id=tenant_id,          # 租户隔离：只检索当前租户的记忆文档
            libraries=libraries,          # 层级隔离：只检索 decision.recall_layers 对应的 library
            max_risk_level="medium",      # 风险过滤：高风险记忆不进入生成上下文
            approved_only=True,           # 审批过滤：未 approved 的记忆不参与召回
        ),
        top_k=max(decision.top_k * 2, decision.top_k),
    )

    # ── Step 3: 二阶段 Memory Rerank ──
    # 在 hybrid 融合分基础上叠加记忆特有的 bonus：
    # is_current（当前有效事实）、certainty=confirmed（已确认）、case_memory 层级、
    # 以及中文字符重叠相关性；最终产出 MemoryRecallItem 并按 rerank_score 降序排列。
    items = _rerank_memory_results(results, decision, tenant_id)

    # ── Step 4: 阈值过滤 + TopK 截断 ──
    # score_threshold 过滤掉与当前 query 相关性过低的记忆，避免噪声进入 prompt；
    # 最后只保留 decision.top_k 条，控制 token 消耗。
    items = [item for item in items if item.rerank_score >= decision.score_threshold][: decision.top_k]

    # ── Step 5: 按层级压缩为 compact_summary ──
    # 将 MemoryRecallItem 列表转为 {preference: {...}, customer_profile: {...}, ...} 结构，
    # 供 restore_memory 写入 memory_context，或 load_business_memory 合并到 profile_state。
    return MemoryRecallResult(
        decision=decision,
        items=items,
        compact_summary=_compact_memory_items(items),
    )


def _rerank_memory_results(
    results: list[RetrievalResult],
    decision: MemoryRecallDecision,
    tenant_id: str,
) -> list[MemoryRecallItem]:
    """二阶段 Memory Rerank：在 Hybrid Search 融合分上叠加记忆业务 bonus。

    HybridRetriever 的 combined score 主要反映文本相关性；记忆场景还需要考虑：
    - **is_current**: 当前有效事实优先于历史过期事实（+0.05）；
    - **certainty=confirmed**: 已确认事实优先于 uncertain（+0.05）；
    - **case_memory**: case 状态在保险场景有更高业务优先级（+0.03）；
    - **中文字符重叠**: 弥补本地无真实 embedding 时的语义匹配不足（最高 +0.18）。

    参数:
        results: HybridRetriever.search 返回的粗排结果。
        decision: 召回决策，用于校验 recall_layers 和拼接 query 文本。
        tenant_id: 租户 ID，二次校验防止跨租户泄漏。

    返回:
        按 rerank_score 降序排列的 MemoryRecallItem 列表。
    """
    query_text = " ".join(query.text for query in decision.queries)
    items: list[MemoryRecallItem] = []
    for result in results:
        metadata = result.document.metadata
        # 租户隔离二次校验（HybridRetriever 已过滤，此处防御性检查）。
        if metadata.tenant_id != tenant_id:
            continue
        extra = dict(metadata.extra)
        layer = extra.get("layer")
        # 只保留 decision.recall_layers 允许的层级。
        if layer not in decision.recall_layers:
            continue

        # ── 记忆业务 bonus 累加 ──
        bonus = 0.0
        if extra.get("is_current", True):
            bonus += 0.05
        if extra.get("certainty") == "confirmed":
            bonus += 0.05
        if metadata.library == "case_memory":
            bonus += 0.03
        bonus += _character_overlap_bonus(query_text, result.document.text)

        final_score = min(1.0, result.score + bonus)
        items.append(
            MemoryRecallItem(
                layer=layer,
                source_id=metadata.source_id,
                chunk_id=metadata.chunk_id,
                content=_truncate_memory_content(result.document.text),
                metadata=extra,
                lexical_score=result.lexical_score,
                vector_score=result.vector_score,
                metadata_score=result.metadata_score,
                rerank_score=final_score,
            )
        )
    return sorted(items, key=lambda item: item.rerank_score, reverse=True)


def _memory_hit_to_recall_item(hit: PersistedMemoryHit) -> MemoryRecallItem:
    """把 PostgreSQL pgvector 检索命中转换为统一的 MemoryRecallItem。

    生产链路 ``ProductionMemoryRetriever.recall`` 使用本函数做格式对齐，
    使生产版和本地版输出相同结构的 MemoryRecallItem，下游 ``_compact_memory_items``
    和 memory_context 写入逻辑无需区分数据来源。

    参数:
        hit: PostgresAgentRepository.search_long_term_memory 返回的单条命中。

    返回:
        MemoryRecallItem，含 lexical/vector/metadata/rerank 四类分数。
    """
    # scope 不在已知 layer 映射中时，降级为 domain_fact。
    layer = hit.scope if hit.scope in LAYER_TO_LIBRARY else "domain_fact"
    metadata = dict(hit.metadata)
    metadata.setdefault("memory_type", hit.memory_type)
    metadata.setdefault("source", "postgres_pgvector")
    return MemoryRecallItem(
        layer=layer,  # type: ignore[arg-type]
        source_id=hit.id,
        chunk_id=hit.id,
        content=_truncate_memory_content(hit.content),
        metadata=metadata,
        lexical_score=hit.lexical_score,
        vector_score=hit.vector_score,
        metadata_score=hit.metadata_score,
        rerank_score=hit.final_score,
    )


def _compact_memory_items(items: list[MemoryRecallItem]) -> dict[str, Any]:
    """把 TopK MemoryRecallItem 按 layer 压缩为 memory_context 可直接使用的 dict。

    压缩规则:
    - preference → {key: value} 扁平 dict；
    - customer_profile → {confirmed: {...}, uncertain: {...}} 按 certainty 分桶；
    - advisor_profile → {fact_key: fact_value}；
    - case_state → metadata 直接 merge；
    - memory_event → 列表 append。

    参数:
        items: hybrid_recall_memory 或 ProductionMemoryRetriever 产出的最终 TopK 条目。

    返回:
        compact_summary dict，写入 memory_context 或 profile_state。
    """
    summary: dict[str, Any] = {
        "preference": {},
        "customer_profile": {"confirmed": {}, "uncertain": {}},
        "advisor_profile": {},
        "case_state": {},
        "memory_events": [],
    }
    for item in items:
        if item.layer == "preference":
            key = str(item.metadata.get("key", item.chunk_id))
            summary["preference"][key] = item.metadata.get("value", item.content)
        elif item.layer == "customer_profile":
            certainty = item.metadata.get("certainty", "confirmed")
            bucket = "uncertain" if certainty == "uncertain" else "confirmed"
            key = str(item.metadata.get("fact_key", item.chunk_id))
            summary["customer_profile"][bucket][key] = item.metadata.get("fact_value", item.content)
        elif item.layer == "advisor_profile":
            key = str(item.metadata.get("fact_key", item.chunk_id))
            summary["advisor_profile"][key] = item.metadata.get("fact_value", item.content)
        elif item.layer == "case_state":
            summary["case_state"].update(item.metadata)
        elif item.layer == "memory_event":
            summary["memory_events"].append(item.metadata)
    return summary


def _memory_document(
    *,
    tenant_id: str,
    library: str,
    source_id: str,
    chunk_id: str,
    text: str,
    tags: list[str],
    extra: dict[str, Any],
) -> RetrievalDocument:
    """创建长期记忆检索文档，统一补齐 DocumentMetadata 默认值。

    所有 ``*_memory_to_documents`` 函数都通过本 helper 创建文档，确保：
    - risk_level 默认 low；
    - approved_for_generation 默认 True；
    - extra 中必须含 layer 字段，供 rerank 和 compact 使用。

    参数:
        tenant_id: 租户 ID。
        library: 知识库名，如 preference_memory / customer_profile_memory。
        source_id: 记忆来源 ID，用于去重键 (source_id, chunk_id)。
        chunk_id: 记忆片段 ID，同一 source 下唯一。
        text: 可检索文本，格式如 ``preference key: value``。
        tags: 标签列表，供 MetadataFilter.required_tags 过滤。
        extra: 业务 metadata，必须含 layer/key/value 等字段。

    返回:
        完整的 RetrievalDocument 对象。
    """
    return RetrievalDocument(
        text=text,
        metadata=DocumentMetadata(
            source_id=source_id,
            chunk_id=chunk_id,
            library=library,
            tenant_id=tenant_id,
            tags=tags,
            risk_level="low",
            approved_for_generation=True,
            extra=extra,
        ),
    )


def _build_memory_queries(
    text: str,
    session_memory: dict[str, Any],
    domain_skill: str | None,
) -> list[RetrievalQuery]:
    """为长期记忆召回生成多条 weighted RetrievalQuery（历史遗留 helper）。

    当前主链路由 ``MemoryRecallRuleEngine.decide`` 和模型决策直接生成 queries，
    本函数保留供测试或旧调用方使用。生成的 query 包括：
    - 原始输入（weight=1.0）；
    - 拼接 last_intent（weight=0.7）；
    - 拼接 last_entity（weight=0.8）；
    - 拼接 domain_skill（weight=0.6，purpose=profile）。

    参数:
        text: 用户本轮输入。
        session_memory: SESSION 记忆，含 last_intent / last_entity。
        domain_skill: 领域 Skill 名。

    返回:
        带 weight 的多条 RetrievalQuery。
    """
    queries = [RetrievalQuery(text=text, purpose="memory", weight=1.0)]
    last_intent = session_memory.get("last_intent")
    if last_intent:
        queries.append(RetrievalQuery(text=f"{text} {last_intent}", purpose="memory", weight=0.7))
    last_entity = session_memory.get("last_entity")
    if last_entity:
        queries.append(RetrievalQuery(text=f"{text} {last_entity}", purpose="memory", weight=0.8))
    if domain_skill:
        queries.append(RetrievalQuery(text=f"{domain_skill} {text}", purpose="profile", weight=0.6))
    return queries


def _is_low_value_for_long_term_memory(text: str, intent: str | None) -> bool:
    """识别不值得召回长期记忆的低价值请求（历史遗留 helper）。

    判断条件：工具类 intent、算术表达式、天气/计算关键词、简单问候语。
    当前主链路已迁移到 ``MemoryRecallRuleEngine``，本函数保留供兼容调用。

    参数:
        text: 用户输入。
        intent: 意图分类结果。

    返回:
        True 表示不应召回长期记忆。
    """
    if intent in {"weather_query", "calculator_query"}:
        return True
    lowered = text.lower()
    if any(symbol in text for symbol in ["+", "-", "*", "/"]) and any(char.isdigit() for char in text):
        return True
    if any(keyword in lowered for keyword in ["天气", "weather", "计算", "calculator"]):
        return True
    if text in {"你好", "hello", "hi", "在吗"}:
        return True
    return False


def _needs_preference_recall(text: str, domain_skill: str | None) -> bool:
    """判断是否需要召回用户长期偏好（历史遗留 helper）。

    命中偏好关键词（「我喜欢」「按我的」「风格」等）或存在 domain_skill 时返回 True。
    当前主链路由 ``MemoryRecallRuleEngine.personalization_terms`` 覆盖。

    参数:
        text: 用户输入。
        domain_skill: 领域 Skill。

    返回:
        True 表示可能需要召回 preference 层。
    """
    preference_terms = ["我喜欢", "我的偏好", "按我的", "上次我说", "记得我", "风格", "继续用"]
    return any(term in text for term in preference_terms) or domain_skill is not None


def _needs_business_recall(
    text: str,
    workflow_name: str,
    intent: str | None,
    domain_skill: str | None,
    metadata: dict[str, Any],
) -> bool:
    """判断是否需要召回客户/从业者/case 业务长期记忆（历史遗留 helper）。

    触发条件：KYC 工作流、保险顾问 intent/skill、metadata 含 customer_id/case_id、
    或输入含保险业务关键词。当前主链路由 ``MemoryRecallRuleEngine.business_terms`` 覆盖。

    参数:
        text: 用户输入。
        workflow_name: 工作流名。
        intent: 意图。
        domain_skill: 领域 Skill。
        metadata: 请求 metadata。

    返回:
        True 表示可能需要召回 customer_profile / case_state 等业务层。
    """
    if workflow_name == "insurance_kyc_coach_workflow":
        return True
    if domain_skill == "insurance_advisor" or intent == "insurance_advisor_help":
        return True
    if metadata.get("customer_id") or metadata.get("opportunity_case_id"):
        return True
    business_terms = ["客户", "这个人", "这个客户", "上次", "之前", "继续", "保险", "kyc", "破冰", "异议", "策略"]
    return any(term in text for term in business_terms)


def _character_overlap_bonus(query_text: str, document_text: str) -> float:
    """计算 query 与 document 的中文字符重叠 bonus，弥补本地无真实 embedding 的限制。

    本地 ``HybridRetriever`` 用 token_jaccard 近似向量分，中文语义匹配较弱。
    本函数提取 query 和 document 中的汉字集合，按重叠比例给最多 +0.18 的 bonus。

    参数:
        query_text: 拼接后的 query 文本。
        document_text: 记忆文档 text 字段。

    返回:
        0.0 ~ 0.18 之间的 bonus 分数。
    """
    query_chars = {char for char in query_text if "\u4e00" <= char <= "\u9fff"}
    document_chars = {char for char in document_text if "\u4e00" <= char <= "\u9fff"}
    if not query_chars or not document_chars:
        return 0.0
    overlap = len(query_chars & document_chars) / max(len(query_chars), 1)
    return min(0.18, overlap * 0.18)


def _truncate_memory_content(text: str, limit: int = 220) -> str:
    """限制单条记忆进入上下文的长度，防止长 evidence 撑爆 token budget。

    参数:
        text: 记忆原始文本（可能含 evidence_text）。
        limit: 最大字符数，默认 220。

    返回:
        截断后的文本，超长时末尾追加 ``...``。
    """
    return text if len(text) <= limit else text[:limit] + "..."
