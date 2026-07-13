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


# MemoryRecallLayer 是允许被模型/规则选择的长期记忆白名单，未知层不得直接检索。
MemoryRecallLayer = Literal[
    "preference",
    "customer_profile",
    "advisor_profile",
    "case_state",
    "memory_event",
    "domain_fact",
]


# RuleDecisionStatus 表示规则层能够直接决定、直接跳过或必须交给模型的三态结果。
RuleDecisionStatus = Literal[
    "must_recall",
    "skip_recall",
    "ambiguous",
]

# MemoryRecallDecisionStatus 记录最终决策来源及失败降级状态，供 Trace 区分规则与模型路径。
MemoryRecallDecisionStatus = Literal[
    "rule_must_recall",
    "rule_skip_recall",
    "model_decision",
    "failed_schema_validation",
    "model_unavailable_safe_skip",
]


class MemoryRecallDecision(BaseModel):
    """长期记忆召回决策。"""

    # should_recall 是检索执行器的总开关，False 时不得访问长期记忆存储。
    should_recall: bool = Field(..., description="本轮是否需要召回长期记忆。")
    # recall_layers 是最小权限范围，检索器会映射为对应 library 过滤条件。
    recall_layers: list[MemoryRecallLayer] = Field(
        default_factory=list,
        description="本轮允许召回的长期记忆层，例如 preference、customer_profile、advisor_profile。",
    )
    # reason 用于审计，不进入用户回答的事实依据。
    reason: str = Field(default="", description="触发或跳过长期记忆召回的原因。")
    # queries 保存一到多条 query rewrite，并保留各自 purpose/weight。
    queries: list[RetrievalQuery] = Field(default_factory=list, description="用于长期记忆检索的 query rewrite 列表。")
    # top_k 和阈值共同限制进入上下文的条数与最低相关度。
    top_k: int = Field(default=5, ge=1, le=20, description="长期记忆召回最多返回条数。")
    score_threshold: float = Field(default=0.08, ge=0, le=1, description="低于该最终分数的记忆不进入上下文。")
    # status 精确标记规则、模型或失败降级路径。
    status: MemoryRecallDecisionStatus = Field(
        default="rule_skip_recall",
        description="召回决策来源或失败状态，用于审计规则和模型各自的命中情况。",
    )
    # filters 承载系统强制的 tenant/user/session/Case 边界；模型值不能覆盖强制身份。
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="长期记忆检索 filters，例如 tenant_id、user_id、session_id、case_id、max_risk_level。",
    )
    # confidence 表示“是否需要召回”的把握度，不是检索内容相关度。
    confidence: float = Field(default=0.0, ge=0, le=1, description="召回决策置信度。规则强命中通常接近 1。")
    # latency_budget_ms 供调用链做超时控制和可观测性记录。
    latency_budget_ms: int = Field(default=1200, description="召回决策允许消耗的延迟预算。")


class MemoryRecallItem(BaseModel):
    """一条经过 hybrid search 和 rerank 的长期记忆结果。"""

    # layer 决定 compact_summary 写入哪个隔离分区。
    layer: MemoryRecallLayer = Field(..., description="该记忆结果所属层级。")
    # source_id/chunk_id 是检索去重和审计回放的稳定复合标识。
    source_id: str = Field(..., description="记忆来源 ID，用于审计和回放。")
    chunk_id: str = Field(..., description="记忆片段 ID，用于定位具体事实或偏好项。")
    # content 已被截断且应为脱敏摘要，不能使用原始长对话替代。
    content: str = Field(..., description="进入上下文前的短摘要，不应包含 PII 或原始长对话。")
    # metadata 保存事实键、certainty 等结构化信息，用于安全压缩而非自由文本解析。
    metadata: dict[str, Any] = Field(default_factory=dict, description="记忆结果的业务 metadata，例如 fact_key、certainty。")
    # 四类分数分别保留粗排各分量和二阶段最终分，便于解释召回排序。
    lexical_score: float = Field(default=0.0, description="关键词检索得分。")
    vector_score: float = Field(default=0.0, description="向量近似检索得分。")
    metadata_score: float = Field(default=0.0, description="metadata 匹配得分。")
    rerank_score: float = Field(default=0.0, description="二阶段 rerank 后的最终得分。")


class MemoryRecallResult(BaseModel):
    """长期记忆召回结果。"""

    # decision 原样保留，确保每组命中都能追溯到召回原因和权限范围。
    decision: MemoryRecallDecision = Field(..., description="本次召回使用的决策对象。")
    # items 是排序后的原子候选，compact_summary 是供生成节点直接消费的分层摘要。
    items: list[MemoryRecallItem] = Field(default_factory=list, description="最终进入上下文候选的 TopK 记忆。")
    compact_summary: dict[str, Any] = Field(default_factory=dict, description="按层级压缩后的记忆摘要。")


class MemoryRecallRuleResult(BaseModel):
    """规则引擎对长期记忆是否召回的初步判断。"""

    # status 决定下一步是短路返回还是调用 memory_recall_decision 模型。
    status: RuleDecisionStatus = Field(..., description="规则判断结果：必须召回、跳过或交给模型判断。")
    # 其余字段保存规则选出的最小层级、审计理由和检索文本。
    recall_layers: list[MemoryRecallLayer] = Field(default_factory=list, description="规则命中的召回层级。")
    reason: str = Field(default="", description="规则命中原因，用于 trace。")
    queries: list[str] = Field(default_factory=list, description="规则生成的检索 query。")


class MemoryRecallDecisionModelOutput(BaseModel):
    """memory_recall_decision 模型必须返回的结构化 JSON。"""

    # 模型只能从受限 Schema 表达决策，不能直接执行检索或返回长期记忆内容。
    should_recall: bool = Field(..., description="模型判断是否召回长期记忆。")
    # recall_scopes 使用 MemoryRecallLayer 白名单，非法层会由 Pydantic 拒绝。
    recall_scopes: list[MemoryRecallLayer] = Field(default_factory=list, description="允许召回的长期记忆 scope。")
    reason: str = Field(..., description="模型决策原因。")
    queries: list[str] = Field(default_factory=list, description="模型生成的长期记忆检索 query。")
    # 模型 filters 仅是建议，系统会在返回前覆盖 tenant/user/session 强制边界。
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

    # 显式历史引用词一旦命中，需要同时召回偏好、画像、Case 和事件层。
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
    # 代词只有同时存在已滑出短期窗口的实体锚点时才升级为长期召回。
    pronoun_terms = ["他", "她", "它", "这个客户", "刚才那个", "上次那个", "之前那个产品"]
    # 工具类关键词用于毫秒级跳过长期记忆。
    skip_terms = ["天气", "weather", "计算", "calculator"]
    # 个性化词触发 Preference 与客户/顾问画像层。
    personalization_terms = ["风格", "偏好", "格式", "保险偏好", "销售偏好", "从业画像", "客户画像"]
    # 保险相关意图/Skill 构成业务召回白名单。
    business_terms = [
        "insurance_advisor",
        "insurance_break_ice",
        "insurance_objection_handling",
        "insurance_strategy",
        "insurance_kyc_collection",
        "sales_intelligence",
    ]
    # 只有这些可从历史事实安全补齐的缺失字段会触发额外业务召回。
    missing_field_terms = [
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
        2. **must_recall** — 命中历史引用、多轮指代、个性化关键词、保险业务、KYC 缺失字段；
        3. **ambiguous** — 以上均不满足，交给 ``decide_long_term_memory_recall`` 的模型层。

        参数:
            input_text: 用户本轮原始输入。
            workflow_name: 兼容请求标签；保险业务召回以 intent/domain_skill/active intent 为准。
            intent: 意图分类结果（可能为 None，restore_memory 阶段尚未 classify）。
            domain_skill: 领域 Skill，如 ``insurance_advisor``。
            session_memory: 短期 SESSION 记忆，含 last_entity / last_intent / last_case_id。
            metadata: 请求 metadata，含 KYC missing_fields、case_id、allow_long_term_memory 等。

        返回:
            MemoryRecallRuleResult，status 三选一：must_recall / skip_recall / ambiguous。
        """
        # 保留原始大小写文本用于中文/精确匹配，同时构造小写副本匹配英文词。
        text = input_text.strip()
        # 小写副本只用于不区分大小写的英文工具词匹配。
        lowered = text.lower()

        # ── 第一优先级：明确跳过 ──
        # 用户显式禁止，或 metadata 关闭长期记忆开关。
        if metadata.get("allow_long_term_memory") is False or "不要使用历史" in text:
            # 返回明确跳过结果，确保后续模型层不会重新开启长期读取。
            return MemoryRecallRuleResult(status="skip_recall", reason="本轮请求或用户显式禁止读取长期记忆。")
        # 工具类意图（天气/计算）不需要个性化或业务画像。
        if intent in {"weather_query", "calculator_query"} or any(term in lowered for term in self.skip_terms):
            # 一次性工具请求直接返回跳过，避免画像对客观结果产生无关影响。
            return MemoryRecallRuleResult(status="skip_recall", reason="一次性工具类或通用事实请求不需要长期记忆。")
        # 含运算符和数字的表达式，判定为计算请求。
        if any(symbol in text for symbol in ["+", "-", "*", "/"]) and any(char.isdigit() for char in text):
            # 算术表达式只需要计算工具，不召回偏好或业务事实。
            return MemoryRecallRuleResult(status="skip_recall", reason="计算类请求跳过长期记忆。")

        # ── 第二优先级：收集 recall_layers ──
        # 多条规则可叠加，最终去重合并。
        layers: list[MemoryRecallLayer] = []
        # 用户显式引用历史：「按我之前说的」「继续上次那个客户」等 → 全层级召回。
        if any(term in text for term in self.explicit_history_terms):
            # 显式历史请求允许召回五个用户相关长期层。
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
        # 分别计算代词命中和实体锚点，避免把普通“他”无条件解释为跨会话引用。
        pronoun_hit = any(term in text for term in self.pronoun_terms)
        # 优先使用最近实体，缺失时回退 Case ID 作为指代锚点。
        entity_anchor = session_memory.get("last_entity") or session_memory.get("last_case_id")
        # 仅在有锚点且锚点不在短期消息窗口时，追加需要跨会话读取的三个业务层。
        if pronoun_hit and entity_anchor and not self._entity_in_short_term(entity_anchor, session_memory):
            # 跨窗口指代需要客户画像、Case 和事件共同恢复上下文。
            layers.extend(["customer_profile", "case_state", "memory_event"])
        # 个性化关键词：「风格」「偏好」「客户画像」→ 偏好 + 画像层。
        if any(term in text for term in self.personalization_terms):
            # 个性化请求追加用户偏好以及顾问/客户画像。
            layers.extend(["preference", "advisor_profile", "customer_profile"])
        # 保险领域工作流必然依赖客户/从业者/case 事实。
        if domain_skill == "insurance_advisor" or workflow_name in self.business_terms:
            # 保险业务追加完整业务记忆和领域事实层。
            layers.extend(["customer_profile", "advisor_profile", "case_state", "memory_event", "domain_fact"])
        # KYC 业务字段缺失时，尝试从长期记忆中补全（如 customer_type、budget）。
        # metadata 未提供缺失字段时使用空列表，不因 None 触发遍历异常。
        missing_fields = metadata.get("missing_fields") or []
        # 命中允许从长期记忆补齐的业务字段时，追加画像与 Case 层。
        if any(field in missing_fields for field in self.missing_field_terms):
            # 缺失 KYC 业务字段时追加可能已有答案的三个状态层。
            layers.extend(["customer_profile", "advisor_profile", "case_state"])

        # ── 第三优先级：must_recall 或 ambiguous ──
        if layers:
            # 生成多条 retrieval query：原始输入 + last_entity 增强 + case_id 增强。
            # 原始输入始终是第一条 Query，实体和 Case 信息按存在性追加增强 Query。
            queries = [text]
            # 存在实体锚点时追加实体增强 Query，提高客户画像命中率。
            if session_memory.get("last_entity"):
                # 将实体与原输入拼接为第二条检索文本。
                queries.append(f"{session_memory['last_entity']} {text}")
            # 存在 Case 标识时追加 Case 增强 Query，提高任务状态命中率。
            if metadata.get("case_id") or metadata.get("opportunity_case_id"):
                # 将可用 Case ID 与原输入拼接为第三条检索文本。
                queries.append(f"case {metadata.get('case_id') or metadata.get('opportunity_case_id')} {text}")
            # 返回按出现顺序去重后的层级和全部增强 Query。
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
        # 空锚点没有可搜索的实体文本，直接视为短期未覆盖。
        if not anchor:
            # 空锚点无法在消息中确认覆盖，因此返回 False。
            return False
        # 逐条扫描最近消息，只要有一条文本包含该实体，即认为短期上下文已覆盖。
        for message in session_memory.get("recent_messages") or []:
            # 字典消息读取 content，异常类型视为无正文。
            content = message.get("content") if isinstance(message, dict) else None
            # 任一消息正文包含完整锚点即可确认短期窗口已覆盖该实体。
            if content and anchor in str(content):
                # 找到任一覆盖消息后立即返回 True，无需继续扫描窗口。
                return True
        # 扫描完成仍未命中说明实体已经滑出或从未出现在短期窗口。
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
    # 将可选参数标准化为空字典，避免后续规则层重复处理 None。
    session_memory = session_memory or {}
    # None 转为空字典，保留调用方已有 metadata 对象内容。
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
    # 规则明确跳过时不再进入模型层，避免无价值延迟和越界读取。
    # 规则明确跳过时不再调用决策模型，直接返回生产安全跳过结果。
    if rule.status == "skip_recall":
        # 兼容规划入口把规则跳过转换为统一 Decision Schema。
        return MemoryRecallDecision(
            should_recall=False,
            reason=rule.reason,
            status="rule_skip_recall",
            confidence=0.95,
        )

    # 规则不确定 + 本地无模型：安全降级，不召回长期记忆（生产链路应走 decide_long_term_memory_recall）。
    if rule.status == "ambiguous":
        # 本地入口没有模型时采用安全跳过，并保留规则 Query 供 Trace 解释。
        return MemoryRecallDecision(
            should_recall=False,
            reason=rule.reason,
            status="model_unavailable_safe_skip",
            queries=[RetrievalQuery(text=query, purpose="memory") for query in rule.queries],
            confidence=0.3,
        )

    # 规则明确召回：组装完整 MemoryRecallDecision，供 hybrid_recall_memory 使用。
    # 规则必须召回时返回允许层级、Query、领域 TopK 和强置信度。
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
    # 从函数入口开始计时，最终把完整规则/模型决策耗时写入强制过滤字段。
    started_at = time.perf_counter()
    # None 统一转换为空短期记忆，不修改调用方原字典。
    session_memory = session_memory or {}
    # 合并 tenant/user/session 到 metadata，供后续检索 filters 和模型输入使用。
    metadata = {
        **(metadata or {}),
        "tenant_id": tenant_id,
        "user_id": user_id,
        "session_id": session_id,
    }
    # 未显式注入时使用严格的 MemoryConfig 默认值，生产通常来自配置加载器。
    memory_config = memory_config or MemoryConfig()

    # ── 租户策略层：全局关闭长期记忆 ──
    if not memory_config.enabled:
        # 租户关闭记忆时返回置信度 1 的策略跳过结果。
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
        # 缺身份边界时返回安全跳过，绝不尝试匿名跨会话检索。
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
    # 规则明确跳过时立即返回，不进入模型判定或任何长期存储读取。
    if rule.status == "skip_recall":
        # 将规则跳过结果包装为生产 Decision 并保留强制 filters。
        return MemoryRecallDecision(
            should_recall=False,
            reason=rule.reason,
            status="rule_skip_recall",
            filters=metadata,
            confidence=0.95,
            latency_budget_ms=memory_config.decision_timeout_ms,
        )
    # 规则明确召回时直接构造带系统过滤器的生产决策。
    if rule.status == "must_recall":
        # 将规则层级和 Query 转换为生产检索可直接执行的 Decision。
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
        # 模型未启用/未注入时返回安全跳过，不把 ambiguous 误当作允许读取。
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
        # complete_json 以 Pydantic Schema 校验模型输出；第二个结果仅含本次无需使用的模型元数据。
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
    # 捕获模型调用、网络和 Schema 校验异常并进入安全跳过分支。
    except Exception as exc:
        # 模型 JSON 不合法或调用失败 → 不召回敏感长期记忆，保留短期 session。
        # 返回显式失败状态供观测，不向上层传播模型异常中断主请求。
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
    # 模型 filters 先展开，随后由系统强制身份和默认风险上限覆盖同名键。
    filters = {
        **output.filters,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "session_id": session_id,
        "max_risk_level": output.filters.get("max_risk_level", "medium"),
        "decision_latency_ms": int((time.perf_counter() - started_at) * 1000),
    }
    # 模型未生成 Query 时回退规则层原始 Query，确保 should_recall 不会带空检索请求。
    queries = output.queries or rule.queries
    # 返回经过系统身份覆盖的最终模型 Decision。
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
        # Repository 执行带租户/用户隔离的 pgvector Hybrid Search。
        self.repository = repository
        # Embedding Client 将多条 Query 批量映射为检索向量。
        self.embedding_client = embedding_client
        # Reranker Client 对粗排候选进行独立二阶段语义精排。
        self.reranker_client = reranker_client
        # RetrievalConfig 提供批量、超时等运行参数；未注入时使用受控默认值。
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
            # 空结果仍保留原 Decision，便于下游 Trace 解释为何没有访问数据库。
            return MemoryRecallResult(decision=decision, items=[], compact_summary={})
        # should_recall 为真却没有 Query 属于上游契约错误，不能执行无范围检索。
        if not decision.queries:
            # 显式抛错使调用方修复决策，而不是退化为全量长期记忆读取。
            raise ValueError("长期记忆召回需要至少一个 query")

        # ── Step 1: 批量 embedding + pgvector hybrid search ──
        # 保持决策 Query 顺序批量向量化，zip(strict=True) 会校验返回数量完全一致。
        query_texts = [query.text for query in decision.queries]
        # 一次批量请求获取与 Query 顺序对应的全部向量。
        embeddings = self.embedding_client.embed(query_texts)
        # merged 按 hit.id 去重，同一记忆被多条 query 命中时保留最高分。
        merged: dict[str, PersistedMemoryHit] = {}
        # Query 与向量严格一一对应，长度不一致由 strict zip 立即报错。
        for query, embedding in zip(decision.queries, embeddings, strict=True):
            # 每条 Query 分别执行 PostgreSQL 词法、向量与 metadata 融合检索。
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
            # 同一记忆被多条改写 Query 命中时，只保留 final_score 最高的一次。
            for hit in hits:
                # 读取该 ID 当前保留的最佳命中。
                previous = merged.get(hit.id)
                # 首次命中或本条分数更高时更新去重映射。
                if previous is None or hit.final_score > previous.final_score:
                    # 首次命中或更高分命中覆盖去重映射。
                    merged[hit.id] = hit

        # ── Step 2: Reranker 精排 ──
        # 在调用 Reranker 前按粗排 final_score 稳定排序，确保索引映射可复现。
        candidates = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)
        # 只有存在候选才调用外部 Reranker，空结果直接跳过网络请求。
        if candidates:
            # Reranker 使用合并 Query 评估全部候选，仅请求最终 top_k 个索引。
            ranking = self.reranker_client.rerank(
                query=" ".join(query_texts),
                documents=[hit.content for hit in candidates],
                top_k=decision.top_k,
            )
            # 防御性忽略模型返回的越界索引，合法索引按 Reranker 顺序映射回候选。
            ranked = [candidates[item.index] for item in ranking if item.index < len(candidates)]
        # 没有候选时进入空列表分支，不调用 Reranker。
        else:
            # 无候选时保持空列表，不调用外部 Reranker。
            ranked = candidates

        # ── Step 3: 转换 + 阈值过滤 + 压缩摘要 ──
        items = [
            _memory_hit_to_recall_item(hit)
            for hit in ranked[: decision.top_k]
            if hit.final_score >= decision.score_threshold
        ]
        # 返回阈值过滤后的原子条目及其固定分层摘要。
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
    # 累积由每个偏好键/候选转换的独立检索文档。
    documents: list[RetrievalDocument] = []
    # 每个普通键单独成文档，memory_candidates 列表再按候选逐项展开。
    for key, value in preference_memory.items():
        # memory_candidates 是 long_term_memory_candidate 节点写入的候选列表，需逐条展开。
        if key == "memory_candidates" and isinstance(value, list):
            # 列表下标形成稳定 chunk_id，使同一次快照中的候选可独立去重。
            for index, candidate in enumerate(value):
                # 字典候选读取显式 type，异常结构使用统一 memory_candidate 类型。
                candidate_type = str(candidate.get("type", "memory_candidate")) if isinstance(candidate, dict) else "memory_candidate"
                # 字典候选读取 value，标量候选直接作为值。
                candidate_value = candidate.get("value") if isinstance(candidate, dict) else candidate
                # 将候选类型和值写入带 Preference metadata 的检索文档。
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
        # 普通 Preference 键值进入一文档一字段分支。
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
    # 返回拆分后的全部 Preference 检索文档，空输入自然得到空列表。
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
    # 累积客户、顾问、Case 和事件四类业务检索文档。
    documents: list[RetrievalDocument] = []

    # ── 客户画像事实 ──
    for fact in customer_facts or []:
        # 过期事实和 PII 不进入检索候选集，避免隐私泄露和过时信息污染。
        if not fact.is_current or fact.sensitivity_level == "pii":
            # 过期或 PII 客户事实不构建任何检索文本。
            continue
        # 标准化值优先，缺失时回退原始事实值。
        value = fact.normalized_value if fact.normalized_value is not None else fact.fact_value
        # 每条安全客户事实转换为独立 customer_profile 文档。
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
        # 非当前顾问事实已经被新版本替代，不参与任何在线检索。
        if not fact.is_current:
            # 非当前顾问事实不参与检索文档构建。
            continue
        # 每条当前顾问事实转换为独立 advisor_profile 文档。
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
        # 当前 Case 的关键路由字段合成为一条 Case 状态文档。
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
        # 每个事件单独成文档，保留 event_type 和 evidence 摘要。
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
    # 返回混合业务层文档集合，由召回 Decision 的 library 过滤进一步收窄。
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
        # 短路结果保留决策对象，且不初始化 HybridRetriever。
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
    # 合并全部 Query 文本用于本地字符重叠 bonus。
    query_text = " ".join(query.text for query in decision.queries)
    # items 累积通过租户和层级二次校验的召回条目。
    items: list[MemoryRecallItem] = []
    # 逐个粗排结果执行租户/层级二次过滤并累加业务加分。
    for result in results:
        # 读取 RetrievalDocument 的强类型 metadata。
        metadata = result.document.metadata
        # 租户隔离二次校验（HybridRetriever 已过滤，此处防御性检查）。
        if metadata.tenant_id != tenant_id:
            # 防御性丢弃任何跨租户粗排结果。
            continue
        # 复制 extra 避免后续读取修改文档元数据对象。
        extra = dict(metadata.extra)
        # layer 决定条目是否属于 Decision 授权范围。
        layer = extra.get("layer")
        # 只保留 decision.recall_layers 允许的层级。
        if layer not in decision.recall_layers:
            # 未授权记忆层即使文本相关也不能进入上下文。
            continue

        # ── 记忆业务 bonus 累加 ──
        bonus = 0.0
        # 当前事实优先于历史事实；缺省 True 兼容不带版本字段的 Case/Preference 文档。
        if extra.get("is_current", True):
            # 当前记录增加 0.05 业务权重。
            bonus += 0.05
        # 用户已确认事实优先于 uncertain 线索。
        if extra.get("certainty") == "confirmed":
            # 已确认事实再增加 0.05。
            bonus += 0.05
        # Case 状态对保险任务推进更直接，因此给予小幅固定加分。
        if metadata.library == "case_memory":
            # Case 聚合状态增加 0.03。
            bonus += 0.03
        # 追加最高 0.18 的中文字符重叠加分。
        bonus += _character_overlap_bonus(query_text, result.document.text)

        # 最终分限制在 1.0 内，避免多项 bonus 超出标准范围。
        final_score = min(1.0, result.score + bonus)
        # 构造统一 RecallItem 并截断可能过长的文档正文。
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
    # 最终按 rerank_score 降序返回，后续统一应用阈值和 TopK。
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
    # 复制数据库 metadata，避免 setdefault 修改原命中对象。
    metadata = dict(hit.metadata)
    # 补齐记忆类型，已有显式值时不覆盖。
    metadata.setdefault("memory_type", hit.memory_type)
    # 补齐 PostgreSQL pgvector 来源标签供审计。
    metadata.setdefault("source", "postgres_pgvector")
    # 转换为与本地检索一致的 RecallItem，供统一摘要逻辑消费。
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
    # 初始化固定分层结构，保证即使某层无命中也有稳定容器类型。
    summary: dict[str, Any] = {
        "preference": {},
        "customer_profile": {"confirmed": {}, "uncertain": {}},
        "advisor_profile": {},
        "case_state": {},
        "memory_events": [],
    }
    # 按层级分派到固定摘要结构，禁止把任意 metadata 平铺到顶层。
    for item in items:
        # Preference 以稳定 key/value 形式合并。
        if item.layer == "preference":
            # 优先使用 metadata key，缺失时回退 chunk_id。
            key = str(item.metadata.get("key", item.chunk_id))
            # 优先使用结构化 value，缺失时回退截断内容。
            summary["preference"][key] = item.metadata.get("value", item.content)
        # 客户画像必须继续区分 confirmed 和 uncertain 两个桶。
        elif item.layer == "customer_profile":
            # 读取 certainty 并缺省视为 confirmed。
            certainty = item.metadata.get("certainty", "confirmed")
            # uncertain 单独入桶，其他值保守落入 confirmed 兼容历史数据。
            bucket = "uncertain" if certainty == "uncertain" else "confirmed"
            # 事实键缺失时回退 chunk_id 保持可寻址。
            key = str(item.metadata.get("fact_key", item.chunk_id))
            # 写入对应 certainty 桶的结构化事实值。
            summary["customer_profile"][bucket][key] = item.metadata.get("fact_value", item.content)
        # 顾问画像直接按事实键合并当前值。
        elif item.layer == "advisor_profile":
            # 顾问事实同样优先使用 fact_key。
            key = str(item.metadata.get("fact_key", item.chunk_id))
            # 写入结构化事实值，缺失时回退内容摘要。
            summary["advisor_profile"][key] = item.metadata.get("fact_value", item.content)
        # Case 状态是单一聚合对象，使用 metadata 更新对应分区。
        elif item.layer == "case_state":
            # Case metadata 合并到唯一 Case 状态分区。
            summary["case_state"].update(item.metadata)
        # 事件保持列表形式，避免多个同类事件互相覆盖。
        elif item.layer == "memory_event":
            # 事件 metadata 按排序顺序追加，保留多事件历史。
            summary["memory_events"].append(item.metadata)
    # 返回固定五分区摘要；未命中分区保持空对象/空列表。
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
    # 返回带低风险、准入标记和业务 extra 的统一检索文档。
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
    # 原始输入是首条最高权重长期记忆 Query。
    queries = [RetrievalQuery(text=text, purpose="memory", weight=1.0)]
    # 读取上一意图用于可选增强 Query。
    last_intent = session_memory.get("last_intent")
    # 存在上一意图时追加低权重意图增强 Query。
    if last_intent:
        # 添加权重 0.7 的意图增强 Query。
        queries.append(RetrievalQuery(text=f"{text} {last_intent}", purpose="memory", weight=0.7))
    # 读取上一实体用于可选实体增强 Query。
    last_entity = session_memory.get("last_entity")
    # 存在实体锚点时追加较高权重实体增强 Query。
    if last_entity:
        # 添加权重 0.8 的实体增强 Query。
        queries.append(RetrievalQuery(text=f"{text} {last_entity}", purpose="memory", weight=0.8))
    # 领域 Skill 作为画像检索 Query，权重低于用户原始文本。
    if domain_skill:
        # 添加权重 0.6 的领域画像 Query。
        queries.append(RetrievalQuery(text=f"{domain_skill} {text}", purpose="profile", weight=0.6))
    # 保持原始 Query 在首位并返回所有存在的增强 Query。
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
    # 已分类的工具型意图无需长期记忆。
    if intent in {"weather_query", "calculator_query"}:
        # 已分类工具意图直接判为低长期记忆价值。
        return True
    # 小写文本用于中英文关键词统一判断。
    lowered = text.lower()
    # 同时含运算符和数字的表达式按一次性计算请求处理。
    if any(symbol in text for symbol in ["+", "-", "*", "/"]) and any(char.isdigit() for char in text):
        # 算术表达式无需画像或历史事实。
        return True
    # 中英文天气/计算关键词同样构成明确跳过信号。
    if any(keyword in lowered for keyword in ["天气", "weather", "计算", "calculator"]):
        # 关键词工具请求同样跳过长期召回。
        return True
    # 简单问候没有个性化事实依赖，跳过长期召回。
    if text in {"你好", "hello", "hi", "在吗"}:
        # 简单问候不依赖跨会话事实。
        return True
    # 未命中任一低价值条件时交给其他召回规则继续判断。
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
    # 稳定偏好召回的显式触发词列表。
    preference_terms = ["我喜欢", "我的偏好", "按我的", "上次我说", "记得我", "风格", "继续用"]
    # 命中显式偏好词或处于任一领域 Skill 时返回需要偏好召回。
    return any(term in text for term in preference_terms) or domain_skill is not None


def _needs_business_recall(
    text: str,
    workflow_name: str,
    intent: str | None,
    domain_skill: str | None,
    metadata: dict[str, Any],
) -> bool:
    """判断是否需要召回客户/从业者/case 业务长期记忆（历史遗留 helper）。

    触发条件：保险顾问 intent/skill、metadata 含 customer_id/case_id、
    或输入含保险业务关键词。当前主链路由 ``MemoryRecallRuleEngine.business_terms`` 覆盖。

    参数:
        text: 用户输入。
        workflow_name: 兼容运行标签；该 helper 不使用它判断保险路由。
        intent: 意图。
        domain_skill: 领域 Skill。
        metadata: 请求 metadata。

    返回:
        True 表示可能需要召回 customer_profile / case_state 等业务层。
    """
    # 兼容参数当前不参与判断，显式删除避免未使用变量误导读者。
    del workflow_name
    # 保险 Skill 或保险意图明确需要客户、顾问和 Case 业务记忆。
    if domain_skill == "insurance_advisor" or intent in {
        "insurance_break_ice",
        "insurance_objection_handling",
        "insurance_strategy",
        "insurance_kyc_collection",
    }:
        # 保险 Skill/意图明确需要业务长期记忆。
        return True
    # 已存在内部客户/Case 关联时说明当前请求具有业务上下文依赖。
    if metadata.get("customer_id") or metadata.get("opportunity_case_id"):
        # 内部主体关联存在时视为业务上下文请求。
        return True
    # 业务上下文兜底触发词列表。
    business_terms = ["客户", "这个人", "这个客户", "上次", "之前", "继续", "保险", "kyc", "破冰", "异议", "策略"]
    # 兜底按业务关键词判断是否可能需要画像/Case 召回。
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
    # 提取 Query 中唯一汉字集合。
    query_chars = {char for char in query_text if "\u4e00" <= char <= "\u9fff"}
    # 提取文档中唯一汉字集合。
    document_chars = {char for char in document_text if "\u4e00" <= char <= "\u9fff"}
    # 任一侧没有汉字时无法计算中文字符重叠，返回零加分。
    if not query_chars or not document_chars:
        # 无可比较中文字符时不提供额外排序加分。
        return 0.0
    # 以 Query 汉字数为分母计算覆盖率，max 防止除零。
    overlap = len(query_chars & document_chars) / max(len(query_chars), 1)
    # 按重叠比例线性缩放，并把 bonus 上限限制为 0.18。
    return min(0.18, overlap * 0.18)


def _truncate_memory_content(text: str, limit: int = 220) -> str:
    """限制单条记忆进入上下文的长度，防止长 evidence 撑爆 token budget。

    参数:
        text: 记忆原始文本（可能含 evidence_text）。
        limit: 最大字符数，默认 220。

    返回:
        截断后的文本，超长时末尾追加 ``...``。
    """
    # 未超限原样返回；超限截取前 limit 字符并追加省略号。
    return text if len(text) <= limit else text[:limit] + "..."
