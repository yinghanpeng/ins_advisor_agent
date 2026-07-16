"""PostgreSQL / pgvector repository.

本模块是生产数据平面的入口。所有方法都要求传入 tenant_id，并且查询条件中显式使用
tenant_id，这是多租户隔离最重要的一条边界。状态迁移、trace、工具调用、RAG 和长期记忆
都写入 PostgreSQL，AgentState 因此可以被审计、回放和做离线评测。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from agent_core.config.runtime import DatabaseConfig, RetrievalConfig
from agent_core.guardrails.output_pii import scan_and_redact_output_pii


# 风险等级映射为可比较整数，SQL 可用单个上界过滤 low/medium/high。
RISK_RANK = {"low": 0, "medium": 1, "high": 2}


class PersistedMemoryHit(BaseModel):
    """从 pgvector 长期记忆表召回的一条结果。"""

    # id 是 memory_items 主键，同时作为多 Query 去重键。
    id: str = Field(..., description="长期记忆条目 ID。")
    # scope/memory_type 分别描述可召回业务层与记录语义类型。
    scope: str = Field(..., description="记忆作用域，例如 preference、customer_profile。")
    memory_type: str = Field(..., description="记忆类型，例如 fact、event、case_state。")
    # content 是允许进入上下文的脱敏摘要，证据原文另行加密存储。
    content: str = Field(..., description="进入上下文前的记忆摘要。")
    # metadata 提供 Case、事实键等结构化过滤/压缩信息。
    metadata: dict[str, Any] = Field(default_factory=dict, description="记忆 metadata。")
    # 以下六个分数保留混合检索各分量与最终融合结果，便于解释排序。
    vector_score: float = Field(default=0.0, description="pgvector 相似度分。")
    lexical_score: float = Field(default=0.0, description="PostgreSQL 全文检索分。")
    metadata_score: float = Field(default=0.0, description="scope、risk、consent 等 metadata 匹配分。")
    recency_score: float = Field(default=0.0, description="按更新时间计算的新近度分。")
    confidence_score: float = Field(default=0.0, description="长期记忆自身置信度分。")
    final_score: float = Field(default=0.0, description="按配置权重融合后的最终分。")


class PersistedRagHit(BaseModel):
    """从 pgvector RAG chunk 表召回的一条结果。"""

    # document_id/chunk_id 组成 RAG 命中的文档与片段溯源标识。
    document_id: str = Field(..., description="命中文档 ID。")
    chunk_id: str = Field(..., description="命中 chunk ID。")
    # content 为通过生成准入过滤的 Chunk 正文。
    content: str = Field(..., description="chunk 正文。")
    # metadata 与 source_uri 保存知识库、版本和原始来源信息。
    metadata: dict[str, Any] = Field(default_factory=dict, description="chunk metadata。")
    source_uri: str | None = Field(default=None, description="原始来源 URI。")
    # 三个分量和 final_score 支撑在线阈值过滤与离线召回分析。
    vector_score: float = Field(default=0.0, description="pgvector 相似度分。")
    lexical_score: float = Field(default=0.0, description="PostgreSQL 全文检索分。")
    metadata_score: float = Field(default=0.0, description="metadata 过滤与匹配分。")
    final_score: float = Field(default=0.0, description="融合后的最终检索分。")


class PostgresAgentRepository:
    """生产 Agent Repository。

    这里不包含业务推理逻辑，只负责真实持久化和 pgvector 检索。业务节点负责决定是否写、
    是否召回、如何压缩；Repository 负责保证写入的记录带 tenant_id、trace_id 等审计字段。
    """

    def __init__(
        self,
        database: DatabaseConfig,
        retrieval: RetrievalConfig | None = None,
        *,
        encryption_key: str | None = None,
    ) -> None:
        """创建共享 PostgreSQL Engine，并保存检索及加密配置。"""

        # 空数据库 URL 无法满足生产持久化语义，在创建连接池前立即失败。
        if not database.database_url:
            # 显式异常阻止应用悄悄退化为内存数据库或无持久化模式。
            raise RuntimeError("DATABASE_URL 不能为空，生产持久化需要真实 PostgreSQL")
        # 保存数据库连接池配置，便于生命周期管理和诊断。
        self.database = database
        # 检索权重/阈值未注入时使用受控默认配置。
        self.retrieval = retrieval or RetrievalConfig()
        # 通用生成输出和审计消息需要字段加密；测试可不配置，调用加密方法时会显式失败。
        self.encryption_key = encryption_key
        # Engine 由 Repository 生命周期复用；pre_ping/recycle 处理失效的池连接。
        self.engine: Engine = create_engine(
            database.database_url,
            pool_size=database.pool_size,
            max_overflow=database.max_overflow,
            pool_timeout=database.pool_timeout_seconds,
            pool_recycle=database.pool_recycle_seconds,
            pool_pre_ping=database.pool_pre_ping,
            echo=database.echo_sql,
            future=True,
        )

    def insert_agent_run(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> None:
        """写入一次 Agent run 的入口记录。"""
        # 按 trace_id 幂等 Upsert 入口 Payload，HTTP 重试会刷新同一 Run 而非重复创建。
        self._execute(
            """
            INSERT INTO agent_runs (tenant_id, trace_id, payload)
            VALUES (:tenant_id, :trace_id, CAST(:payload AS jsonb))
            ON CONFLICT (trace_id) DO UPDATE
            SET payload = EXCLUDED.payload, updated_at = now()
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            payload=_json(payload),
        )

    def insert_trace_event(self, *, tenant_id: str, trace_id: str, event: dict[str, Any]) -> None:
        """写入结构化 trace event。"""
        # Trace Event 采用 append-only INSERT，保留节点/事件名及完整结构化审计 Payload。
        self._execute(
            """
            INSERT INTO agent_trace_events (tenant_id, trace_id, span_id, node_name, event_name, payload)
            VALUES (:tenant_id, :trace_id, :span_id, :node_name, :event_name, CAST(:payload AS jsonb))
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            span_id=event.get("span_id"),
            node_name=event.get("node_name"),
            event_name=event.get("event") or event.get("trace_event_name") or "unknown",
            payload=_json(event),
        )

    def insert_state_transition(self, *, tenant_id: str, trace_id: str, transition: dict[str, Any]) -> None:
        """写入状态迁移记录，专门支持 workflow replay。"""
        # 单独保存 from/to/reason 便于按状态查询，同时保留扩展 metadata JSONB。
        self._execute(
            """
            INSERT INTO state_transitions (tenant_id, trace_id, from_state, to_state, reason, metadata)
            VALUES (:tenant_id, :trace_id, :from_state, :to_state, :reason, CAST(:metadata AS jsonb))
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            from_state=transition.get("from_state"),
            to_state=transition.get("to_state"),
            reason=transition.get("reason", ""),
            metadata=_json(transition.get("metadata", {})),
        )

    def insert_memory_recall_decision(self, *, tenant_id: str, trace_id: str, decision: dict[str, Any]) -> None:
        """写入长期记忆召回决策，便于分析召回是否过度或不足。"""
        # 决策作为不可变审计 JSONB 保存，用 trace_id 关联对应 Agent Run。
        self._execute(
            """
            INSERT INTO memory_recall_decisions (tenant_id, trace_id, decision)
            VALUES (:tenant_id, :trace_id, CAST(:decision AS jsonb))
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            decision=_json(decision),
        )

    def insert_memory_recall_result(self, *, tenant_id: str, trace_id: str, result: dict[str, Any]) -> None:
        """写入长期记忆召回结果摘要。"""
        # 只保存上层传入的召回摘要；证据密文仍保留在 memory_items 独立字段。
        self._execute(
            """
            INSERT INTO memory_recall_results (tenant_id, trace_id, result)
            VALUES (:tenant_id, :trace_id, CAST(:result AS jsonb))
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            result=_json(result),
        )

    def upsert_long_term_memory_item(
        self,
        *,
        tenant_id: str,
        user_id: str,
        scope: str,
        memory_type: str,
        memory_key: str,
        content: str,
        embedding: Sequence[float] | None,
        source_type: str,
        source_id: str,
        evidence_text: str,
        confidence: float,
        risk_level: str = "low",
        consent_status: str = "granted",
        normalized_content: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
        embedding_model: str = "configured-runtime",
    ) -> str:
        """写入或更新长期记忆条目。

        长期记忆必须有 evidence_text 和 source_type。没有证据的模型推断不能落成 confirmed
        记忆，否则后续请求会把推测当事实召回。
        """
        # 长期记忆必须同时绑定租户和用户，缺任一身份都拒绝写入。
        if not tenant_id or not user_id:
            # 显式异常阻止无主体事实成为可跨用户召回的孤立记录。
            raise ValueError("长期记忆写入必须带 tenant_id 和 user_id")
        # 来源、证据和稳定 memory_key 都是版本化 Upsert 的必填条件。
        if not source_type or not evidence_text or not memory_key:
            # 无法溯源或无法稳定去重的长期记忆不允许持久化。
            raise ValueError("长期记忆写入必须带 source_type 和 evidence_text")
        # risk_level 必须能映射为 SQL 比较使用的 risk_rank。
        if risk_level not in RISK_RANK:
            # 拒绝未知等级，避免默认落入低风险而绕过召回上限。
            raise ValueError(f"不支持的长期记忆风险等级：{risk_level}")
        # evidence_text 必须使用 pgcrypto 密文保存，因此写入路径强制要求有效密钥。
        if not self.encryption_key or len(self.encryption_key) < 24:
            # 密钥缺失时 fail-closed，不能把证据改写成明文列。
            raise RuntimeError("长期记忆证据持久化需要 MEMORY_ENCRYPTION_KEY")
        # 按 tenant/user/scope/key Upsert；相同事实增强证据和置信度并递增版本。
        row = self._fetch_one(
            """
            INSERT INTO memory_items (
                tenant_id, user_id, scope, memory_type, memory_key, content, normalized_content,
                source_type, source_id, evidence_ciphertext, evidence_hash,
                confidence, status, risk_level, risk_rank,
                consent_status, expires_at, metadata, version
            )
            VALUES (
                :tenant_id, :user_id, :scope, :memory_type, :memory_key, :content, :normalized_content,
                :source_type, :source_id,
                pgp_sym_encrypt(:evidence_text, :encryption_key, 'cipher-algo=aes256'),
                encode(digest(:evidence_text, 'sha256'), 'hex'),
                :confidence, 'active', :risk_level,
                :risk_rank, :consent_status, :expires_at, CAST(:metadata AS jsonb), 1
            )
            ON CONFLICT (tenant_id, user_id, scope, memory_key) DO UPDATE SET
                memory_type = EXCLUDED.memory_type,
                content = EXCLUDED.content,
                normalized_content = EXCLUDED.normalized_content,
                source_type = EXCLUDED.source_type,
                source_id = EXCLUDED.source_id,
                evidence_ciphertext = EXCLUDED.evidence_ciphertext,
                evidence_hash = EXCLUDED.evidence_hash,
                confidence = GREATEST(memory_items.confidence, EXCLUDED.confidence),
                status = 'active',
                risk_level = EXCLUDED.risk_level,
                risk_rank = EXCLUDED.risk_rank,
                consent_status = EXCLUDED.consent_status,
                expires_at = EXCLUDED.expires_at,
                deleted_at = NULL,
                metadata = memory_items.metadata || EXCLUDED.metadata,
                version = memory_items.version + 1,
                updated_at = now()
            RETURNING id, version
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            scope=scope,
            memory_type=memory_type,
            memory_key=memory_key,
            content=content,
            normalized_content=normalized_content or content,
            source_type=source_type,
            source_id=source_id,
            evidence_text=evidence_text,
            encryption_key=self.encryption_key,
            confidence=confidence,
            risk_level=risk_level,
            risk_rank=RISK_RANK[risk_level],
            consent_status=consent_status,
            expires_at=expires_at,
            metadata=_json(metadata or {}),
        )
        # Embedding 与事实内容分表更新，内容或 TTL 更新时不会重写大向量。
        # 只有调用方提供向量时才写独立 Embedding 表；规则偏好允许暂时没有向量。
        if embedding is not None:
            # Embedding Upsert 与内容记录通过 memory_item_id 关联，并记录模型/维度。
            self._execute(
                """
                INSERT INTO memory_item_embeddings (
                    memory_item_id, tenant_id, embedding_model, embedding_dimensions, embedding
                )
                VALUES (:memory_item_id, :tenant_id, :embedding_model, :dimensions, CAST(:embedding AS halfvec))
                ON CONFLICT (memory_item_id) DO UPDATE SET
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dimensions = EXCLUDED.embedding_dimensions,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                memory_item_id=row["id"],
                tenant_id=tenant_id,
                embedding_model=embedding_model,
                dimensions=len(embedding),
                embedding=_vector_literal(embedding),
            )
        # 返回数据库最终主键，供后续血缘、Embedding 或 Trace 关联。
        return str(row["id"])

    def search_long_term_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query: str,
        query_embedding: Sequence[float],
        scopes: list[str],
        case_id: str | None = None,
        max_risk_level: str = "medium",
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[PersistedMemoryHit]:
        """用 pgvector + PostgreSQL 全文检索召回长期记忆。

        final_score 的权重来自 configs/retrieval.yaml。向量分解决语义相关，全文分解决关键词精确命中，
        metadata 分保证 scope/risk/consent 不越界，recency/confidence 防止过旧或低置信记忆污染上下文。
        """
        # 空 scopes 无法形成最小权限过滤条件，禁止退化为所有长期记忆层。
        if not scopes:
            # 契约错误显式抛出，调用方必须由决策层给出至少一个范围。
            raise ValueError("长期记忆检索 scopes 不能为空")
        # 风险上限必须使用已知等级，才能安全转换为 SQL 整数比较。
        if max_risk_level not in RISK_RANK:
            # 未知等级默认拒绝而不是视为最高权限。
            raise ValueError(f"不支持的长期记忆风险等级：{max_risk_level}")
        # top_k 缺省取配置值；显式传入值由 SQL LIMIT 控制。
        limit = top_k or self.retrieval.top_k
        # 阈值未覆盖时使用全局检索配置，允许单次决策收紧或放宽。
        threshold = self.retrieval.score_threshold if score_threshold is None else score_threshold
        # SQL 在同一 CTE 计算向量、全文、授权/风险、时效和置信度并按配置融合。
        rows = self._fetch_all(
            """
            WITH scored AS (
                SELECT
                    id, scope, memory_type, content, metadata,
                    CASE
                        WHEN e.embedding IS NULL THEN 0.0
                        ELSE GREATEST(0, 1 - (e.embedding <=> CAST(:embedding AS halfvec)))
                    END AS vector_score,
                    ts_rank_cd(
                        to_tsvector('simple', coalesce(content, '') || ' ' || coalesce(normalized_content, '')),
                        plainto_tsquery('simple', :query)
                    ) AS lexical_score,
                    CASE
                        WHEN consent_status = 'granted' AND status = 'active' THEN 1.0
                        ELSE 0.0
                    END AS metadata_score,
                    LEAST(1, GREATEST(0, 1 - EXTRACT(EPOCH FROM (now() - m.updated_at)) / 2592000.0)) AS recency_score,
                    m.confidence AS confidence_score
                FROM memory_items m
                LEFT JOIN memory_item_embeddings e
                  ON e.memory_item_id = m.id AND e.tenant_id = m.tenant_id
                WHERE m.tenant_id = :tenant_id
                  AND m.user_id = :user_id
                  AND m.scope = ANY(:scopes)
                  AND m.status = 'active'
                  AND m.deleted_at IS NULL
                  AND m.consent_status = 'granted'
                  AND (m.scope <> 'preference' OR EXISTS (
                      SELECT 1 FROM memory_consents consent
                      WHERE consent.tenant_id=m.tenant_id
                        AND consent.subject_type='user'
                        AND consent.subject_id=m.user_id
                        AND consent.purpose='preference_memory'
                        AND consent.status='granted'
                  ))
                  AND m.risk_rank <= :max_risk_rank
                  AND (m.expires_at IS NULL OR m.expires_at > now())
                  AND (:case_id IS NULL OR m.metadata ->> 'case_id' = :case_id)
            )
            SELECT *,
                (:vector_weight * vector_score
                 + :lexical_weight * lexical_score
                 + :metadata_weight * metadata_score
                 + :recency_weight * recency_score
                 + :confidence_weight * confidence_score) AS final_score
            FROM scored
            WHERE (:vector_weight * vector_score
                   + :lexical_weight * lexical_score
                   + :metadata_weight * metadata_score
                   + :recency_weight * recency_score
                   + :confidence_weight * confidence_score) >= :threshold
            ORDER BY final_score DESC
            LIMIT :limit
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            scopes=scopes,
            case_id=case_id,
            max_risk_rank=RISK_RANK[max_risk_level],
            query=query,
            embedding=_vector_literal(query_embedding),
            vector_weight=self.retrieval.vector_weight,
            lexical_weight=self.retrieval.lexical_weight,
            metadata_weight=self.retrieval.metadata_weight,
            recency_weight=self.retrieval.recency_weight,
            confidence_weight=self.retrieval.confidence_weight,
            threshold=threshold,
            limit=limit,
        )
        # 每行通过 Pydantic 校验后返回，防止数据库列变化静默污染检索结果。
        return [PersistedMemoryHit.model_validate(dict(row)) for row in rows]

    def append_short_term_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message_key: str,
        trace_id: str | None,
        speaker_role: str,
        content: str,
        encryption_key: str,
        retention_days: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """幂等追加加密审计消息；在线读取仍以 Redis 为主。"""
        # 低权限查询只看到脱敏文本；完整原文使用 pgcrypto 加密保存。
        redacted, _scan = scan_and_redact_output_pii(content)
        # tenant/session/message_key 构成幂等键；正文密文、脱敏副本和 Hash 同时写入。
        self._execute(
            """
            INSERT INTO short_term_messages (
                tenant_id, session_id, message_key, trace_id, speaker_role,
                content_ciphertext, content_redacted, content_hash, metadata, expires_at
            ) VALUES (
                :tenant_id, :session_id, :message_key, :trace_id, :speaker_role,
                pgp_sym_encrypt(:content, :encryption_key, 'cipher-algo=aes256'),
                :content_redacted, encode(digest(:content, 'sha256'), 'hex'), CAST(:metadata AS jsonb),
                now() + make_interval(days => :retention_days)
            )
            ON CONFLICT (tenant_id, session_id, message_key) DO NOTHING
            """,
            tenant_id=tenant_id,
            session_id=session_id,
            message_key=message_key,
            trace_id=trace_id,
            speaker_role=speaker_role,
            content=content,
            content_redacted=redacted,
            encryption_key=encryption_key,
            retention_days=retention_days,
            metadata=_json(metadata or {}),
        )

    def list_short_term_messages(
        self,
        *,
        tenant_id: str,
        session_id: str,
        encryption_key: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按时间顺序读取审计消息；只有显式提供密钥时才返回解密原文。"""
        # 默认只查询脱敏文本，避免普通恢复链路意外取得原始敏感数据。
        # 只有显式提供密钥才将 SELECT 表达式切换为 pgcrypto 解密，否则固定返回 redacted 列。
        content_expression = (
            "pgp_sym_decrypt(content_ciphertext, :encryption_key)::text"
            if encryption_key
            else "content_redacted"
        )
        # 查询始终按租户与 Session 过滤，并将 limit 限制在 1-500 的安全范围。
        rows = self._fetch_all(
            f"""
            SELECT message_key, trace_id, speaker_role, {content_expression} AS content,
                   content_redacted, metadata, created_at
            FROM short_term_messages
            WHERE tenant_id = :tenant_id AND session_id = :session_id
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            tenant_id=tenant_id,
            session_id=session_id,
            encryption_key=encryption_key,
            limit=max(1, min(limit, 500)),
        )
        # SQL 为高效取最近记录采用倒序，API 返回前反转为自然时间正序。
        return list(reversed(rows))

    def read_preference_memory(self, *, tenant_id: str, user_id: str) -> dict[str, Any]:
        """读取用户已授权且未过期的 Preference，并恢复兼容 memory_candidates 结构。"""
        # 查询同时要求条目自身 consent_status 与权威 Consent 表均为 granted。
        rows = self._fetch_all(
            """
            SELECT memory_key, content, metadata, confidence, version
            FROM memory_items
            WHERE tenant_id = :tenant_id
              AND user_id = :user_id
              AND scope = 'preference'
              AND status = 'active'
              AND deleted_at IS NULL
              AND consent_status = 'granted'
              AND EXISTS (
                  SELECT 1 FROM memory_consents consent
                  WHERE consent.tenant_id=memory_items.tenant_id
                    AND consent.subject_type='user'
                    AND consent.subject_id=memory_items.user_id
                    AND consent.purpose='preference_memory'
                    AND consent.status='granted'
              )
              AND (expires_at IS NULL OR expires_at > now())
            ORDER BY updated_at DESC
            """,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        # 将关系行恢复为 MemoryManager 兼容的 memory_candidates 列表结构。
        candidates = []
        # 每条 Preference 仅暴露类型、值、置信度和版本，不返回加密证据。
        for row in rows:
            # metadata 缺失时使用空字典并回退稳定 memory_key/content。
            metadata = row.get("metadata") or {}
            # 将当前数据库 Preference 行转换为兼容候选结构。
            candidates.append(
                {
                    "type": metadata.get("preference_type") or row["memory_key"],
                    "value": metadata.get("value", row["content"]),
                    "confidence": row["confidence"],
                    "version": row["version"],
                }
            )
        # 没有有效授权偏好时返回空对象，避免生成一个空候选字段误导上层。
        return {"memory_candidates": candidates} if candidates else {}

    def merge_preference_candidates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        candidates: list[dict[str, Any]],
        ttl_days: int,
    ) -> int:
        """按 preference_type 去重合并偏好；不会用整句用户输入覆盖历史列表。"""
        # 没有显式同意时不写长期偏好，调用方通过返回 0 写入降级 trace。
        # Consent 检查缺失或 revoked 时 fail-closed，不执行任何 upsert。
        if not self.has_memory_consent(
            tenant_id=tenant_id,
            subject_type="user",
            subject_id=user_id,
            purpose="preference_memory",
        ):
            # 缺少有效授权时返回零写入并保持数据库不变。
            return 0
        # 本批候选共用根据配置计算的 UTC 过期时间。
        expires_at = (datetime.now(UTC) + timedelta(days=ttl_days)).isoformat()
        # written 统计实际写入的非空候选数量。
        written = 0
        # 每个候选按 preference_type 独立 Upsert，避免整句输入覆盖历史列表。
        for candidate in candidates:
            # 缺省类型使用明确占位符；value 仍需通过下一步非空判断。
            preference_type = str(candidate.get("type") or "unknown_preference")
            # 读取候选值并在下一步过滤空标量/容器。
            value = candidate.get("value")
            # 空标量或空容器没有稳定偏好语义，直接跳过。
            if value in (None, "", [], {}):
                # 继续处理本批下一候选。
                continue
            # 规范化 JSON 作为稳定内容，sort_keys 消除字典键顺序导致的伪差异。
            normalized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            # 复用长期记忆 Upsert 的证据、风险、加密和版本边界。
            self.upsert_long_term_memory_item(
                tenant_id=tenant_id,
                user_id=user_id,
                scope="preference",
                memory_type="preference",
                memory_key=preference_type,
                content=normalized,
                normalized_content=normalized,
                embedding=None,
                source_type=str(candidate.get("source_type") or "user_message"),
                source_id=str(candidate.get("source_id") or f"preference:{user_id}:{preference_type}"),
                evidence_text=str(candidate.get("evidence_text") or "用户明确表达稳定偏好"),
                confidence=float(candidate.get("confidence", 0.8)),
                risk_level=str(candidate.get("risk_level") or "low"),
                consent_status="granted",
                expires_at=expires_at,
                metadata={"preference_type": preference_type, "value": value},
            )
            # 只有完整执行 Upsert 后才增加成功数量。
            written += 1
        # 返回本批实际写入数，Consent 拒绝或全空候选均为零。
        return written

    def export_user_memory(self, *, tenant_id: str, user_id: str) -> dict[str, Any]:
        """导出用户长期记忆和召回审计摘要，不导出其它用户数据。"""
        # 导出按 tenant/user 精确过滤，包含条目状态和 metadata 但不包含证据密文/明文。
        items = self._fetch_all(
            """
            SELECT id, scope, memory_type, memory_key, content, source_type, source_id,
                   confidence, status, risk_level, consent_status, version,
                   created_at, updated_at, expires_at, metadata
            FROM memory_items
            WHERE tenant_id = :tenant_id AND user_id = :user_id
            ORDER BY updated_at DESC
            """,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        # 返回带主体边界的结构化结果，便于隐私响应核对范围。
        return {"tenant_id": tenant_id, "user_id": user_id, "memory_items": items}

    def withdraw_memory_consent(
        self,
        *,
        tenant_id: str,
        user_id: str,
        policy_version: str = "runtime-revocation",
    ) -> int:
        """撤回长期记忆同意并立即使全部条目不可召回。"""
        # Consent 记录是权威授权状态，先原子更新为 revoked。
        self.record_memory_consent(
            tenant_id=tenant_id,
            subject_type="user",
            subject_id=user_id,
            purpose="preference_memory",
            status="revoked",
            policy_version=policy_version,
        )
        # 权威 Consent 撤回后，再将现有条目标记 revoked+disabled，形成双重召回阻断。
        row = self._fetch_one(
            """
            WITH updated AS (
                UPDATE memory_items
                SET consent_status = 'revoked', status = 'disabled', version = version + 1, updated_at = now()
                WHERE tenant_id = :tenant_id AND user_id = :user_id AND consent_status <> 'revoked'
                RETURNING 1
            ) SELECT count(*) AS count FROM updated
            """,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        # 返回本次实际禁用条目数，已撤回条目不会重复计数。
        return int(row["count"])

    def has_memory_consent(
        self,
        *,
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        purpose: str,
    ) -> bool:
        """检查主体对指定记忆用途是否存在有效同意。"""
        # 查询租户、主体类型、主体 ID、用途和 granted 状态的精确组合。
        rows = self._fetch_all(
            """
            SELECT 1 FROM memory_consents
            WHERE tenant_id=:tenant_id AND subject_type=:subject_type
              AND subject_id=:subject_id AND purpose=:purpose AND status='granted'
            LIMIT 1
            """,
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            purpose=purpose,
        )
        # 是否存在至少一行即代表当前有效授权。
        return bool(rows)

    def record_memory_consent(
        self,
        *,
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        purpose: str,
        status: str,
        policy_version: str,
    ) -> None:
        """Upsert 用户或客户的用途级 Consent。"""
        # 只接受 granted/revoked 两种状态，未知值不得进入权限判断。
        if status not in {"granted", "revoked"}:
            # 显式异常避免 SQL CASE 将未知状态误当成撤回时间。
            raise ValueError("consent status 必须是 granted 或 revoked")
        # 以租户、主体、用途为唯一键更新权威授权状态和政策版本。
        self._execute(
            """
            INSERT INTO memory_consents (
                tenant_id, subject_type, subject_id, purpose, status, policy_version,
                granted_at, revoked_at
            ) VALUES (
                :tenant_id, :subject_type, :subject_id, :purpose, :status, :policy_version,
                CASE WHEN :status='granted' THEN now() ELSE NULL END,
                CASE WHEN :status='revoked' THEN now() ELSE NULL END
            ) ON CONFLICT (tenant_id, subject_type, subject_id, purpose) DO UPDATE SET
                status=EXCLUDED.status, policy_version=EXCLUDED.policy_version,
                granted_at=EXCLUDED.granted_at, revoked_at=EXCLUDED.revoked_at, updated_at=now()
            """,
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            purpose=purpose,
            status=status,
            policy_version=policy_version,
        )

    def delete_user_memory(self, *, tenant_id: str, user_id: str) -> int:
        """物理删除用户长期记忆；Embedding 通过外键级联清理。"""
        # CTE 删除精确 tenant/user 条目并返回实际行数，向量依赖外键级联。
        row = self._fetch_one(
            """
            WITH deleted AS (
                DELETE FROM memory_items
                WHERE tenant_id = :tenant_id AND user_id = :user_id
                RETURNING 1
            ) SELECT count(*) AS count FROM deleted
            """,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        # 将数据库 count 转为 API 使用的整数。
        return int(row["count"])

    def delete_subject_memory(self, *, tenant_id: str, subject_id: str) -> int:
        """物理删除通用主体的消息和长期记忆，并依赖外键清理向量。"""
        # 单条 SQL 原子删除 Session 消息和用户长期记忆并汇总计数。
        row = self._fetch_one(
            """
            WITH deleted_messages AS (
                DELETE FROM short_term_messages
                WHERE tenant_id=:tenant_id AND session_id=:subject_id RETURNING 1
            ), deleted_items AS (
                DELETE FROM memory_items
                WHERE tenant_id=:tenant_id AND user_id=:subject_id RETURNING 1
            )
            SELECT
                (SELECT count(*) FROM deleted_messages)
              + (SELECT count(*) FROM deleted_items) AS count
            """,
            tenant_id=tenant_id,
            subject_id=subject_id,
        )
        # 返回消息和长期条目两类记录实际删除总数。
        return int(row["count"])

    def insert_privacy_audit(
        self,
        *,
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        action: str,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        """写入不含原始主体 ID 的隐私操作审计。"""
        # 主体标识只能以 HMAC 形式进入审计，因此强制要求足够长度的 Secret。
        if not self.encryption_key or len(self.encryption_key) < 24:
            # 缺密钥时拒绝写入可反查的明文主体 ID。
            raise RuntimeError("隐私审计 HMAC 需要 MEMORY_ENCRYPTION_KEY")
        # pgcrypto hmac 在数据库内计算不可逆 subject_hash，摘要另存 JSONB。
        self._execute(
            """
            INSERT INTO privacy_audit_events (
                tenant_id, subject_type, subject_hash, action, result_summary
            ) VALUES (
                :tenant_id, :subject_type,
                encode(hmac(:subject_id, :hmac_key, 'sha256'), 'hex'),
                :action, CAST(:result_summary AS jsonb)
            )
            """,
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            hmac_key=self.encryption_key,
            action=action,
            result_summary=_json(result_summary or {}),
        )

    def purge_expired_memory(self, *, tenant_id: str, batch_size: int = 1000) -> int:
        """分批清理过期长期记忆和加密审计，避免大事务锁表。"""
        # 第一批按 expires_at 选择长期记忆牺牲行并物理删除，Embedding 随外键级联。
        memory_row = self._fetch_one(
            """
            WITH victims AS (
                SELECT id FROM memory_items
                WHERE tenant_id = :tenant_id AND expires_at <= now()
                ORDER BY expires_at LIMIT :batch_size
            ), deleted AS (
                DELETE FROM memory_items m USING victims v
                WHERE m.id = v.id RETURNING 1
            ) SELECT count(*) AS count FROM deleted
            """,
            tenant_id=tenant_id,
            batch_size=batch_size,
        )
        # 第二批清理过期短期消息审计密文。
        message_row = self._fetch_one(
            """
            WITH victims AS (
                SELECT id FROM short_term_messages
                WHERE tenant_id=:tenant_id AND expires_at<=now()
                LIMIT :batch_size
            ), deleted AS (
                DELETE FROM short_term_messages message USING victims
                WHERE message.id=victims.id RETURNING 1
            ) SELECT count(*) AS count FROM deleted
            """,
            tenant_id=tenant_id,
            batch_size=max(1, min(batch_size, 10000)),
        )
        # 第三批清理过期通用生成输出密文和脱敏审计副本。
        output_row = self._fetch_one(
            """
            WITH victims AS (
                SELECT id FROM generated_outputs
                WHERE tenant_id=:tenant_id AND expires_at<=now()
                LIMIT :batch_size
            ), deleted AS (
                DELETE FROM generated_outputs output USING victims
                WHERE output.id=victims.id RETURNING 1
            ) SELECT count(*) AS count FROM deleted
            """,
            tenant_id=tenant_id,
            batch_size=max(1, min(batch_size, 10000)),
        )
        # 汇总三张表本批实际删除数量。
        return sum(
            int(row["count"])
            for row in [memory_row, message_row, output_row]
        )

    def insert_rag_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        title: str,
        source_uri: str | None,
        metadata: dict[str, Any],
    ) -> None:
        """写入 RAG 文档元数据。"""
        # 文档主记录按 ID Upsert 标题、来源和 metadata，不在此处写 Chunk 正文。
        self._execute(
            """
            INSERT INTO rag_documents (id, tenant_id, title, source_uri, metadata)
            VALUES (:id, :tenant_id, :title, :source_uri, CAST(:metadata AS jsonb))
            ON CONFLICT (id) DO UPDATE
            SET title = EXCLUDED.title, source_uri = EXCLUDED.source_uri,
                metadata = EXCLUDED.metadata, updated_at = now()
            """,
            id=document_id,
            tenant_id=tenant_id,
            title=title,
            source_uri=source_uri,
            metadata=_json(metadata),
        )

    def insert_rag_chunk(
        self,
        *,
        tenant_id: str,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        content: str,
        embedding: Sequence[float],
        metadata: dict[str, Any],
    ) -> None:
        """写入 RAG chunk 和对应向量。"""
        # 先 Upsert Chunk 正文与 metadata，确保向量外键目标存在。
        self._execute(
            """
            INSERT INTO rag_chunks (id, tenant_id, document_id, chunk_index, content, metadata)
            VALUES (:id, :tenant_id, :document_id, :chunk_index, :content, CAST(:metadata AS jsonb))
            ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, updated_at = now()
            """,
            id=chunk_id,
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            metadata=_json(metadata),
        )
        # 再按 chunk_id Upsert pgvector，正文更新与向量更新保持相同租户参数。
        self._execute(
            """
            INSERT INTO rag_chunk_embeddings (tenant_id, chunk_id, embedding)
            VALUES (:tenant_id, :chunk_id, CAST(:embedding AS halfvec))
            ON CONFLICT (chunk_id) DO UPDATE
            SET embedding = EXCLUDED.embedding, updated_at = now()
            """,
            tenant_id=tenant_id,
            chunk_id=chunk_id,
            embedding=_vector_literal(embedding),
        )

    def search_rag_chunks(
        self,
        *,
        tenant_id: str,
        query: str,
        query_embedding: Sequence[float],
        libraries: list[str] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[PersistedRagHit]:
        """在线 RAG 检索：pgvector + 全文检索 + metadata 过滤。"""
        # top_k 缺省使用检索配置，显式参数可覆盖单次查询数量。
        limit = top_k or self.retrieval.top_k
        # score_threshold 未传时采用全局配置。
        threshold = self.retrieval.score_threshold if score_threshold is None else score_threshold
        # SQL 强制租户、可选 Library 及 JSON boolean 准入过滤，再融合三类得分。
        rows = self._fetch_all(
            """
            WITH scored AS (
                SELECT
                    d.id AS document_id,
                    c.id AS chunk_id,
                    c.content,
                    c.metadata,
                    d.source_uri,
                    GREATEST(0, 1 - (e.embedding <=> CAST(:embedding AS halfvec))) AS vector_score,
                    ts_rank_cd(to_tsvector('simple', c.content), plainto_tsquery('simple', :query)) AS lexical_score,
                    CASE
                        WHEN (:libraries_is_empty OR c.metadata ->> 'library' = ANY(:libraries)) THEN 1.0
                        ELSE 0.0
                    END AS metadata_score
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id
                JOIN rag_chunk_embeddings e ON e.chunk_id = c.id AND e.tenant_id = c.tenant_id
                WHERE c.tenant_id = :tenant_id
                  AND (:libraries_is_empty OR c.metadata ->> 'library' = ANY(:libraries))
                  -- 审批标志必须是 JSON boolean true；缺字段、null、字符串 "true" 和非法值全部拒绝。
                  AND jsonb_typeof(c.metadata -> 'approved_for_generation') = 'boolean'
                  AND c.metadata ->> 'approved_for_generation' = 'true'
            )
            SELECT *,
                (:vector_weight * vector_score
                 + :lexical_weight * lexical_score
                 + :metadata_weight * metadata_score) AS final_score
            FROM scored
            WHERE (:vector_weight * vector_score
                   + :lexical_weight * lexical_score
                   + :metadata_weight * metadata_score) >= :threshold
            ORDER BY final_score DESC
            LIMIT :limit
            """,
            tenant_id=tenant_id,
            query=query,
            embedding=_vector_literal(query_embedding),
            libraries=libraries or ["__none__"],
            libraries_is_empty=not bool(libraries),
            vector_weight=self.retrieval.vector_weight,
            lexical_weight=self.retrieval.lexical_weight,
            metadata_weight=self.retrieval.metadata_weight,
            threshold=threshold,
            limit=limit,
        )
        # 每行通过 Pydantic 契约校验后返回，数据库字段异常不会静默传播。
        return [PersistedRagHit.model_validate(dict(row)) for row in rows]

    def insert_tool_call(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> str:
        """写入工具调用审计记录。"""
        # Tool Call 保存名称和完整结构化 Payload，并返回数据库 ID 关联后续结果。
        row = self._fetch_one(
            """
            INSERT INTO tool_calls (tenant_id, trace_id, tool_name, payload)
            VALUES (:tenant_id, :trace_id, :tool_name, CAST(:payload AS jsonb))
            RETURNING id
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            tool_name=payload.get("name") or payload.get("tool_name") or "unknown",
            payload=_json(payload),
        )
        # 统一将数据库主键转换为字符串供 ToolResult 使用。
        return str(row["id"])

    def insert_tool_result(self, *, tenant_id: str, tool_call_id: str, payload: dict[str, Any]) -> None:
        """写入工具结果，供后续 grounding 和审计回放。"""
        # Tool Result 通过 tool_call_id 关联调用，status 单列便于失败率查询。
        self._execute(
            """
            INSERT INTO tool_results (tenant_id, tool_call_id, status, payload)
            VALUES (:tenant_id, :tool_call_id, :status, CAST(:payload AS jsonb))
            """,
            tenant_id=tenant_id,
            tool_call_id=tool_call_id,
            status=payload.get("status", "unknown"),
            payload=_json(payload),
        )

    def insert_generated_output(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> None:
        """写入最终输出、策略、补问或低压维护消息。"""
        # 通用输出正文必须加密，因此缺少足够长度密钥时 fail-closed。
        if not self.encryption_key or len(self.encryption_key) < 24:
            # 禁止以明文替代密文持久化。
            raise RuntimeError("通用生成输出持久化需要 MEMORY_ENCRYPTION_KEY")
        # 从 Payload 提取正文并转换为字符串，空值统一为空文本。
        output_text = str(payload.get("output_text", ""))
        # 生成低权限可见脱敏副本；原正文仅进入 pgcrypto 参数。
        redacted, _scan = scan_and_redact_output_pii(output_text)
        # JSON Payload 中同步替换为脱敏文本，避免密文旁边又保存一份明文副本。
        # 复制 Payload 避免修改调用方对象。
        safe_payload = dict(payload)
        # 替换副本中的输出正文，防止 JSONB 旁路保存同一明文。
        safe_payload["output_text"] = redacted
        # 同时写入正文密文、脱敏副本、Hash、输入上下文和安全 Payload。
        self._execute(
            """
            INSERT INTO generated_outputs (
                tenant_id, trace_id, output_type, input_context, output_ciphertext,
                output_redacted, output_hash, payload, expires_at
            )
            VALUES (
                :tenant_id, :trace_id, :output_type, CAST(:input_context AS jsonb),
                pgp_sym_encrypt(:output_text, :encryption_key, 'cipher-algo=aes256'),
                :output_redacted, encode(digest(:output_text, 'sha256'), 'hex'),
                CAST(:payload AS jsonb), now() + make_interval(days => :retention_days)
            )
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            output_type=payload.get("output_type", "final_answer"),
            input_context=_json(payload.get("input_context", {})),
            output_text=output_text,
            output_redacted=redacted,
            encryption_key=self.encryption_key,
            retention_days=int(payload.get("retention_days", 365)),
            payload=_json(safe_payload),
        )

    def insert_feedback_event(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> None:
        """写入用户反馈或离线评测事件。"""
        # feedback_type 单列便于聚合，其余上下文保留为 JSONB。
        self._execute(
            """
            INSERT INTO feedback_events (tenant_id, trace_id, feedback_type, payload)
            VALUES (:tenant_id, :trace_id, :feedback_type, CAST(:payload AS jsonb))
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            feedback_type=payload.get("feedback_type", "unknown"),
            payload=_json(payload),
        )

    def _execute(self, sql: str, **params: Any) -> None:
        """在独立事务中设置 RLS 租户上下文并执行无返回 SQL。"""

        # begin 确保 set_config(local=true) 与业务 SQL 处于同一事务连接。
        with self.engine.begin() as connection:
            # 任何租户表访问前强制验证并设置 app.tenant_id。
            # 将 params 中 tenant_id 设置为当前事务 RLS 上下文。
            self._set_tenant_context(connection, params.get("tenant_id"))
            # text + params 使用绑定参数执行，避免把业务值拼入 SQL。
            connection.execute(text(sql), params)

    def _fetch_one(self, sql: str, **params: Any) -> dict[str, Any]:
        """在 RLS 事务中执行 SQL，并要求恰好返回一行 Mapping。"""

        # 单行读取和写入 RETURNING 共享同一租户事务边界。
        with self.engine.begin() as connection:
            # 将 params 中 tenant_id 设置为当前事务 RLS 上下文。
            self._set_tenant_context(connection, params.get("tenant_id"))
            # one() 对零行或多行显式报错，防止调用方误用单行契约。
            row = connection.execute(text(sql), params).mappings().one()
        # 事务结束前已复制 Mapping 所需值，返回普通 dict 隔离数据库 Row 生命周期。
        return dict(row)

    def _fetch_all(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        """在 RLS 事务中执行 SQL，并返回全部 Mapping 的普通字典副本。"""

        # 多行检索同样在事务开始后先设置 tenant 上下文。
        with self.engine.begin() as connection:
            # 将 params 中 tenant_id 设置为当前多行查询事务的 RLS 上下文。
            self._set_tenant_context(connection, params.get("tenant_id"))
            # all() 一次物化结果，连接归还池后返回值仍可安全使用。
            rows = connection.execute(text(sql), params).mappings().all()
        # 转成普通 dict，避免向业务层泄露 SQLAlchemy RowMapping 对象。
        return [dict(row) for row in rows]

    @staticmethod
    def _set_tenant_context(connection: Any, tenant_id: Any) -> None:
        """为当前事务设置 RLS 租户上下文；缺少 tenant_id 时拒绝访问租户表。"""
        # None、空字符串和纯空白都不构成有效租户边界。
        if tenant_id is None or not str(tenant_id).strip():
            # fail-closed 阻止 Repository 方法遗漏 tenant_id 后绕过 RLS 预期。
            raise ValueError("PostgreSQL tenant-scoped operation requires tenant_id")
        # 第三个参数 true 表示事务局部设置，连接回池后不会残留上一个租户。
        connection.execute(
            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
            {"tenant_id": str(tenant_id)},
        )


def _json(value: Any) -> str:
    """把 Python 值稳定序列化为 PostgreSQL JSONB 绑定参数字符串。"""

    # ensure_ascii=False 保留中文，default=str 兼容 datetime 等可字符串化类型。
    return json.dumps(value, ensure_ascii=False, default=str)


def _vector_literal(values: Sequence[float]) -> str:
    """校验统一 Embedding 维度并生成 pgvector 文本字面量绑定值。"""

    # 所有通用记忆和 RAG 表统一使用 halfvec(3072)，运行时禁止模型维度漂移。
    if len(values) != 3072:
        # 维度不匹配时明确报错，避免 pgvector 在 SQL 层返回难定位的类型异常。
        raise ValueError(f"pgvector 需要 3072 维向量，实际收到 {len(values)} 维")
    # 每个元素显式转 float 后拼为 pgvector 接受的 `[v1,v2,...]` 格式。
    return "[" + ",".join(str(float(value)) for value in values) + "]"
