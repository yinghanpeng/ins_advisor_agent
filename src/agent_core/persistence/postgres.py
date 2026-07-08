"""PostgreSQL / pgvector repository.

本模块是生产数据平面的入口。所有方法都要求传入 tenant_id，并且查询条件中显式使用
tenant_id，这是多租户隔离最重要的一条边界。状态迁移、trace、工具调用、RAG 和长期记忆
都写入 PostgreSQL，AgentState 因此可以被审计、回放和做离线评测。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from agent_core.config.runtime import DatabaseConfig, RetrievalConfig


class PersistedMemoryHit(BaseModel):
    """从 pgvector 长期记忆表召回的一条结果。"""

    id: str = Field(..., description="长期记忆条目 ID。")
    scope: str = Field(..., description="记忆作用域，例如 preference、customer_profile。")
    memory_type: str = Field(..., description="记忆类型，例如 fact、event、case_state。")
    content: str = Field(..., description="进入上下文前的记忆摘要。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="记忆 metadata。")
    vector_score: float = Field(default=0.0, description="pgvector 相似度分。")
    lexical_score: float = Field(default=0.0, description="PostgreSQL 全文检索分。")
    metadata_score: float = Field(default=0.0, description="scope、risk、consent 等 metadata 匹配分。")
    recency_score: float = Field(default=0.0, description="按更新时间计算的新近度分。")
    confidence_score: float = Field(default=0.0, description="长期记忆自身置信度分。")
    final_score: float = Field(default=0.0, description="按配置权重融合后的最终分。")


class PersistedRagHit(BaseModel):
    """从 pgvector RAG chunk 表召回的一条结果。"""

    document_id: str = Field(..., description="命中文档 ID。")
    chunk_id: str = Field(..., description="命中 chunk ID。")
    content: str = Field(..., description="chunk 正文。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="chunk metadata。")
    source_uri: str | None = Field(default=None, description="原始来源 URI。")
    vector_score: float = Field(default=0.0, description="pgvector 相似度分。")
    lexical_score: float = Field(default=0.0, description="PostgreSQL 全文检索分。")
    metadata_score: float = Field(default=0.0, description="metadata 过滤与匹配分。")
    final_score: float = Field(default=0.0, description="融合后的最终检索分。")


class PostgresAgentRepository:
    """生产 Agent Repository。

    这里不包含业务推理逻辑，只负责真实持久化和 pgvector 检索。业务节点负责决定是否写、
    是否召回、如何压缩；Repository 负责保证写入的记录带 tenant_id、trace_id 等审计字段。
    """

    def __init__(self, database: DatabaseConfig, retrieval: RetrievalConfig | None = None) -> None:
        if not database.database_url:
            raise RuntimeError("DATABASE_URL 不能为空，生产持久化需要真实 PostgreSQL")
        self.database = database
        self.retrieval = retrieval or RetrievalConfig()
        self.engine: Engine = create_engine(
            database.database_url,
            pool_size=database.pool_size,
            echo=database.echo_sql,
            future=True,
        )

    def insert_agent_run(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> None:
        """写入一次 Agent run 的入口记录。"""
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
        content: str,
        embedding: Sequence[float],
        source_type: str,
        source_id: str,
        evidence_text: str,
        confidence: float,
        risk_level: str = "low",
        consent_status: str = "granted",
        normalized_content: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> str:
        """写入或更新长期记忆条目。

        长期记忆必须有 evidence_text 和 source_type。没有证据的模型推断不能落成 confirmed
        记忆，否则后续请求会把推测当事实召回。
        """
        if not tenant_id or not user_id:
            raise ValueError("长期记忆写入必须带 tenant_id 和 user_id")
        if not source_type or not evidence_text:
            raise ValueError("长期记忆写入必须带 source_type 和 evidence_text")
        row = self._fetch_one(
            """
            INSERT INTO long_term_memory_items (
                tenant_id, user_id, scope, memory_type, content, normalized_content, embedding,
                source_type, source_id, evidence_text, confidence, status, risk_level,
                consent_status, expires_at, metadata
            )
            VALUES (
                :tenant_id, :user_id, :scope, :memory_type, :content, :normalized_content,
                CAST(:embedding AS vector), :source_type, :source_id, :evidence_text,
                :confidence, 'active', :risk_level, :consent_status, :expires_at,
                CAST(:metadata AS jsonb)
            )
            RETURNING id
            """,
            tenant_id=tenant_id,
            user_id=user_id,
            scope=scope,
            memory_type=memory_type,
            content=content,
            normalized_content=normalized_content or content,
            embedding=_vector_literal(embedding),
            source_type=source_type,
            source_id=source_id,
            evidence_text=evidence_text,
            confidence=confidence,
            risk_level=risk_level,
            consent_status=consent_status,
            expires_at=expires_at,
            metadata=_json(metadata or {}),
        )
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
        if not scopes:
            raise ValueError("长期记忆检索 scopes 不能为空")
        limit = top_k or self.retrieval.top_k
        threshold = self.retrieval.score_threshold if score_threshold is None else score_threshold
        rows = self._fetch_all(
            """
            WITH scored AS (
                SELECT
                    id, scope, memory_type, content, metadata,
                    GREATEST(0, 1 - (embedding <=> CAST(:embedding AS vector))) AS vector_score,
                    ts_rank_cd(
                        to_tsvector('simple', coalesce(content, '') || ' ' || coalesce(normalized_content, '')),
                        plainto_tsquery('simple', :query)
                    ) AS lexical_score,
                    CASE
                        WHEN consent_status = 'granted' AND status = 'active' THEN 1.0
                        ELSE 0.0
                    END AS metadata_score,
                    GREATEST(0, 1 - EXTRACT(EPOCH FROM (now() - updated_at)) / 2592000.0) AS recency_score,
                    confidence AS confidence_score
                FROM long_term_memory_items
                WHERE tenant_id = :tenant_id
                  AND user_id = :user_id
                  AND scope = ANY(:scopes)
                  AND status = 'active'
                  AND deleted_at IS NULL
                  AND consent_status = 'granted'
                  AND risk_level <= :max_risk_level
                  AND (expires_at IS NULL OR expires_at > now())
                  AND (:case_id IS NULL OR metadata ->> 'case_id' = :case_id)
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
            max_risk_level=max_risk_level,
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
        return [PersistedMemoryHit.model_validate(dict(row)) for row in rows]

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
        self._execute(
            """
            INSERT INTO rag_chunk_embeddings (tenant_id, chunk_id, embedding)
            VALUES (:tenant_id, :chunk_id, CAST(:embedding AS vector))
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
        limit = top_k or self.retrieval.top_k
        threshold = self.retrieval.score_threshold if score_threshold is None else score_threshold
        rows = self._fetch_all(
            """
            WITH scored AS (
                SELECT
                    d.id AS document_id,
                    c.id AS chunk_id,
                    c.content,
                    c.metadata,
                    d.source_uri,
                    GREATEST(0, 1 - (e.embedding <=> CAST(:embedding AS vector))) AS vector_score,
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
                  AND COALESCE((c.metadata ->> 'approved_for_generation')::boolean, true) = true
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
        return [PersistedRagHit.model_validate(dict(row)) for row in rows]

    def insert_tool_call(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> str:
        """写入工具调用审计记录。"""
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
        return str(row["id"])

    def insert_tool_result(self, *, tenant_id: str, tool_call_id: str, payload: dict[str, Any]) -> None:
        """写入工具结果，供后续 grounding 和审计回放。"""
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

    def insert_human_approval_request(self, *, tenant_id: str, payload: dict[str, Any]) -> None:
        """持久化人工审批请求和 checkpoint 引用。"""
        self._execute(
            """
            INSERT INTO human_approval_requests (
                id, tenant_id, trace_id, checkpoint_id, pending_action, risk_reason,
                approval_payload, required_approver_role, status, expires_at
            )
            VALUES (
                :id, :tenant_id, :trace_id, :checkpoint_id, :pending_action, :risk_reason,
                CAST(:approval_payload AS jsonb), :required_approver_role, :status, :expires_at
            )
            ON CONFLICT (id) DO UPDATE
            SET status = EXCLUDED.status, approval_payload = EXCLUDED.approval_payload, updated_at = now()
            """,
            id=payload["approval_id"],
            tenant_id=tenant_id,
            trace_id=payload["trace_id"],
            checkpoint_id=payload["checkpoint_id"],
            pending_action=payload["pending_action"],
            risk_reason=payload["risk_reason"],
            approval_payload=_json(payload.get("approval_payload", {})),
            required_approver_role=payload.get("required_approver_role", "advisor"),
            status=payload.get("status", "pending"),
            expires_at=payload.get("expires_at"),
        )

    def insert_generated_output(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> None:
        """写入最终输出、策略、补问或低压维护消息。"""
        self._execute(
            """
            INSERT INTO generated_outputs (tenant_id, trace_id, output_type, input_context, output_text, payload)
            VALUES (
                :tenant_id, :trace_id, :output_type, CAST(:input_context AS jsonb),
                :output_text, CAST(:payload AS jsonb)
            )
            """,
            tenant_id=tenant_id,
            trace_id=trace_id,
            output_type=payload.get("output_type", "final_answer"),
            input_context=_json(payload.get("input_context", {})),
            output_text=payload.get("output_text", ""),
            payload=_json(payload),
        )

    def insert_feedback_event(self, *, tenant_id: str, trace_id: str, payload: dict[str, Any]) -> None:
        """写入用户反馈或离线评测事件。"""
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
        with self.engine.begin() as connection:
            connection.execute(text(sql), params)

    def _fetch_one(self, sql: str, **params: Any) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(text(sql), params).mappings().one()
        return dict(row)

    def _fetch_all(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(text(sql), params).mappings().all()
        return [dict(row) for row in rows]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _vector_literal(values: Sequence[float]) -> str:
    if not values:
        raise ValueError("pgvector 写入或检索需要非空向量")
    return "[" + ",".join(str(float(value)) for value in values) + "]"
