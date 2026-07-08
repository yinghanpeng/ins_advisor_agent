"""生产 RAG 入库与在线召回。

RAG 被拆成两条链路：
1. 离线入库：解析文档、分块、补 metadata、权限标记、PII/合规过滤、Embedding、写入 pgvector；
2. 在线召回：模型 query rewrite、Embedding、pgvector hybrid search、Rerank、保留 citation。

生产链路禁止从代码里塞内置资料；所有知识都必须来自入库文档和 PostgreSQL / pgvector。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from agent_core.models.client import (
    OpenAICompatibleChatClient,
    OpenAICompatibleEmbeddingClient,
    RerankerClient,
)
from agent_core.persistence.postgres import PersistedRagHit, PostgresAgentRepository
from agent_core.rag.chunking import chunk_text
from agent_core.utils.ids import new_id


class RagDocumentInput(BaseModel):
    """待入库文档。"""

    tenant_id: str = Field(..., description="文档所属租户，入库和检索都必须使用该字段隔离。")
    title: str = Field(..., description="文档标题，用于前端 citation 和排障。")
    content: str = Field(..., description="文档正文。入库前会做分块和 PII 过滤。")
    source_uri: str | None = Field(default=None, description="原始来源 URI，例如文件路径、网页 URL 或对象存储地址。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="业务 metadata，例如 library、版本、权限标签。")


class RagIngestionResult(BaseModel):
    """一次 RAG 入库结果。"""

    document_id: str = Field(..., description="写入 rag_documents 的文档 ID。")
    chunk_ids: list[str] = Field(default_factory=list, description="写入 rag_chunks 的 chunk ID 列表。")


class QueryRewriteOutput(BaseModel):
    """在线检索前的模型改写结果。"""

    queries: list[str] = Field(..., min_length=1, description="用于检索的 query rewrite 列表。")
    filters: dict[str, Any] = Field(default_factory=dict, description="模型建议的 metadata filters。")
    reason: str = Field(default="", description="为什么这样改写和过滤，用于 trace。")


class RagRetrievalResult(BaseModel):
    """在线 RAG 召回结果。"""

    rewritten_queries: list[str] = Field(default_factory=list, description="实际执行的检索 query。")
    hits: list[PersistedRagHit] = Field(default_factory=list, description="rerank 后的证据片段。")
    citations: list[dict[str, Any]] = Field(default_factory=list, description="前端和 grounding 可用的引用信息。")


class RagIngestionPipeline:
    """离线 RAG 入库管道。"""

    def __init__(
        self,
        *,
        repository: PostgresAgentRepository,
        embedding_client: OpenAICompatibleEmbeddingClient,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.repository = repository
        self.embedding_client = embedding_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_text(self, document: RagDocumentInput) -> RagIngestionResult:
        """把一份正文入库到 PostgreSQL / pgvector。"""
        if not document.tenant_id:
            raise ValueError("RAG 入库必须带 tenant_id")
        cleaned_content = _redact_pii(document.content)
        chunks = chunk_text(cleaned_content, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            raise ValueError("RAG 入库文档正文为空，无法生成 chunk")

        document_id = new_id("rag_doc")
        metadata = {
            "library": document.metadata.get("library", "generic"),
            "permission_label": document.metadata.get("permission_label", "tenant"),
            "approved_for_generation": bool(document.metadata.get("approved_for_generation", True)),
            "index_version": document.metadata.get("index_version", "v1"),
            **document.metadata,
        }
        self.repository.insert_rag_document(
            tenant_id=document.tenant_id,
            document_id=document_id,
            title=document.title,
            source_uri=document.source_uri,
            metadata=metadata,
        )

        embeddings = self.embedding_client.embed(chunks)
        chunk_ids: list[str] = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = new_id("rag_chunk")
            chunk_ids.append(chunk_id)
            self.repository.insert_rag_chunk(
                tenant_id=document.tenant_id,
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
                metadata={
                    **metadata,
                    "chunk_index": index,
                    "contains_redacted_pii": cleaned_content != document.content,
                },
            )
        return RagIngestionResult(document_id=document_id, chunk_ids=chunk_ids)


class ProductionRagRetriever:
    """在线 RAG 检索器。"""

    def __init__(
        self,
        *,
        repository: PostgresAgentRepository,
        query_rewrite_client: OpenAICompatibleChatClient,
        embedding_client: OpenAICompatibleEmbeddingClient,
        reranker_client: RerankerClient,
    ) -> None:
        self.repository = repository
        self.query_rewrite_client = query_rewrite_client
        self.embedding_client = embedding_client
        self.reranker_client = reranker_client

    def retrieve(
        self,
        *,
        tenant_id: str,
        user_query: str,
        libraries: list[str] | None = None,
        top_k: int = 8,
    ) -> RagRetrievalResult:
        """执行 query rewrite、embedding、pgvector 检索和 rerank。"""
        rewrite, _model_result = self.query_rewrite_client.complete_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是企业知识库检索 query rewrite 节点。"
                        "只输出 JSON：queries、filters、reason。不要回答用户问题。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"用户问题：{user_query}\n可用知识库：{libraries or ['generic']}",
                },
            ],
            schema_model=QueryRewriteOutput,
        )
        query_embeddings = self.embedding_client.embed(rewrite.queries)
        merged: dict[str, PersistedRagHit] = {}
        for query, embedding in zip(rewrite.queries, query_embeddings, strict=True):
            hits = self.repository.search_rag_chunks(
                tenant_id=tenant_id,
                query=query,
                query_embedding=embedding,
                libraries=libraries,
                top_k=max(top_k * 2, top_k),
            )
            for hit in hits:
                old = merged.get(hit.chunk_id)
                if old is None or hit.final_score > old.final_score:
                    merged[hit.chunk_id] = hit

        candidates = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)
        if candidates:
            ranking = self.reranker_client.rerank(
                query=user_query,
                documents=[hit.content for hit in candidates],
                top_k=top_k,
            )
            ranked_hits = [candidates[item.index] for item in ranking if item.index < len(candidates)]
        else:
            ranked_hits = candidates

        citations = [
            {
                "document_id": hit.document_id,
                "chunk_id": hit.chunk_id,
                "source_uri": hit.source_uri,
                "score": hit.final_score,
            }
            for hit in ranked_hits[:top_k]
        ]
        return RagRetrievalResult(
            rewritten_queries=rewrite.queries,
            hits=ranked_hits[:top_k],
            citations=citations,
        )


PII_PATTERNS = [
    re.compile(r"1[3-9]\d{9}"),
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    re.compile(r"\b\d{17}[\dXx]\b"),
]


def _redact_pii(text: str) -> str:
    """入库前清理常见 PII，避免把手机号、邮箱、身份证号写进生成上下文。"""
    redacted = text
    for pattern in PII_PATTERNS:
        redacted = pattern.sub("[已脱敏]", redacted)
    return redacted
