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

    # tenant_id 是知识隔离主键，Repository 的写入和搜索都必须携带同一值。
    tenant_id: str = Field(..., description="文档所属租户，入库和检索都必须使用该字段隔离。")
    # title 为引用和运维排障提供可读名称，不替代服务端 document_id。
    title: str = Field(..., description="文档标题，用于前端 citation 和排障。")
    # content 是待清洗、分块和向量化的原始正文，不能未经脱敏直接持久化为 chunk。
    content: str = Field(..., description="文档正文。入库前会做分块和 PII 过滤。")
    # source_uri 只记录来源定位信息，缺失时允许为空且不影响租户隔离。
    source_uri: str | None = Field(default=None, description="原始来源 URI，例如文件路径、网页 URL 或对象存储地址。")
    # metadata 承载知识库、版本和准入标签，安全关键键会在管道中受控覆盖。
    metadata: dict[str, Any] = Field(default_factory=dict, description="业务 metadata，例如 library、版本、权限标签。")


class RagIngestionResult(BaseModel):
    """一次 RAG 入库结果。"""

    # document_id 返回服务端生成的父文档主键，调用方可用于追踪或撤销入库。
    document_id: str = Field(..., description="写入 rag_documents 的文档 ID。")
    # chunk_ids 按实际写入顺序返回所有分块主键，便于核对 Embedding 与引用。
    chunk_ids: list[str] = Field(default_factory=list, description="写入 rag_chunks 的 chunk ID 列表。")


class QueryRewriteOutput(BaseModel):
    """在线检索前的模型改写结果。"""

    # queries 至少包含一条可执行检索词，由结构化模型输出约束而非自由文本解析。
    queries: list[str] = Field(..., min_length=1, description="用于检索的 query rewrite 列表。")
    # filters 保存模型建议的 metadata 条件，真正权限过滤仍由 Repository 强制执行。
    filters: dict[str, Any] = Field(default_factory=dict, description="模型建议的 metadata filters。")
    # reason 只用于 trace 解释改写依据，不参与最终回答事实生成。
    reason: str = Field(default="", description="为什么这样改写和过滤，用于 trace。")


class RagRetrievalResult(BaseModel):
    """在线 RAG 召回结果。"""

    # rewritten_queries 记录实际用于召回的改写词，支持检索排障和离线评测。
    rewritten_queries: list[str] = Field(default_factory=list, description="实际执行的检索 query。")
    # hits 保存权限过滤、融合去重和 rerank 后的证据对象。
    hits: list[PersistedRagHit] = Field(default_factory=list, description="rerank 后的证据片段。")
    # citations 投影前端和 Grounding 所需的最小来源信息，不暴露内部全文。
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
        """注入持久化与向量客户端，并保存可配置分块窗口。"""
        # Repository 负责租户隔离 SQL 与 pgvector 写入，本管道不直接拼接 SQL。
        self.repository = repository
        # Embedding Client 在应用生命周期内复用模型配置和 HTTP 连接。
        self.embedding_client = embedding_client
        # 分块大小与重叠由构造参数注入，避免散落在入库业务代码中。
        self.chunk_size = chunk_size
        # 保存窗口重叠长度，使跨边界语义在相邻 chunk 中得到保留。
        self.chunk_overlap = chunk_overlap

    def ingest_text(self, document: RagDocumentInput) -> RagIngestionResult:
        """把一份正文入库到 PostgreSQL / pgvector。"""
        # tenant_id 为空时立即拒绝，禁止生成无租户归属的知识记录。
        if not document.tenant_id:
            # 抛出明确契约错误，调用方需要修复入库请求而不是重试数据库。
            raise ValueError("RAG 入库必须带 tenant_id")
        # 入库前先脱敏常见 PII，原始正文不能直接进入 chunk 或 Embedding。
        cleaned_content = _redact_pii(document.content)
        # 使用配置化窗口切分正文，chunk 顺序随后写入 chunk_index。
        chunks = chunk_text(cleaned_content, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        # 空正文或清洗后无 chunk 时拒绝创建空文档。
        if not chunks:
            # 空文档属于输入错误，不写 rag_documents 孤儿记录。
            raise ValueError("RAG 入库文档正文为空，无法生成 chunk")

        # 为本次文档生成服务端 ID，外部 metadata 无权覆盖主键。
        document_id = new_id("rag_doc")
        # 安全关键 metadata 使用受控归一化值覆盖调用方同名字段。
        metadata = {
            # 先复制扩展字段，再用下方受控值覆盖安全关键键，防止字符串或缺省值绕过归一化。
            **document.metadata,
            "library": document.metadata.get("library", "generic"),
            "permission_label": document.metadata.get("permission_label", "tenant"),
            # 入库不等于审批；只有 literal True 才允许该批 chunk 进入在线生成检索。
            "approved_for_generation": document.metadata.get("approved_for_generation", False) is True,
            "index_version": document.metadata.get("index_version", "v1"),
        }
        # 先写父文档，再写带外键的 chunk；Repository 负责实际事务边界。
        self.repository.insert_rag_document(
            tenant_id=document.tenant_id,
            document_id=document_id,
            title=document.title,
            source_uri=document.source_uri,
            metadata=metadata,
        )

        # 一次批量生成所有 chunk 向量，输出数量由 strict zip 再次校验。
        embeddings = self.embedding_client.embed(chunks)
        # 保存真实写入的 chunk ID，作为入库结果返回给调用方。
        chunk_ids: list[str] = []
        # strict=True 确保 Embedding 数量与 chunk 数量不一致时立即失败。
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            # 每个 chunk 使用独立服务端 ID，不能用数组下标作为跨文档主键。
            chunk_id = new_id("rag_chunk")
            # 按写入顺序保存主键，最终结果可以完整反映成功入库的分块。
            chunk_ids.append(chunk_id)
            # 将正文、向量、受控 metadata 与父文档关系交给 Repository 持久化。
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
        # 所有分块写入完成后返回父文档及 chunk 主键，不返回清洗前正文。
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
        """注入在线检索所需的 Repository、改写、Embedding 与重排客户端。"""
        # Repository 执行租户过滤和数据库混合检索。
        self.repository = repository
        # Query Rewrite Client 只输出结构化 queries/filters，不回答用户问题。
        self.query_rewrite_client = query_rewrite_client
        # 查询 Embedding 必须与离线入库模型和维度一致。
        self.embedding_client = embedding_client
        # Reranker 只重排已通过数据库权限/生成准入过滤的候选。
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
        # 用 Pydantic Schema 约束模型改写，避免自由文本直接成为检索控制结构。
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
        # 对所有改写 Query 批量生成向量，和离线入库保持同一客户端。
        query_embeddings = self.embedding_client.embed(rewrite.queries)
        # merged 按 chunk_id 去重，多 Query 命中只保留最高融合分。
        merged: dict[str, PersistedRagHit] = {}
        # 每个 Query/Embedding 对独立执行租户过滤检索。
        for query, embedding in zip(rewrite.queries, query_embeddings, strict=True):
            # 数据库先扩大候选集到 2*TopK，为 Reranker 保留召回空间。
            hits = self.repository.search_rag_chunks(
                tenant_id=tenant_id,
                query=query,
                query_embedding=embedding,
                libraries=libraries,
                top_k=max(top_k * 2, top_k),
            )
            # 逐条合并同一 chunk 的不同 Query 命中结果。
            for hit in hits:
                # 读取当前 chunk 已保存的最好结果。
                old = merged.get(hit.chunk_id)
                # 首次命中或新融合分更高时更新。
                if old is None or hit.final_score > old.final_score:
                    # 以 chunk_id 为键仅保留最高分命中，避免多条改写 query 重复占用上下文。
                    merged[hit.chunk_id] = hit

        # 先按数据库融合分稳定排序，再交给模型 Reranker。
        candidates = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)
        # 有候选时调用 Reranker；空候选不发起无意义模型请求。
        if candidates:
            # 基于原始用户问题对已过滤候选重新排序，避免改写词改变最终相关性目标。
            ranking = self.reranker_client.rerank(
                query=user_query,
                documents=[hit.content for hit in candidates],
                top_k=top_k,
            )
            # 将合法索引映射回候选对象，防御外部重排服务返回越界位置。
            ranked_hits = [candidates[item.index] for item in ranking if item.index < len(candidates)]
        # 完全没有候选时跳过外部 Reranker，直接使用空候选列表。
        else:
            # 空候选直接保持空列表并返回空 citations。
            ranked_hits = candidates

        # 从最终 TopK 命中投影 citation，正文仍保留在 hits 供 Grounding 使用。
        citations = [
            {
                "document_id": hit.document_id,
                "chunk_id": hit.chunk_id,
                "source_uri": hit.source_uri,
                "score": hit.final_score,
            }
            for hit in ranked_hits[:top_k]
        ]
        # 返回检索改写、证据和引用三部分，供 trace、生成与 Grounding 分别消费。
        return RagRetrievalResult(
            rewritten_queries=rewrite.queries,
            hits=ranked_hits[:top_k],
            citations=citations,
        )


# 入库 PII 模式覆盖中国手机号、常见邮箱和身份证号，命中内容统一替换后再向量化。
PII_PATTERNS = [
    re.compile(r"1[3-9]\d{9}"),
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    re.compile(r"\b\d{17}[\dXx]\b"),
]


def _redact_pii(text: str) -> str:
    """入库前清理常见 PII，避免把手机号、邮箱、身份证号写进生成上下文。"""
    # 从原始文本副本开始依次替换，调用方传入字符串不被原地修改。
    redacted = text
    # 每类 PII 使用统一占位符，避免 Hash/Embedding 保留原值特征。
    for pattern in PII_PATTERNS:
        # 替换当前类别的全部命中，保留非敏感上下文和文档结构。
        redacted = pattern.sub("[已脱敏]", redacted)
    # 返回仅用于分块与向量化的脱敏副本，原输入字符串保持不变。
    return redacted
