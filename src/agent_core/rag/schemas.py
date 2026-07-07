"""RAG contracts for query rewrite, hybrid search, metadata, and reranking."""

# 文件说明：
# - 本文件属于 RAG 检索层，负责 query rewrite、metadata、hybrid search、rerank 或 evidence。
# - 检索内容只能作为证据，不能覆盖系统规则。
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A rewritten retrieval query with an explicit purpose."""

    # text 是真正送入检索器的 query，可以来自用户原文，也可以来自 query rewrite。
    text: str = Field(..., description="实际用于检索的 query 文本，可能是用户原文，也可能是 query rewrite 结果。")
    # purpose 说明这条 query 的生成目的，便于多 query 召回后分析哪类 query 起作用。
    purpose: Literal["original", "sales_pain", "customer_type", "scene", "strategy"] = Field(
        default="original",
        description="该 query 的检索目的，用于区分原始问题、销售痛点、客户类型、场景或策略检索。",
    )
    # weight 控制这条 query 在融合排序中的影响力。
    weight: float = Field(
        default=1.0,
        description="该 query 在多 query 检索合并时的权重。越高表示对最终排序影响越大。",
    )


class DocumentMetadata(BaseModel):
    """Metadata used for filtering, ranking, and traceability."""

    # source_id 用来追溯原始资料，例如销售访谈、制度文件或网页缓存。
    source_id: str = Field(..., description="原始文档、访谈或知识条目的来源 ID，用于追溯证据。")
    # chunk_id 用来定位原始资料中的具体片段。
    chunk_id: str = Field(..., description="当前检索片段 ID。一个 source_id 可以拆成多个 chunk。")
    # library 区分资料所属知识库，避免销售经验、产品条款、网页缓存混检。
    library: str = Field(
        default="generic",
        description="知识库或资料库名称，例如 sales_insights、insurance_docs、web_cache。",
    )
    # tenant_id 是多租户隔离字段，检索时必须尊重它。
    tenant_id: str = Field(
        default="local",
        description="片段所属租户。检索时必须和请求 tenant_id 对齐，避免跨租户泄露。",
    )
    # tags 用于场景过滤和 rerank，例如企业主、破冰、异议处理。
    tags: list[str] = Field(
        default_factory=list,
        description="片段标签，例如 破冰、企业主、异议处理，用于过滤、召回增强和 rerank。",
    )
    # risk_level 控制证据进入生成链路的风险上限。
    risk_level: Literal["low", "medium", "high"] = Field(
        default="low",
        description="片段风险等级。高风险内容可用于内部分析，但默认不应直接进入生成。",
    )
    # approved_for_generation 决定该 chunk 是否允许作为最终回答证据。
    approved_for_generation: bool = Field(
        default=True,
        description="该片段是否允许作为最终生成证据。未审批内容应只用于分析或评估，不直接给用户。",
    )
    # extra 保存暂未结构化的来源信息，例如发布时间、URL 或业务阶段。
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展 metadata，例如作者、发布时间、渠道、业务阶段或外部 URL。",
    )


class RetrievalDocument(BaseModel):
    """Searchable document with explicit metadata."""

    # text 是参与检索和压缩的正文片段。
    text: str = Field(..., description="可检索的正文片段。进入生成前可被 evidence compressor 进一步压缩。")
    # metadata 保存过滤、排序、风险和溯源信息。
    metadata: DocumentMetadata = Field(..., description="该正文片段对应的过滤、排序和溯源 metadata。")


class RetrievalResult(BaseModel):
    """Scored retrieval result returned by hybrid search and reranking."""

    # document 保存命中的正文与 metadata。
    document: RetrievalDocument = Field(..., description="命中的文档片段及其 metadata。")
    # lexical_score 是关键词/词法召回分。
    lexical_score: float = Field(default=0.0, description="关键词/BM25 等词法检索得分。")
    # vector_score 是语义相似分。
    vector_score: float = Field(default=0.0, description="向量或语义相似度得分。")
    # metadata_score 是业务约束匹配分。
    metadata_score: float = Field(default=0.0, description="metadata 匹配加权得分，例如标签、租户、风险等级。")
    # rerank_score 保存重排后的分数。
    rerank_score: float = Field(default=0.0, description="reranker 对候选结果重新排序后的得分。")
    # score 是最终排序分，调用方按它选择 TopK。
    score: float = Field(default=0.0, description="融合后的最终排序得分。调用方按该字段选择证据。")


class MetadataFilter(BaseModel):
    """Metadata filter for tenant, libraries, risk, and tags."""

    # tenant_id 限制只能检索当前租户资料。
    tenant_id: str | None = Field(
        default=None,
        description="检索允许访问的租户 ID。为空表示本地演示不限制租户。",
    )
    # libraries 限定可检索知识库集合。
    libraries: list[str] = Field(
        default_factory=list,
        description="允许检索的知识库名称列表。为空表示不按 library 过滤。",
    )
    # required_tags 要求返回 chunk 必须包含这些标签。
    required_tags: list[str] = Field(
        default_factory=list,
        description="必须命中的标签列表。用于把检索限定在破冰、KYC、异议处理等业务场景。",
    )
    # max_risk_level 控制可进入候选集的最高风险等级。
    max_risk_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="允许返回的最高风险等级。生产生成链路建议默认不返回 high。",
    )
    # approved_only 为 True 时，只返回已经审批可用于生成的片段。
    approved_only: bool = Field(
        default=True,
        description="是否只返回 approved_for_generation=True 的片段，避免未审查资料进入最终回答。",
    )
