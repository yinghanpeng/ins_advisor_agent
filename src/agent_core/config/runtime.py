"""集中运行配置加载。

生产级 Agent 不能把模型名、URL、数据库连接和检索权重散落在业务节点里。
本模块负责从 `configs/*.yaml` 读取配置，并把 `${ENV_NAME}` 占位符替换为环境变量。

设计意图：
1. 业务代码只依赖结构化 RuntimeSettings；
2. 缺少生产必需配置时 fail-fast，而不是回退到本地替代数据；
3. 测试可以构造 RuntimeSettings fixture，但生产路径必须来自配置和环境变量。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# 预编译 ${ENV_NAME} 占位符规则；仅接受大写字母、数字和下划线的环境变量名。
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
# OpenAI-compatible 网关在不同团队常使用 OPENAI_* 或 LLM_* 命名；显式 LLM_* 始终优先。
ENV_FALLBACK_ALIASES = {
    "LLM_BASE_URL": "OPENAI_BASE_URL",
    "LLM_API_KEY": "OPENAI_API_KEY",
}


class _StrictConfigModel(BaseModel):
    """所有运行配置的严格基类，拒绝拼错或已经废弃的配置字段。"""

    # extra="forbid" 让配置拼写错误在启动阶段暴露，避免 Pydantic 静默丢弃未知键。
    model_config = ConfigDict(extra="forbid")


class ModelEndpointConfig(_StrictConfigModel):
    """一个模型端点配置。"""

    # provider 决定使用哪一种模型客户端适配协议，目前默认兼容 OpenAI HTTP 契约。
    provider: str = Field(default="openai_compatible", description="模型供应商类型，例如 openai_compatible。")
    # model 是供应商端真实模型标识，生产通常由环境变量插值填入。
    model: str | None = Field(default="", description="模型名称，由环境变量或配置文件提供。")
    # base_url 保存模型网关根地址，使业务节点不需要硬编码服务地址。
    base_url: str | None = Field(default="", description="模型服务 Base URL，不允许硬编码在业务代码中。")
    # api_key 只从运行配置读取，调用方不得通过业务 metadata 临时覆盖凭据。
    api_key: str | None = Field(default="", description="模型服务 API Key，从环境变量加载。")
    # timeout_ms 限制单次模型请求等待时间，避免一个下游请求长期占用 Agent 执行线程。
    timeout_ms: int = Field(default=15000, description="模型请求超时时间，单位毫秒。")
    # max_retries 控制客户端对瞬态网络错误的最多重试次数。
    max_retries: int = Field(default=2, description="模型请求最大重试次数。")
    # dimensions 仅约束 Embedding 输出维度，普通生成模型保持 None。
    dimensions: int | None = Field(default=None, description="Embedding 维度；非 embedding 模型为空。")


class DatabaseConfig(_StrictConfigModel):
    """数据库和 Redis 配置。"""

    # database_url 是 PostgreSQL/pgvector 主连接串，生产由 Secret 环境变量插值提供。
    database_url: str | None = Field(default="", description="PostgreSQL 连接字符串。")
    # redis_url 是短期记忆和分布式限流共用的 Redis 连接串。
    redis_url: str | None = Field(default=None, description="Redis 连接字符串，用于限流、队列或缓存。")
    # redis_sentinel_master 非空时改由 Sentinel 发现当前 Redis Master，不再读取 redis_url。
    redis_sentinel_master: str | None = Field(
        default=None,
        description="Redis Sentinel 主库名称；非空时启用 Sentinel 连接模式。",
    )
    # redis_sentinel_nodes 保存逗号分隔的 host:port 列表，运行时会拆分并校验每个节点。
    redis_sentinel_nodes: str | None = Field(
        default=None,
        description="Redis Sentinel 节点列表，格式为 host:port,host:port。",
    )
    # redis_password 是发现 Master 后连接实际 Redis 数据节点使用的密码；允许本地无密码实例为空。
    redis_password: str | None = Field(
        default=None,
        description="Redis Master 认证密码；空值表示不发送 AUTH。",
    )
    # redis_sentinel_password 单独用于 Sentinel 节点认证，和 Redis Master 密码可以不同。
    redis_sentinel_password: str | None = Field(
        default=None,
        description="Redis Sentinel 节点认证密码；空值表示 Sentinel 不需要 AUTH。",
    )
    # redis_database 以字符串接收环境变量，连接 Runtime 会严格转换为非负整数并默认使用 0。
    redis_database: str | None = Field(
        default=None,
        description="Sentinel 模式连接 Redis Master 使用的逻辑数据库编号。",
    )
    # pool_size 定义 SQLAlchemy 长驻连接数量。
    pool_size: int = Field(default=5, description="数据库连接池大小。")
    # max_overflow 允许流量突发时临时增加连接，但不允许配置负值。
    max_overflow: int = Field(default=10, ge=0, description="PostgreSQL 连接池突发连接上限。")
    # pool_timeout_seconds 限制等待空闲数据库连接的最长时间。
    pool_timeout_seconds: int = Field(default=10, ge=1, description="获取数据库连接的最长等待秒数。")
    # pool_recycle_seconds 主动回收长寿命连接，降低网络设备提前断开造成的陈旧连接风险。
    pool_recycle_seconds: int = Field(default=1800, ge=60, description="数据库连接回收周期。")
    # pool_pre_ping 在借出连接前验证存活，避免把断开的 Socket 交给业务请求。
    pool_pre_ping: bool = Field(default=True, description="借出连接前是否执行存活检查。")
    # redis_max_connections 限制一个 API 进程最多占用的 Redis 连接数。
    redis_max_connections: int = Field(default=100, ge=1, description="Redis 连接池最大连接数。")
    # redis_health_check_interval_seconds 控制 redis-py 对空闲连接执行探活的周期。
    redis_health_check_interval_seconds: int = Field(
        default=30,
        ge=0,
        description="Redis 空闲连接健康检查间隔秒数；0 表示禁用周期健康检查。",
    )
    # redis_socket_connect_timeout_seconds 限制 Redis TCP/TLS 握手等待时间。
    redis_socket_connect_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        description="建立 Redis TCP/TLS 连接的最长等待秒数。",
    )
    # redis_socket_timeout_seconds 限制连接建立后单条 Redis 命令的读写等待时间。
    redis_socket_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        description="Redis 命令读写 Socket 的最长等待秒数。",
    )
    # redis_retry_on_timeout 决定命令超时后是否使用 redis-py 内置重试策略。
    redis_retry_on_timeout: bool = Field(
        default=True,
        description="Redis 命令超时时是否按 redis-py 策略执行重试。",
    )
    # echo_sql 只用于本地调试 SQL，生产应保持关闭以免日志泄露业务参数。
    echo_sql: bool = Field(default=False, description="是否输出 SQL 调试日志。")

    @model_validator(mode="after")
    def validate_redis_connection_mode(self) -> "DatabaseConfig":
        """确保 Redis 直连与 Sentinel 模式的配置边界明确。"""

        # Sentinel Master 名存在时，必须同时给出至少一个可发现 Master 的 Sentinel 节点。
        if self.redis_sentinel_master and not (self.redis_sentinel_nodes or "").strip():
            # 缺少节点时无法完成主库发现，启动前直接返回可读配置错误。
            raise ValueError("配置 redis_sentinel_master 时必须同时配置 redis_sentinel_nodes")
        # 节点列表存在但没有 Master 名同样没有连接语义，禁止半配置状态。
        if (self.redis_sentinel_nodes or "").strip() and not self.redis_sentinel_master:
            # 发现节点无法推断要订阅哪个 Master，要求部署方明确填写名称。
            raise ValueError("配置 redis_sentinel_nodes 时必须同时配置 redis_sentinel_master")
        # Sentinel 与 redis_url 同时有效会让实际连接目标不确定，要求二选一。
        if self.redis_sentinel_master and (self.redis_url or "").strip():
            # 报错而不是静默优先其中一方，防止切换期间误连旧 Redis。
            raise ValueError("Redis 直连 redis_url 与 Sentinel 配置不能同时启用")
        # 返回通过互斥与完整性校验的数据库配置。
        return self


class RetrievalConfig(_StrictConfigModel):
    """检索、RAG 和记忆召回配置。"""

    # top_k 控制融合排序后最多返回多少条证据。
    top_k: int = Field(default=8, description="默认召回 TopK。")
    # score_threshold 过滤综合得分过低的条目，防止弱相关内容进入 Prompt。
    score_threshold: float = Field(default=0.05, description="召回结果最低分数阈值。")
    # vector_weight 表示语义向量相似度在融合分中的占比。
    vector_weight: float = Field(default=0.45, description="向量相似度在最终分中的权重。")
    # lexical_weight 表示关键词精确匹配在融合分中的占比。
    lexical_weight: float = Field(default=0.25, description="关键词分在最终分中的权重。")
    # metadata_weight 为租户、领域、标签等元数据匹配保留权重。
    metadata_weight: float = Field(default=0.15, description="metadata 分在最终分中的权重。")
    # recency_weight 提升较新知识或记忆的排名，但不单独决定最终结果。
    recency_weight: float = Field(default=0.10, description="时间新近度在最终分中的权重。")
    # confidence_weight 将事实自身置信度纳入最终融合排序。
    confidence_weight: float = Field(default=0.05, description="事实置信度在最终分中的权重。")


class IntentRoutingConfig(_StrictConfigModel):
    """向量召回、LLM 语义裁定和活跃意图续接的集中配置。"""

    # provider 决定意图样例从本地配置还是生产 pgvector 知识库召回。
    provider: Literal["local", "pgvector"] = Field(
        default="local",
        description="意图知识库提供方：local 使用配置样例，pgvector 使用生产向量库。",
    )
    # catalog_path 只在 local provider 下使用，便于测试和未接知识库时稳定运行。
    catalog_path: str = Field(
        default="configs/intent_catalog.yaml",
        description="本地意图目录路径；生产切换 pgvector 后仍可作为故障降级目录。",
    )
    # knowledge_library 是 pgvector 中意图知识条目使用的 library metadata。
    knowledge_library: str = Field(
        default="intent_catalog",
        description="生产向量库中存放标准意图语料的 library 名称。",
    )
    # top_k 控制交给 LLM 裁定的候选数量，避免把整个意图库塞进 Prompt。
    top_k: int = Field(default=5, ge=1, le=20, description="向量召回的候选意图数量。")
    # 0.85 以上视为稳定高频表达，可以绕过 LLM 直接命中。
    high_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="高频固定意图直接路由的向量相似度阈值。",
    )
    # 0.60~0.85 属于相近但仍有歧义的表达，必须交给 LLM 做语义裁定。
    adjudication_similarity_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="进入候选意图 LLM 裁定区间的最低向量相似度。",
    )
    # 裁定置信度大于等于 0.80 时按高置信路由直接分发。
    high_confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="LLM 高置信路由阈值。",
    )
    # 0.60~0.80 允许路由，但必须写入中置信审计日志供离线补充知识库。
    medium_confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="LLM 中置信路由与主动澄清的分界阈值。",
    )
    # 活跃意图存在时先用轻量模型判断用户是否换题，不重新执行完整向量召回。
    intent_shift_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="小模型判定用户已经切换意图所需的最低置信度。",
    )
    # 活跃意图保存在 Redis Session 中，超过该时长后由 Redis Session TTL 自然清理。
    active_intent_ttl_seconds: int = Field(
        default=1800,
        ge=60,
        description="保险补问活跃意图的业务有效期，单位秒。",
    )
    # KYC 最多连续补问三轮，达到上限后基于已有事实生成初版策略。
    max_kyc_question_rounds: int = Field(
        default=3,
        ge=1,
        le=6,
        description="代码化保险会话允许的最大连续 KYC 补问轮数。",
    )
    # 模型抽取值只有携带原文证据且达到该置信度，才允许作为 confirmed 业务事实写入。
    kyc_evidence_min_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="保险 KYC 模型事实进入 confirmed 状态所需的最低证据置信度。",
    )

    @model_validator(mode="after")
    def validate_routing_thresholds(self) -> "IntentRoutingConfig":
        """拒绝重叠阈值区间，确保每个相似度和置信度只有一个确定动作。"""
        # 相似度的 LLM 裁定下界必须严格低于高相似直接路由阈值。
        if self.adjudication_similarity_threshold >= self.high_similarity_threshold:
            # 中间裁定区下界必须严格低于直接命中阈值，否则区间会重叠或消失。
            raise ValueError("intent_routing 相似度阈值必须满足 adjudication < high")
        # 裁定置信度的中档下界必须严格低于高置信直接分发阈值。
        if self.medium_confidence_threshold >= self.high_confidence_threshold:
            # 中置信下界必须严格低于高置信阈值，保证路由动作映射唯一。
            raise ValueError("intent_routing 置信度阈值必须满足 medium < high")
        # 返回校验后的模型实例，完成 Pydantic after-validator 契约。
        return self


class InsuranceKnowledgeConfig(_StrictConfigModel):
    """代码化保险处理器使用的双知识库和新闻素材配置。"""

    # provider=local 时只读取请求注入的脱敏测试素材；生产应配置 pgvector。
    provider: Literal["local", "pgvector"] = Field(
        default="local",
        description="保险知识提供方：local 或 pgvector。",
    )
    # 沟通方法库保存已审核方法、匿名案例和推荐动作。
    method_library: str = Field(default="insurance_methods", description="沟通方法知识库名称。")
    # 合规库保存产品合同边界、利益说明和禁用表达。
    compliance_library: str = Field(default="insurance_compliance", description="合同与合规知识库名称。")
    # 方法库返回较少的强相关内容，避免案例模板淹没客户事实。
    method_top_k: int = Field(default=3, ge=1, le=10, description="方法库 TopK。")
    # 合规边界允许多取一条，以覆盖利益、退保、汇率和持有期限等不同维度。
    compliance_top_k: int = Field(default=4, ge=1, le=10, description="合规库 TopK。")
    # 阈值沿用附件初始值，但接真实 Embedding 后必须用验证集重新校准。
    method_score_threshold: float = Field(default=0.30, ge=0.0, le=1.0, description="方法库最低融合分。")
    # compliance_score_threshold 独立控制合同/合规库证据进入上下文的最低融合分。
    compliance_score_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="合规库最低融合分。",
    )
    # 新闻只在 objective_material_need 非空时使用，实际 Provider 地址继续由 tools.yaml 配置。
    news_enabled: bool = Field(default=True, description="是否允许按需获取公开新闻素材。")


class MemoryConfig(_StrictConfigModel):
    """长期记忆策略配置。"""

    # MemoryConfig 同时关闭 model_ 前缀保护，并继承配置字段的 extra fail-fast 约束。
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # enabled 是长期记忆总开关；关闭后仍可使用本轮显式上下文。
    enabled: bool = Field(default=True, description="租户是否允许使用长期记忆。")
    # model_decision_enabled 控制规则不确定时是否允许模型判断是否召回长期记忆。
    model_decision_enabled: bool = Field(default=True, description="规则无法确定时是否调用模型做召回决策。")
    # decision_timeout_ms 给记忆决策小模型设置严格延迟预算，避免阻塞主回答。
    decision_timeout_ms: int = Field(default=1200, description="长期记忆召回决策模型的延迟预算。")
    # default_ttl_days 允许按环境整体覆盖 Preference TTL，None 时使用专用配置。
    default_ttl_days: int | None = Field(default=None, ge=1, description="长期偏好 TTL 覆盖值，空时使用 preference_ttl_days。")
    # max_recall_items 限制注入 Prompt 的长期记忆条目数。
    max_recall_items: int = Field(default=8, description="长期记忆最大召回条数。")
    # session_ttl_seconds 控制 Redis 会话上下文保留时间。
    session_ttl_seconds: int = Field(default=604800, ge=60, description="Redis Session 记忆 TTL，默认 7 天。")
    # task_ttl_seconds 控制当前任务执行状态和活跃意图的短期保留时间。
    task_ttl_seconds: int = Field(default=86400, ge=60, description="Redis Task 记忆 TTL，默认 1 天。")
    # entity_anchor_ttl_seconds 单独限制代词指代锚点寿命，不能跟随七天 Session 无限保留。
    entity_anchor_ttl_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
        description="Session 最近实体锚点 TTL，默认 30 分钟。",
    )
    # preference_ttl_days 定义 PostgreSQL 稳定偏好事实的默认生命周期。
    preference_ttl_days: int = Field(default=365, ge=1, description="PostgreSQL Preference 默认保留天数。")
    # max_session_messages 限制每个 Session 保存的最近消息数量，避免上下文无限增长。
    max_session_messages: int = Field(default=12, ge=2, le=200, description="一个 Session 最多保留的最近消息数。")
    # max_payload_bytes 防止单条 Redis JSON 占用过多内存或被超大输入放大。
    max_payload_bytes: int = Field(default=262144, ge=1024, description="单条 Redis 记忆 Payload 最大字节数。")
    # max_entries_per_tenant 为每层短期记忆增加租户级 Key 数硬上限。
    max_entries_per_tenant: int = Field(default=100000, ge=1, description="每个租户每层短期记忆最大 Key 数。")
    # cas_max_retries 限制 Redis 乐观锁冲突后的重试次数，避免高竞争下无限循环。
    cas_max_retries: int = Field(default=8, ge=1, le=32, description="Redis 乐观锁冲突最大重试次数。")
    # retention_batch_size 控制定时清理任务每批删除规模，平衡事务时间和吞吐。
    retention_batch_size: int = Field(default=1000, ge=1, le=10000, description="Retention Job 单批清理条数。")
    # message_audit_ttl_days 设置加密原始消息审计记录的保留期限。
    message_audit_ttl_days: int = Field(default=90, ge=1, description="加密消息审计保留天数。")
    # session_snapshot_ttl_days 设置业务会话快照的保留期限。
    session_snapshot_ttl_days: int = Field(default=30, ge=1, description="业务会话快照保留天数。")
    # analysis_ttl_days 设置分析过程和生成结果的保留期限。
    analysis_ttl_days: int = Field(default=365, ge=1, description="分析运行和生成结果保留天数。")
    # business_event_ttl_days 设置机会事件与结果事件的保留期限。
    business_event_ttl_days: int = Field(default=730, ge=1, description="业务事件与结果保留天数。")
    # business_fact_ttl_days 设置客户、顾问稳定事实的最长默认保留期限。
    business_fact_ttl_days: int = Field(default=1095, ge=1, description="客户/顾问稳定事实保留天数。")
    # encryption_key_env 只保存 Secret 变量名称，密钥值不进入 YAML 或日志。
    encryption_key_env: str = Field(default="MEMORY_ENCRYPTION_KEY", description="业务记忆字段加密密钥环境变量名。")
    # require_encryption 为真时，生产 Runtime 会检查密钥存在且达到最小长度。
    require_encryption: bool = Field(default=True, description="生产环境是否强制要求业务记忆字段加密密钥。")


class ApiRuntimeConfig(_StrictConfigModel):
    """FastAPI 中间件与生产 Runtime 配置。"""

    # shared_engine 要求 API 生命周期复用单个 WorkflowEngine 和连接池。
    shared_engine: bool = Field(default=True, description="是否在应用生命周期内复用 WorkflowEngine。")
    # require_api_key 控制 Agent Gateway 是否强制执行 API Key 校验。
    require_api_key: bool = Field(default=True, description="是否要求 Agent Gateway API Key。")
    # api_key_env 是兼容全局密钥的环境变量名，不保存密钥本身。
    api_key_env: str = Field(default="AGENT_API_KEY", description="读取 Gateway API Key 的环境变量名。")
    # tenant_api_keys_env 指向 tenant_id -> API Key JSON 映射的 Secret 环境变量。
    tenant_api_keys_env: str = Field(
        default="AGENT_TENANT_API_KEYS",
        description="读取 tenant 到 API Key JSON 映射的环境变量名。",
    )
    # allow_global_api_key 默认关闭，避免一个共享凭据横跨全部租户。
    allow_global_api_key: bool = Field(default=False, description="是否允许不绑定租户的全局 API Key。")
    # rate_limit_per_minute 实际表示每个固定窗口允许的单租户请求数，名称为兼容旧配置保留。
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="单租户每个固定窗口的请求上限；字段名为兼容旧配置保留。",
    )
    # rate_limit_window_seconds 定义 Redis 固定窗口的实际秒数。
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        description="Redis 固定窗口限流的窗口长度，单位秒；默认 60 秒。",
    )
    # rate_limit_key_ttl_seconds 决定窗口计数 Key 何时清理，必须覆盖窗口本身。
    rate_limit_key_ttl_seconds: int = Field(
        default=120,
        ge=1,
        description="Redis 限流计数 Key 的过期秒数，必须不短于限流窗口。",
    )
    # max_request_bytes 在 Pydantic 解析前限制声明长度和实际 Body 字节数。
    max_request_bytes: int = Field(default=1048576, ge=1024, description="HTTP 请求体最大字节数。")
    # allowed_origins 为空时不注册 CORS；非空时只允许精确列出的浏览器 Origin。
    allowed_origins: list[str] = Field(default_factory=list, description="允许的 CORS Origin 白名单。")
    # allowed_methods 限定浏览器预检允许的 Method，不影响服务端实际路由定义。
    allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST"],
        description="CORS 预检允许的 HTTP Method 精确列表。",
    )
    # allowed_headers 限定跨域请求可携带的 Header，默认只开放网关所需字段。
    allowed_headers: list[str] = Field(
        default_factory=lambda: ["content-type", "x-api-key", "x-tenant-id", "x-trace-id"],
        description="CORS 预检允许的请求 Header 精确列表。",
    )
    # allow_credentials 控制 CORS 是否允许 Cookie/HTTP Auth，并受通配符校验约束。
    allow_credentials: bool = Field(
        default=False,
        description="CORS 是否允许浏览器携带 Cookie 或 HTTP 认证凭据。",
    )
    # production_backends_required 防止生产 API 在后端故障时悄悄降级到进程内存。
    production_backends_required: bool = Field(default=True, description="API 启动时是否强制 Redis/PostgreSQL 可用。")

    @model_validator(mode="after")
    def validate_middleware_boundaries(self) -> "ApiRuntimeConfig":
        """校验限流 Key 生命周期和带凭据 CORS 的浏览器安全约束。"""
        # Key 若早于窗口过期，同一个自然窗口会重新计数，导致限流预算可被绕过。
        if self.rate_limit_key_ttl_seconds < self.rate_limit_window_seconds:
            # 计数 Key 必须至少覆盖完整窗口，否则过期后可在同一窗口重新获得预算。
            raise ValueError("api.rate_limit_key_ttl_seconds 不能小于 rate_limit_window_seconds")
        # 浏览器规范要求携带凭据时 Origin、Method 和 Header 都必须显式列举。
        wildcard_fields = (
            "*" in self.allowed_origins
            or "*" in self.allowed_methods
            or "*" in self.allowed_headers
        )
        # 只有允许携带浏览器凭据时才额外禁止任一白名单中的通配符。
        if self.allow_credentials and wildcard_fields:
            # 带 Cookie/HTTP 凭据的跨域请求必须使用精确白名单，拒绝任意通配来源。
            raise ValueError("api.allow_credentials=true 时 CORS 白名单不能包含通配符 *")
        # 返回通过约束检查的配置实例，完成 Pydantic after-validator。
        return self


class ArtifactRegistryRuntimeConfig(_StrictConfigModel):
    """Unified Artifact Registry and layered loading configuration."""

    enabled: bool = Field(default=True, description="是否在 Agent Run 开始时创建固定版本快照。")
    bootstrap_repository_artifacts: bool = Field(
        default=True,
        description="是否注册仓库内签名/受信示例 Artifact；不会扫描或执行上传代码。",
    )
    catalog_top_k: int = Field(default=5, ge=1, le=20, description="Tool/Skill 轻量目录候选 TopK。")
    schema_token_budget: int = Field(default=2500, ge=100, description="候选 Tool Schema Token 预算。")
    skill_context_budget: int = Field(default=6000, ge=100, description="选中 Skill 全文上下文预算。")
    cache_ttl_seconds: int = Field(default=300, ge=1, description="不可变版本本地缓存 TTL。")
    negative_cache_ttl_seconds: int = Field(default=15, ge=1, description="加载失败负缓存 TTL。")
    maximum_cache_entries: int = Field(default=256, ge=1, description="单进程不可变版本 LRU 上限。")


class RuntimeSettings(_StrictConfigModel):
    """Agent Runtime 的集中配置。"""

    # app_env 决定是否启用 staging/prod 的严格 Provider 和模型完整性检查。
    app_env: str = Field(default="local", description="运行环境，例如 local、test、staging、prod。")
    # models 按任务名称索引模型端点，例如 embedding、intent_classifier 和 KYC extractor。
    models: dict[str, ModelEndpointConfig] = Field(default_factory=dict, description="所有模型端点配置。")
    # database 聚合 PostgreSQL、Redis 和连接池参数，是生产 Runtime 的必需配置。
    database: DatabaseConfig = Field(..., description="数据库配置。")
    # retrieval 保存通用混合检索的 TopK、阈值和融合权重。
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig, description="检索配置。")
    # intent_routing 保存向量召回、LLM 裁定、活跃意图和 KYC 证据阈值。
    intent_routing: IntentRoutingConfig = Field(
        default_factory=IntentRoutingConfig,
        description="向量意图召回、LLM 裁定、置信度路由和活跃意图配置。",
    )
    # insurance_knowledge 保存保险方法库、合规库和可选新闻素材开关。
    insurance_knowledge: InsuranceKnowledgeConfig = Field(
        default_factory=InsuranceKnowledgeConfig,
        description="代码化保险处理器的双知识库配置。",
    )
    # memory 统一管理短期/长期/业务记忆容量、TTL、CAS 和加密要求。
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="长期记忆配置。")
    # api 保存网关鉴权、限流、CORS、请求大小和生命周期约束。
    api: ApiRuntimeConfig = Field(default_factory=ApiRuntimeConfig, description="FastAPI 中间件与生命周期配置。")
    # artifact_registry controls repository bootstrap and layered runtime loading without changing Agent APIs.
    artifact_registry: "ArtifactRegistryRuntimeConfig" = Field(
        default_factory=lambda: ArtifactRegistryRuntimeConfig(),
        description="统一 Tool/Skill/Prompt Registry 运行时配置。",
    )

    def require_model(self, name: str) -> ModelEndpointConfig:
        """读取必需模型配置；缺失时直接报错，避免业务节点静默降级。"""
        # 按任务稳定名称查找端点配置，不根据供应商或模型名做模糊匹配。
        config = self.models.get(name)
        # 没有对应任务配置时禁止调用方自行猜测默认模型。
        if config is None:
            # 未声明任务模型时立即报错，调用方不能误用其它默认模型替代。
            raise RuntimeError(f"缺少模型配置：models.{name}")
        # 已声明配置仍必须同时包含可调用 URL、凭据和模型标识。
        if not config.base_url or not config.api_key or not config.model:
            # URL、密钥、模型名任一为空都表示端点不可调用，按启动配置错误处理。
            raise RuntimeError(f"模型配置不完整：models.{name}")
        # 返回经过完整性检查的端点配置供客户端构造。
        return config


def load_runtime_settings(config_dir: str | Path = "configs") -> RuntimeSettings:
    """从配置目录加载 RuntimeSettings。"""
    # 将字符串或 Path 统一规范化，便于后续使用 `/` 拼接各约定文件名。
    base = Path(config_dir)
    # 每个文件只允许一个约定的顶层 Section，防止 section 名拼错后悄悄使用默认配置。
    models = _load_yaml_section(base / "models.yaml", "models")
    # 加载 PostgreSQL、Redis 及连接池配置。
    database = _load_yaml_section(base / "database.yaml", "database")
    # 加载通用混合检索阈值和权重。
    retrieval = _load_yaml_section(base / "retrieval.yaml", "retrieval")
    # 加载双层意图路由阈值、Provider 和活跃意图参数。
    intent_routing = _load_yaml_section(base / "intent_routing.yaml", "intent_routing")
    # 从保险处理器配置文件读取双知识库 section。
    insurance_knowledge = _load_yaml_section(
        base / "insurance_handler.yaml",
        "insurance_knowledge",
    )
    # 加载记忆 TTL、容量、并发和加密配置。
    memory = _load_yaml_section(base / "memory.yaml", "memory")
    # 加载 API 鉴权、限流、CORS 和请求大小配置。
    api = _load_yaml_section(base / "api.yaml", "api")
    # 加载统一 Artifact Registry、候选预算和缓存策略。
    artifact_registry = _load_yaml_section(base / "artifact_registry.yaml", "artifact_registry")
    # 交给严格 Pydantic 模型统一类型转换、未知字段拒绝和跨字段校验。
    return RuntimeSettings.model_validate(
        {
            "app_env": os.getenv("APP_ENV", "local"),
            "models": models,
            "database": database,
            "retrieval": retrieval,
            "intent_routing": intent_routing,
            "insurance_knowledge": insurance_knowledge,
            "memory": memory,
            "api": api,
            "artifact_registry": artifact_registry,
        }
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 并做环境变量插值。"""
    # 配置文件不存在时允许上层默认值接管，不直接在文件读取层决定是否必填。
    if not path.exists():
        # 文件缺失时返回空 Mapping，由上层模型决定使用默认值还是报告必填项缺失。
        return dict()
    # 使用 UTF-8 读取配置文本，避免平台默认编码导致中文注释或值解析异常。
    raw = path.read_text(encoding="utf-8")
    # 在 YAML 解析前替换环境变量占位符；缺失变量替换为空并由结构化校验决定是否合法。
    expanded = ENV_PATTERN.sub(lambda match: _resolve_environment_value(match.group(1)), raw)
    # safe_load 禁止构造任意 Python 对象；空文档规范化为空 Mapping。
    data = yaml.safe_load(expanded) or {}
    # 顶层只接受字典，确保后续 section 选择和未知键校验语义稳定。
    if not isinstance(data, dict):
        # 所有配置文件顶层必须是 Mapping，列表或标量会使 section 语义不明确。
        raise ValueError(f"配置文件必须是 YAML mapping：{path}")
    # 返回已经完成环境变量插值并验证顶层类型的字典。
    return data


def _resolve_environment_value(name: str) -> str:
    """读取配置占位符对应环境变量，并兼容 OpenAI-compatible 的常用别名。"""

    # 显式 LLM_* 或其它同名变量优先，部署方可用它覆盖兼容别名而不改变 YAML。
    value = os.getenv(name, "")
    # 只有显式变量提供非空值时，才允许它覆盖下面的兼容别名。
    if value:
        # 非空显式值是唯一优先级最高来源，直接用于 YAML 插值。
        return value
    # 只有 LLM 地址与密钥允许回退 OpenAI-compatible 别名，避免任意变量之间隐式串线。
    fallback_name = ENV_FALLBACK_ALIASES.get(name)
    # 命中白名单别名时，才从对应的 OpenAI-compatible 环境变量取值。
    if fallback_name:
        # 兼容企业模型网关使用 OPENAI_BASE_URL/OPENAI_API_KEY 的标准命名。
        return os.getenv(fallback_name, "")
    # 无显式值且没有白名单别名时保留原有空字符串语义。
    return ""


def _load_yaml_section(path: Path, section_name: str) -> dict[str, Any]:
    """读取单 Section 配置，并拒绝意外顶层字段或非 Mapping Section。"""
    # 先完成文件存在性、环境变量插值和顶层 Mapping 类型校验。
    data = _load_yaml(path)
    # 空文件或文件不存在时仍交给 Pydantic 默认值处理；非空文件只能声明约定 Section。
    unexpected_sections = sorted(set(data) - {section_name})
    # 任一未知顶层键都视为配置拼写或文件布局错误。
    if unexpected_sections:
        # 排序并拼接未知 section，生成稳定、可读且便于测试断言的错误文本。
        names = ", ".join(unexpected_sections)
        # 未知顶层字段通常是拼写错误，必须 fail-fast 而不是静默使用默认配置。
        raise ValueError(f"配置文件包含未知顶层字段：{path}: {names}")
    # 文件为空或 section 缺失时使用空 Mapping，后续 Pydantic 应用默认值。
    section = data.get(section_name, {})
    # Section 本身必须是字段到值的 Mapping，不能是列表或标量。
    if not isinstance(section, dict):
        # 约定 section 必须是键值 Mapping，拒绝列表或标量配置。
        raise ValueError(f"配置 Section 必须是 YAML mapping：{path}: {section_name}")
    # 返回单个约定 section，防止业务层直接依赖 YAML 文件布局。
    return section
