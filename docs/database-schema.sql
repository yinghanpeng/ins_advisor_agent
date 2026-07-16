-- 重要：本文件是历史设计说明，不是可执行 migration 入口。
-- 生产结构以 migrations/ 下的编号 SQL 为唯一权威来源，请使用 make db-upgrade。
-- 保险高客沟通教练业务记忆系统 PostgreSQL DDL
-- 第一版本地 demo 使用 InMemoryBusinessMemoryStore；本文件是生产落地方案。
-- 生产建议启用：
--   CREATE EXTENSION IF NOT EXISTS pgcrypto;
--   CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE advisors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    display_alias TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_advisors_tenant ON advisors (tenant_id);

CREATE TABLE advisor_profile_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    advisor_id TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    fact_value JSONB NOT NULL,
    confidence NUMERIC(4,3) NOT NULL DEFAULT 1.0,
    source_type TEXT NOT NULL,
    source_conversation_id TEXT,
    source_message_id TEXT,
    evidence_text TEXT NOT NULL,
    is_current BOOLEAN NOT NULL DEFAULT true,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_advisor_facts_tenant_subject ON advisor_profile_facts (tenant_id, advisor_id);
CREATE INDEX idx_advisor_facts_current_key
    ON advisor_profile_facts (tenant_id, advisor_id, fact_key)
    WHERE is_current = true;

CREATE TABLE customers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    display_alias TEXT NOT NULL DEFAULT '',
    pii_ref_id TEXT,
    phone_hash TEXT,
    email_hash TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_customers_tenant ON customers (tenant_id);
CREATE INDEX idx_customers_phone_hash ON customers (tenant_id, phone_hash) WHERE phone_hash IS NOT NULL;
CREATE INDEX idx_customers_email_hash ON customers (tenant_id, email_hash) WHERE email_hash IS NOT NULL;

CREATE TABLE customer_profile_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    fact_value JSONB NOT NULL,
    normalized_value JSONB,
    confidence NUMERIC(4,3) NOT NULL DEFAULT 1.0,
    certainty TEXT NOT NULL CHECK (certainty IN ('confirmed', 'uncertain')),
    sensitivity_level TEXT NOT NULL DEFAULT 'internal',
    source_type TEXT NOT NULL,
    source_conversation_id TEXT,
    source_message_id TEXT,
    evidence_text TEXT NOT NULL,
    extraction_run_id TEXT,
    is_current BOOLEAN NOT NULL DEFAULT true,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_customer_facts_tenant_subject ON customer_profile_facts (tenant_id, customer_id);
CREATE INDEX idx_customer_facts_current_key
    ON customer_profile_facts (tenant_id, customer_id, fact_key)
    WHERE is_current = true;
CREATE INDEX idx_customer_facts_uncertain
    ON customer_profile_facts (tenant_id, customer_id, fact_key)
    WHERE is_current = true AND certainty = 'uncertain';

CREATE TABLE opportunity_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    advisor_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    subject_type TEXT NOT NULL DEFAULT 'unclear',
    case_status TEXT NOT NULL DEFAULT 'active',
    target_persona TEXT NOT NULL DEFAULT 'unknown',
    trigger_module TEXT NOT NULL DEFAULT 'unknown',
    current_stage TEXT NOT NULL DEFAULT 'collect_kyc',
    relationship_strength TEXT NOT NULL DEFAULT '',
    latest_kyc_completeness_score INTEGER NOT NULL DEFAULT 0,
    latest_opportunity_score INTEGER NOT NULL DEFAULT 0,
    latest_external_grade TEXT NOT NULL DEFAULT 'D',
    latest_missing_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
    latest_support_note TEXT NOT NULL DEFAULT '',
    next_best_action TEXT NOT NULL DEFAULT '',
    workflow_version TEXT NOT NULL DEFAULT 'local-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_cases_active
    ON opportunity_cases (tenant_id, advisor_id, customer_id)
    WHERE case_status = 'active';

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    advisor_id TEXT NOT NULL,
    customer_id TEXT,
    opportunity_case_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_conversations_tenant_case ON conversations (tenant_id, opportunity_case_id);

-- 通用短期消息只作为加密审计副本；在线会话窗口由 Redis Messages List 提供。
CREATE TABLE short_term_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_key TEXT NOT NULL,
    trace_id TEXT,
    speaker_role TEXT NOT NULL,
    content_ciphertext BYTEA NOT NULL,
    content_redacted TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, session_id, message_key)
);

CREATE INDEX idx_short_term_messages_session_created
    ON short_term_messages (tenant_id, session_id, created_at DESC);

CREATE TABLE agent_session_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    opportunity_case_id TEXT,
    profile_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    practitioner_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    information_status TEXT NOT NULL DEFAULT 'insufficient',
    subject_type TEXT NOT NULL DEFAULT 'unclear',
    target_persona TEXT NOT NULL DEFAULT 'unknown',
    advisor_stage TEXT NOT NULL DEFAULT 'unknown',
    trigger_module TEXT NOT NULL DEFAULT 'unknown',
    current_stage TEXT NOT NULL DEFAULT 'collect_kyc',
    missing_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
    asked_focuses JSONB NOT NULL DEFAULT '[]'::jsonb,
    kyc_question_round_count INTEGER NOT NULL DEFAULT 0,
    kyc_completeness_score INTEGER NOT NULL DEFAULT 0,
    opportunity_score INTEGER NOT NULL DEFAULT 0,
    external_grade TEXT NOT NULL DEFAULT 'D',
    objective_material_need TEXT NOT NULL DEFAULT '',
    support_note TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_states_latest ON agent_session_states (tenant_id, conversation_id, created_at DESC);

CREATE TABLE kyc_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    opportunity_case_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    focus_key TEXT NOT NULL,
    question_text TEXT NOT NULL,
    question_status TEXT NOT NULL DEFAULT 'asked',
    -- 回答证据由 short_term_messages 的会话审计记录承载，不再维护旧消息表外键。
    extracted_fact_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    asked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX idx_kyc_questions_focus
    ON kyc_questions (tenant_id, opportunity_case_id, focus_key);

CREATE TABLE analysis_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    opportunity_case_id TEXT,
    model_name TEXT NOT NULL DEFAULT 'deterministic-local',
    workflow_version TEXT NOT NULL DEFAULT 'local-v1',
    prompt_version TEXT NOT NULL DEFAULT 'kyc-analyzer-v1',
    input_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    information_status TEXT NOT NULL DEFAULT 'insufficient',
    target_persona TEXT NOT NULL DEFAULT 'unknown',
    trigger_module TEXT NOT NULL DEFAULT 'unknown',
    current_stage TEXT NOT NULL DEFAULT 'collect_kyc',
    kyc_completeness_score INTEGER NOT NULL DEFAULT 0,
    opportunity_score INTEGER NOT NULL DEFAULT 0,
    external_grade TEXT NOT NULL DEFAULT 'D',
    match_evidence TEXT NOT NULL DEFAULT '',
    route_reason TEXT NOT NULL DEFAULT '',
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_analysis_runs_case_created ON analysis_runs (tenant_id, opportunity_case_id, created_at DESC);

CREATE TABLE business_generated_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    opportunity_case_id TEXT,
    output_type TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'deterministic-local',
    workflow_version TEXT NOT NULL DEFAULT 'local-v1',
    prompt_version TEXT NOT NULL DEFAULT 'strategy-generator-v1',
    input_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_text TEXT NOT NULL,
    safety_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
    used_news_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    used_case_pattern_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_business_generated_outputs_case_created ON business_generated_outputs (tenant_id, opportunity_case_id, created_at DESC);

CREATE TABLE memory_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    conversation_id TEXT,
    opportunity_case_id TEXT,
    customer_id TEXT,
    advisor_id TEXT,
    event_type TEXT NOT NULL,
    event_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence_text TEXT NOT NULL DEFAULT '',
    -- 事件保留脱敏证据文本；不再引用已删除的旧会话消息表。
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_memory_events_case_created ON memory_events (tenant_id, opportunity_case_id, created_at DESC);

CREATE TABLE case_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    opportunity_case_id TEXT NOT NULL,
    outcome_type TEXT NOT NULL,
    outcome_detail TEXT NOT NULL DEFAULT '',
    source_conversation_id TEXT,
    -- 结果通过会话标识关联上下文，不再依赖旧会话消息主键。
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_case_outcomes_case_created ON case_outcomes (tenant_id, opportunity_case_id, created_at DESC);

CREATE TABLE corpus_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    batch_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    upload_by TEXT NOT NULL DEFAULT '',
    raw_file_uri TEXT NOT NULL DEFAULT '',
    total_conversations INTEGER NOT NULL DEFAULT 0,
    pii_status TEXT NOT NULL DEFAULT 'raw',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_corpus_batches_tenant_created ON corpus_batches (tenant_id, created_at DESC);

CREATE TABLE corpus_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    case_title TEXT NOT NULL,
    scene_type TEXT NOT NULL DEFAULT '',
    target_persona TEXT NOT NULL DEFAULT 'unknown',
    trigger_module TEXT NOT NULL DEFAULT 'unknown',
    advisor_stage TEXT NOT NULL DEFAULT 'unknown',
    customer_stage TEXT NOT NULL DEFAULT 'unknown',
    relationship_strength TEXT NOT NULL DEFAULT '',
    final_outcome TEXT NOT NULL DEFAULT '',
    quality_score INTEGER NOT NULL DEFAULT 0,
    raw_conversation_uri TEXT NOT NULL DEFAULT '',
    redacted_conversation_uri TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_corpus_cases_tags ON corpus_cases (tenant_id, target_persona, trigger_module, scene_type);

CREATE TABLE corpus_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    corpus_case_id TEXT NOT NULL,
    seq_no INTEGER NOT NULL,
    speaker_role TEXT NOT NULL,
    content_redacted TEXT NOT NULL,
    message_type TEXT NOT NULL DEFAULT '',
    detected_intent TEXT NOT NULL DEFAULT '',
    sentiment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_corpus_messages_seq ON corpus_messages (tenant_id, corpus_case_id, seq_no);

CREATE TABLE dialogue_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    scene_type TEXT NOT NULL DEFAULT '',
    target_persona TEXT NOT NULL DEFAULT 'unknown',
    trigger_module TEXT NOT NULL DEFAULT 'unknown',
    advisor_stage TEXT NOT NULL DEFAULT 'unknown',
    situation_summary TEXT NOT NULL,
    customer_signal TEXT NOT NULL DEFAULT '',
    recommended_move TEXT NOT NULL,
    bad_move TEXT NOT NULL DEFAULT '',
    example_wording TEXT NOT NULL DEFAULT '',
    outcome_label TEXT NOT NULL DEFAULT '',
    confidence NUMERIC(4,3) NOT NULL DEFAULT 0.8,
    source_corpus_case_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    approved_for_generation BOOLEAN NOT NULL DEFAULT false,
    risk_level TEXT NOT NULL DEFAULT 'medium',
    compliance_notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_dialogue_patterns_generation
    ON dialogue_patterns (tenant_id, pattern_type, target_persona, trigger_module)
    WHERE approved_for_generation = true AND risk_level <> 'high';

CREATE TABLE memory_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    owner_table TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL DEFAULT 'configured-embedding-model',
    embedding halfvec(3072),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_memory_embeddings_owner ON memory_embeddings (tenant_id, owner_table, owner_id);
-- 3072 维 Embedding 使用 halfvec，才能被 pgvector 的 ivfflat 索引支持：
-- CREATE INDEX idx_memory_embeddings_vector ON memory_embeddings USING ivfflat (embedding halfvec_cosine_ops) WITH (lists = 100);
-- 重要：本文件是历史设计说明，不是可执行 migration 入口。
-- 生产结构以 migrations/001_initial_pgvector.sql 到最新编号文件为唯一权威来源，
-- `make db-upgrade` 会通过 schema_migrations 台账和校验和顺序执行。
