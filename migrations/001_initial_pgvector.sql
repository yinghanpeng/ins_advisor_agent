CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS agent_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL UNIQUE,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_trace_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    span_id text,
    node_name text,
    event_name text NOT NULL,
    latency_ms integer,
    model_name text,
    token_input integer,
    token_output integer,
    model_cost numeric,
    tool_name text,
    tool_latency_ms integer,
    retrieval_latency_ms integer,
    memory_recall_latency_ms integer,
    db_latency_ms integer,
    error_type text,
    error_message text,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS state_transitions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    from_state text,
    to_state text NOT NULL,
    reason text NOT NULL DEFAULT '',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS short_term_messages (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    session_id text NOT NULL,
    message_key text NOT NULL,
    trace_id text,
    speaker_role text NOT NULL,
    content_ciphertext bytea NOT NULL,
    content_redacted text NOT NULL DEFAULT '',
    content_hash text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, session_id, message_key)
);

CREATE INDEX IF NOT EXISTS idx_short_term_messages_session_created
    ON short_term_messages (tenant_id, session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS task_memory (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    session_id text NOT NULL,
    task_id text NOT NULL DEFAULT 'active',
    state_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
    version bigint NOT NULL DEFAULT 1,
    source_version bigint NOT NULL DEFAULT 1,
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, session_id, task_id)
);


CREATE TABLE IF NOT EXISTS memory_items (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    user_id text NOT NULL,
    scope text NOT NULL,
    memory_type text NOT NULL,
    memory_key text NOT NULL,
    content text NOT NULL,
    normalized_content text,
    source_type text NOT NULL,
    source_id text NOT NULL,
    evidence_ciphertext bytea NOT NULL,
    evidence_hash text NOT NULL,
    confidence double precision NOT NULL DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    status text NOT NULL DEFAULT 'active',
    risk_level text NOT NULL DEFAULT 'low',
    risk_rank smallint NOT NULL DEFAULT 0 CHECK (risk_rank BETWEEN 0 AND 2),
    consent_status text NOT NULL DEFAULT 'granted',
    version bigint NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    deleted_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (tenant_id, user_id, scope, memory_key)
);

CREATE INDEX IF NOT EXISTS idx_memory_items_tenant_user_scope
    ON memory_items (tenant_id, user_id, scope)
    WHERE status = 'active' AND deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_memory_items_expiration
    ON memory_items (expires_at)
    WHERE status = 'active' AND deleted_at IS NULL AND expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS memory_item_embeddings (
    memory_item_id uuid PRIMARY KEY REFERENCES memory_items(id) ON DELETE CASCADE,
    tenant_id text NOT NULL,
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL CHECK (embedding_dimensions = 3072),
    embedding halfvec(3072) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_item_embeddings_vector
    ON memory_item_embeddings
    USING ivfflat (embedding halfvec_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS memory_recall_decisions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    decision jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_recall_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_documents (
    id text PRIMARY KEY,
    tenant_id text NOT NULL,
    title text NOT NULL,
    source_uri text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id text PRIMARY KEY,
    tenant_id text NOT NULL,
    document_id text NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_chunk_embeddings (
    chunk_id text PRIMARY KEY REFERENCES rag_chunks(id) ON DELETE CASCADE,
    tenant_id text NOT NULL,
    embedding halfvec(3072) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_tenant_document ON rag_chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector
    ON rag_chunk_embeddings
    USING ivfflat (embedding halfvec_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS tool_calls (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    tool_name text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    tool_call_id uuid NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    status text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS human_approval_requests (
    id text PRIMARY KEY,
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    checkpoint_id text NOT NULL,
    pending_action text NOT NULL,
    risk_reason text NOT NULL,
    approval_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    required_approver_role text NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS generated_outputs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    output_type text NOT NULL,
    input_context jsonb NOT NULL DEFAULT '{}'::jsonb,
    output_ciphertext bytea NOT NULL,
    output_redacted text NOT NULL DEFAULT '',
    output_hash text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS feedback_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    trace_id text NOT NULL,
    feedback_type text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
