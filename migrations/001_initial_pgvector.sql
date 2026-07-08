CREATE EXTENSION IF NOT EXISTS vector;

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
    trace_id text,
    speaker_role text NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS task_memory (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    session_id text NOT NULL,
    task_id text,
    state_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS long_term_memory_items (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text NOT NULL,
    user_id text NOT NULL,
    scope text NOT NULL,
    memory_type text NOT NULL,
    content text NOT NULL,
    normalized_content text,
    embedding vector(3072) NOT NULL,
    source_type text NOT NULL,
    source_id text NOT NULL,
    evidence_text text NOT NULL,
    confidence double precision NOT NULL DEFAULT 0.0,
    status text NOT NULL DEFAULT 'active',
    risk_level text NOT NULL DEFAULT 'low',
    consent_status text NOT NULL DEFAULT 'granted',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    deleted_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ltm_tenant_user_scope
    ON long_term_memory_items (tenant_id, user_id, scope)
    WHERE status = 'active' AND deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_ltm_embedding
    ON long_term_memory_items
    USING ivfflat (embedding vector_cosine_ops)
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
    embedding vector(3072) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_tenant_document ON rag_chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector
    ON rag_chunk_embeddings
    USING ivfflat (embedding vector_cosine_ops)
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
    output_text text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
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
