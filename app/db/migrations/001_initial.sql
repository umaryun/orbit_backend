-- ============================================================================
-- Orbit Backend — Initial Migration
-- Run against Supabase SQL Editor or via psql
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Enums ────────────────────────────────────────────────────────────────────

DO $$ BEGIN
    CREATE TYPE task_status_enum AS ENUM ('todo', 'in_progress', 'blocked', 'done');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE task_priority_enum AS ENUM ('low', 'medium', 'high', 'urgent');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE knowledge_category_enum AS ENUM (
        'architecture', 'api_contract', 'client_note', 'prd',
        'technical_decision', 'gotcha', 'general'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE conversation_role_enum AS ENUM ('user', 'assistant', 'tool');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE project_status_enum AS ENUM ('active', 'paused', 'completed', 'archived');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ── Users ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number  VARCHAR(20) NOT NULL UNIQUE,
    first_name    VARCHAR(100),
    timezone      VARCHAR(50) DEFAULT 'UTC',
    onboarding_state JSONB DEFAULT '{}',
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_users_phone ON users (phone_number);


-- ── Projects ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS projects (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        VARCHAR(200) NOT NULL,
    client_name VARCHAR(200),
    status      project_status_enum DEFAULT 'active',
    created_at  TIMESTAMPTZ DEFAULT now()
);


-- ── Project Profiles ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS project_profiles (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id        UUID NOT NULL UNIQUE REFERENCES projects(id) ON DELETE CASCADE,
    tech_stack        JSONB DEFAULT '[]',
    repo_url          VARCHAR(500),
    deployment_target VARCHAR(200),
    client_quirks     TEXT,
    primary_goal      TEXT,
    updated_at        TIMESTAMPTZ DEFAULT now()
);


-- ── Tasks ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tasks (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id     UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title          VARCHAR(500) NOT NULL,
    status         task_status_enum DEFAULT 'todo',
    blocker_reason TEXT,
    priority       task_priority_enum DEFAULT 'medium',
    due_at         TIMESTAMPTZ,
    completed_at   TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_tasks_project_status ON tasks (project_id, status);
CREATE INDEX IF NOT EXISTS ix_tasks_user_due ON tasks (user_id, due_at);


-- ── Project Knowledge (pgvector) ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS project_knowledge (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id  UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    category    knowledge_category_enum NOT NULL,
    title       VARCHAR(500),
    content     TEXT NOT NULL,
    embedding   vector(768),
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_knowledge_embedding
    ON project_knowledge
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);


-- ── Conversation Logs ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversation_logs (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role       conversation_role_enum NOT NULL,
    content    TEXT NOT NULL,
    media_type VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_convlog_user_created ON conversation_logs (user_id, created_at);
