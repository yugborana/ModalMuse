-- ═══════════════════════════════════════════════════════════════════════════════
-- MODALMUSE - SUPABASE SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════
-- Run this in Supabase SQL Editor to set up all required tables
-- ═══════════════════════════════════════════════════════════════════════════════
--
-- REQUIRED STORAGE BUCKETS (create manually in Supabase Storage dashboard):
-- 1. "images" - Public bucket for indexed document images
--    - Enable public access for image display in frontend
--
-- ═══════════════════════════════════════════════════════════════════════════════

-- Enable UUID extension (should already be enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ═══════════════════════════════════════════════════════════════════════════════
-- CONVERSATIONS TABLE
-- Stores chat conversation sessions
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster sorting by updated_at
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- MESSAGES TABLE
-- Stores individual messages within conversations
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources JSONB,  -- Stores retrieved sources for assistant messages
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster message lookup by conversation
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(conversation_id, created_at);

-- ═══════════════════════════════════════════════════════════════════════════════
-- INDEXING TASKS TABLE
-- Tracks document indexing progress (replaces .task_state.json)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS indexing_tasks (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('started', 'processing', 'completed', 'failed')),
    progress INTEGER DEFAULT 0,
    message TEXT,
    file_name TEXT,
    text_vectors INTEGER DEFAULT 0,
    image_vectors INTEGER DEFAULT 0,
    error TEXT,
    from_cache BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for finding active tasks
CREATE INDEX IF NOT EXISTS idx_indexing_tasks_status ON indexing_tasks(status);

-- ═══════════════════════════════════════════════════════════════════════════════
-- QUERY CACHE TABLE (Optional)
-- Caches query embeddings to reduce API calls
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS query_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    embedding JSONB NOT NULL,  -- Stored as JSON array of floats
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for cache expiration cleanup
CREATE INDEX IF NOT EXISTS idx_query_cache_expires ON query_cache(expires_at);

-- ═══════════════════════════════════════════════════════════════════════════════
-- PARSE CACHE TABLE (Optional)
-- Caches LlamaParse results to avoid re-parsing documents
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS parse_cache (
    file_hash TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    job_id TEXT,
    parsed_json JSONB,  -- Parsed document content
    images_data JSONB,  -- Image metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for auto-updating updated_at
DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_indexing_tasks_updated_at ON indexing_tasks;
CREATE TRIGGER update_indexing_tasks_updated_at
    BEFORE UPDATE ON indexing_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_parse_cache_updated_at ON parse_cache;
CREATE TRIGGER update_parse_cache_updated_at
    BEFORE UPDATE ON parse_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ═══════════════════════════════════════════════════════════════════════════════
-- ROW LEVEL SECURITY (Optional - Enable if using Supabase Auth)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Uncomment these if you want to add user authentication later:

-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Users can manage their own conversations"
--     ON conversations FOR ALL
--     USING (auth.uid() = user_id);

-- CREATE POLICY "Users can manage messages in their conversations"
--     ON messages FOR ALL
--     USING (
--         conversation_id IN (
--             SELECT id FROM conversations WHERE user_id = auth.uid()
--         )
--     );

-- ═══════════════════════════════════════════════════════════════════════════════
-- CLEANUP FUNCTIONS (Optional)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Function to clean up expired query cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM query_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old completed/failed indexing tasks (older than 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_tasks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM indexing_tasks 
    WHERE status IN ('completed', 'failed') 
    AND updated_at < NOW() - INTERVAL '7 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ═══════════════════════════════════════════════════════════════════════════════
-- SAMPLE DATA (Optional - for testing)
-- ═══════════════════════════════════════════════════════════════════════════════

-- Uncomment to insert sample data:

-- INSERT INTO conversations (title) VALUES 
--     ('Getting started with ModalMuse'),
--     ('Document analysis session');

-- INSERT INTO messages (conversation_id, role, content) VALUES
--     ((SELECT id FROM conversations LIMIT 1), 'user', 'What is in this document?'),
--     ((SELECT id FROM conversations LIMIT 1), 'assistant', 'This document contains information about...');

-- ═══════════════════════════════════════════════════════════════════════════════
-- VERIFICATION
-- ═══════════════════════════════════════════════════════════════════════════════

-- Run these to verify tables were created:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
