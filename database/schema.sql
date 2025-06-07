
-- Prowzi Database Schema
-- Complete implementation with all required tables and relationships

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom types
CREATE TYPE mission_status AS ENUM ('planning', 'active', 'paused', 'completed', 'failed');
CREATE TYPE agent_status AS ENUM ('starting', 'running', 'paused', 'stopping', 'stopped', 'failed');
CREATE TYPE domain_type AS ENUM ('crypto', 'ai');
CREATE TYPE impact_level AS ENUM ('critical', 'high', 'medium', 'low');
CREATE TYPE band_type AS ENUM ('instant', 'same_day', 'weekly', 'archive');

-- Missions table
CREATE TABLE missions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    status mission_status DEFAULT 'planning',
    plan JSONB NOT NULL DEFAULT '{}',
    config JSONB NOT NULL DEFAULT '{}',
    resource_usage JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    user_id UUID,
    tenant_id VARCHAR(100)
);

-- Agents table
CREATE TABLE agents (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status agent_status DEFAULT 'starting',
    mission_id UUID REFERENCES missions(id) ON DELETE CASCADE,
    config JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_heartbeat TIMESTAMPTZ,
    resource_usage JSONB DEFAULT '{}'
);

-- Events table (partitioned by created_at)
CREATE TABLE events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mission_id UUID REFERENCES missions(id) ON DELETE SET NULL,
    domain domain_type NOT NULL,
    source VARCHAR(100) NOT NULL,
    topic_hints TEXT[] DEFAULT '{}',
    payload JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    evi_scores JSONB,
    band band_type,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
) PARTITION BY RANGE (created_at);

-- Create partitions for events table
CREATE TABLE events_2024 PARTITION OF events 
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE events_2025 PARTITION OF events 
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Briefs table
CREATE TABLE briefs (
    brief_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mission_id UUID REFERENCES missions(id) ON DELETE SET NULL,
    headline VARCHAR(500) NOT NULL,
    content JSONB NOT NULL,
    event_ids UUID[] DEFAULT '{}',
    impact_level impact_level NOT NULL,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    user_feedback JSONB DEFAULT '{}'
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE,
    wallet_address VARCHAR(255) UNIQUE,
    tier VARCHAR(50) DEFAULT 'free',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ,
    tenant_id VARCHAR(100)
);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    brief_id UUID REFERENCES briefs(brief_id) ON DELETE CASCADE,
    rating VARCHAR(20) NOT NULL CHECK (rating IN ('positive', 'negative', 'neutral')),
    comment TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Budget tracking table
CREATE TABLE budget_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mission_id UUID REFERENCES missions(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL,
    amount_used BIGINT NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- API keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    permissions JSONB DEFAULT '{}',
    rate_limit INTEGER DEFAULT 1000,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);

-- Notification preferences table
CREATE TABLE notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    channel VARCHAR(50) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Mission summaries materialized view
CREATE MATERIALIZED VIEW mission_summaries AS
SELECT 
    m.id,
    m.name,
    m.status,
    m.created_at,
    m.completed_at,
    COUNT(DISTINCT a.id) as agent_count,
    COUNT(DISTINCT e.event_id) as event_count,
    COUNT(DISTINCT b.brief_id) as brief_count,
    COALESCE(SUM((bu.amount_used)::numeric), 0) as total_budget_used,
    EXTRACT(EPOCH FROM (COALESCE(m.completed_at, NOW()) - m.created_at)) / 3600 as duration_hours
FROM missions m
LEFT JOIN agents a ON m.id = a.mission_id
LEFT JOIN events e ON m.id = e.mission_id
LEFT JOIN briefs b ON m.id = b.mission_id
LEFT JOIN budget_usage bu ON m.id = bu.mission_id
GROUP BY m.id, m.name, m.status, m.created_at, m.completed_at;

-- Indexes for performance
CREATE INDEX idx_events_mission_domain ON events(mission_id, domain);
CREATE INDEX idx_events_created_at ON events(created_at DESC);
CREATE INDEX idx_events_evi_score ON events(((evi_scores->>'total_evi')::float) DESC) WHERE evi_scores IS NOT NULL;
CREATE INDEX idx_events_band ON events(band);

CREATE INDEX idx_briefs_mission_impact ON briefs(mission_id, impact_level);
CREATE INDEX idx_briefs_created_at ON briefs(created_at DESC);

CREATE INDEX idx_agents_mission_status ON agents(mission_id, status);
CREATE INDEX idx_agents_type ON agents(type);

CREATE INDEX idx_missions_status ON missions(status);
CREATE INDEX idx_missions_user_tenant ON missions(user_id, tenant_id);

CREATE INDEX idx_feedback_brief_rating ON feedback(brief_id, rating);
CREATE INDEX idx_budget_usage_mission_type ON budget_usage(mission_id, resource_type);

-- Functions for EVI calculation
CREATE OR REPLACE FUNCTION calculate_evi_score(
    freshness FLOAT,
    novelty FLOAT,
    impact FLOAT,
    confidence FLOAT,
    gap FLOAT,
    weights JSONB DEFAULT '{"freshness": 0.25, "novelty": 0.25, "impact": 0.30, "confidence": 0.15, "gap": 0.05}'
)
RETURNS FLOAT AS $$
BEGIN
    RETURN (
        freshness * (weights->>'freshness')::float +
        novelty * (weights->>'novelty')::float +
        impact * (weights->>'impact')::float +
        confidence * (weights->>'confidence')::float +
        gap * (weights->>'gap')::float
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to determine band based on EVI score
CREATE OR REPLACE FUNCTION determine_band(evi_score FLOAT)
RETURNS band_type AS $$
BEGIN
    CASE 
        WHEN evi_score >= 0.8 THEN RETURN 'instant';
        WHEN evi_score >= 0.6 THEN RETURN 'same_day';
        WHEN evi_score >= 0.3 THEN RETURN 'weekly';
        ELSE RETURN 'archive';
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_missions_updated_at 
    BEFORE UPDATE ON missions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at 
    BEFORE UPDATE ON agents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security for multi-tenancy
ALTER TABLE missions ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE briefs ENABLE ROW LEVEL SECURITY;

-- Sample data for testing
INSERT INTO users (email, tier) VALUES 
('admin@prowzi.io', 'enterprise'),
('demo@prowzi.io', 'pro');

INSERT INTO missions (name, prompt, status, plan) VALUES 
('Demo Mission', 'Track Solana token launches', 'active', '{"objectives": [{"id": "obj1", "description": "Monitor mempool", "priority": "high"}]}');

-- Refresh materialized view
REFRESH MATERIALIZED VIEW mission_summaries;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
