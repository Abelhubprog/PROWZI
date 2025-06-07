-- Migration: 0003_tenant_isolation.sql
-- Description: Complete tenant isolation setup for authentication
-- Author: Prowzi Team
-- Date: 2025-06-03

BEGIN;

-- Ensure tenant_id is not null and has proper constraints
ALTER TABLE users ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE users ALTER COLUMN tenant_id SET DEFAULT 'default';

-- Create tenants table if it doesn't exist
CREATE TABLE IF NOT EXISTS tenants (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(50) DEFAULT 'starter',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Add foreign key constraint
ALTER TABLE users ADD CONSTRAINT fk_users_tenant 
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT;

-- Create default tenant
INSERT INTO tenants (id, name, tier) VALUES ('default', 'Default Tenant', 'starter')
ON CONFLICT (id) DO NOTHING;

-- Enable RLS on users table
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Create tenant isolation policy for users
DROP POLICY IF EXISTS tenant_isolation_users ON users;
CREATE POLICY tenant_isolation_users ON users
    USING (tenant_id = current_setting('prowzi.tenant', true)::text)
    WITH CHECK (tenant_id = current_setting('prowzi.tenant', true)::text);

-- Enable RLS on tenants table
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;

-- Create tenant isolation policy for tenants (users can only see their own tenant)
DROP POLICY IF EXISTS tenant_isolation_tenants ON tenants;
CREATE POLICY tenant_isolation_tenants ON tenants
    USING (id = current_setting('prowzi.tenant', true)::text);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_wallet_address ON users(wallet_address);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Update existing users to have default tenant if null
UPDATE users SET tenant_id = 'default' WHERE tenant_id IS NULL;

-- Create user session tracking table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    jti VARCHAR(255) NOT NULL, -- JWT ID
    token_type VARCHAR(20) NOT NULL CHECK (token_type IN ('access', 'refresh')),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    tenant_id VARCHAR(100) NOT NULL
);

-- Enable RLS on user_sessions
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- Create tenant isolation policy for user_sessions
CREATE POLICY tenant_isolation_user_sessions ON user_sessions
    USING (tenant_id = current_setting('prowzi.tenant', true)::text)
    WITH CHECK (tenant_id = current_setting('prowzi.tenant', true)::text);

-- Create indexes for session management
CREATE INDEX IF NOT EXISTS idx_user_sessions_jti ON user_sessions(jti);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_tenant_id ON user_sessions(tenant_id);

-- Create function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions() RETURNS void AS $$
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < NOW() 
    OR revoked_at IS NOT NULL AND revoked_at < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Create function to revoke user sessions
CREATE OR REPLACE FUNCTION revoke_user_sessions(p_user_id UUID) RETURNS void AS $$
BEGIN
    UPDATE user_sessions 
    SET revoked_at = NOW() 
    WHERE user_id = p_user_id 
    AND revoked_at IS NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMIT;

-- Rollback commands (for reference)
/*
BEGIN;

-- Drop policies
DROP POLICY IF EXISTS tenant_isolation_users ON users;
DROP POLICY IF EXISTS tenant_isolation_tenants ON tenants;
DROP POLICY IF EXISTS tenant_isolation_user_sessions ON user_sessions;

-- Disable RLS
ALTER TABLE users DISABLE ROW LEVEL SECURITY;
ALTER TABLE tenants DISABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions DISABLE ROW LEVEL SECURITY;

-- Drop tables and constraints
DROP TABLE IF EXISTS user_sessions;
ALTER TABLE users DROP CONSTRAINT IF EXISTS fk_users_tenant;
DROP TABLE IF EXISTS tenants;

-- Drop functions
DROP FUNCTION IF EXISTS cleanup_expired_sessions();
DROP FUNCTION IF EXISTS revoke_user_sessions(UUID);

COMMIT;
*/
