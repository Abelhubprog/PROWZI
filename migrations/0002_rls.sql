-- Migration: 0002_rls.sql
-- Description: Enables Row Level Security (RLS) for tenant isolation
-- Author: Prowzi Team
-- Date: 2025-06-03

-- Start transaction for atomicity
BEGIN;

-- Function to check if a table exists
CREATE OR REPLACE FUNCTION table_exists(table_name text) RETURNS boolean AS $$
BEGIN
    RETURN EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'prowzi' 
        AND table_name = table_name
    );
END;
$$ LANGUAGE plpgsql;

-- Enable RLS on missions table
DO $$
BEGIN
    IF table_exists('missions') THEN
        -- Enable row level security
        ALTER TABLE prowzi.missions ENABLE ROW LEVEL SECURITY;
        
        -- Create policy for tenant isolation
        DROP POLICY IF EXISTS tenant_isolation_missions ON prowzi.missions;
        CREATE POLICY tenant_isolation_missions ON prowzi.missions
            USING (tenant_id = current_setting('prowzi.tenant', true)::text)
            WITH CHECK (tenant_id = current_setting('prowzi.tenant', true)::text);
            
        RAISE NOTICE 'RLS enabled on missions table';
    ELSE
        RAISE NOTICE 'Skipping missions table - does not exist';
    END IF;
END
$$;

-- Enable RLS on events table
DO $$
BEGIN
    IF table_exists('events') THEN
        -- Enable row level security
        ALTER TABLE prowzi.events ENABLE ROW LEVEL SECURITY;
        
        -- Create policy for tenant isolation
        DROP POLICY IF EXISTS tenant_isolation_events ON prowzi.events;
        CREATE POLICY tenant_isolation_events ON prowzi.events
            USING (tenant_id = current_setting('prowzi.tenant', true)::text)
            WITH CHECK (tenant_id = current_setting('prowzi.tenant', true)::text);
            
        RAISE NOTICE 'RLS enabled on events table';
    ELSE
        RAISE NOTICE 'Skipping events table - does not exist';
    END IF;
END
$$;

-- Enable RLS on briefs table
DO $$
BEGIN
    IF table_exists('briefs') THEN
        -- Enable row level security
        ALTER TABLE prowzi.briefs ENABLE ROW LEVEL SECURITY;
        
        -- Create policy for tenant isolation
        DROP POLICY IF EXISTS tenant_isolation_briefs ON prowzi.briefs;
        CREATE POLICY tenant_isolation_briefs ON prowzi.briefs
            USING (tenant_id = current_setting('prowzi.tenant', true)::text)
            WITH CHECK (tenant_id = current_setting('prowzi.tenant', true)::text);
            
        RAISE NOTICE 'RLS enabled on briefs table';
    ELSE
        RAISE NOTICE 'Skipping briefs table - does not exist';
    END IF;
END
$$;

-- Enable RLS on feedback table
DO $$
BEGIN
    IF table_exists('feedback') THEN
        -- Enable row level security
        ALTER TABLE prowzi.feedback ENABLE ROW LEVEL SECURITY;
        
        -- Create policy for tenant isolation
        DROP POLICY IF EXISTS tenant_isolation_feedback ON prowzi.feedback;
        CREATE POLICY tenant_isolation_feedback ON prowzi.feedback
            USING (tenant_id = current_setting('prowzi.tenant', true)::text)
            WITH CHECK (tenant_id = current_setting('prowzi.tenant', true)::text);
            
        RAISE NOTICE 'RLS enabled on feedback table';
    ELSE
        RAISE NOTICE 'Skipping feedback table - does not exist';
    END IF;
END
$$;

-- Create helper function to set tenant context
CREATE OR REPLACE FUNCTION prowzi.set_tenant_context(p_tenant_id text) RETURNS void AS $$
BEGIN
    PERFORM set_config('prowzi.tenant', p_tenant_id, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMIT;

-- Rollback commands (to be run manually if needed)
/*
BEGIN;

-- Disable RLS on all tables
ALTER TABLE IF EXISTS prowzi.missions DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS prowzi.events DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS prowzi.briefs DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS prowzi.feedback DISABLE ROW LEVEL SECURITY;

-- Drop policies
DROP POLICY IF EXISTS tenant_isolation_missions ON prowzi.missions;
DROP POLICY IF EXISTS tenant_isolation_events ON prowzi.events;
DROP POLICY IF EXISTS tenant_isolation_briefs ON prowzi.briefs;
DROP POLICY IF EXISTS tenant_isolation_feedback ON prowzi.feedback;

-- Drop helper function
DROP FUNCTION IF EXISTS prowzi.set_tenant_context(text);
DROP FUNCTION IF EXISTS table_exists(text);

COMMIT;
*/
