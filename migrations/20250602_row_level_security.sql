-- Enable RLS on all tenant-scoped tables
ALTER TABLE prowzi.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE prowzi.missions ENABLE ROW LEVEL SECURITY;
ALTER TABLE prowzi.briefs ENABLE ROW LEVEL SECURITY;
ALTER TABLE prowzi.feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE prowzi.users ENABLE ROW LEVEL SECURITY;

-- Create tenant isolation policies
CREATE POLICY tenant_isolation_events ON prowzi.events
  FOR ALL
  USING (tenant_id = current_setting('prowzi.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('prowzi.tenant_id', true));

CREATE POLICY tenant_isolation_missions ON prowzi.missions
  FOR ALL
  USING (tenant_id = current_setting('prowzi.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('prowzi.tenant_id', true));

CREATE POLICY tenant_isolation_briefs ON prowzi.briefs
  FOR ALL
  USING (tenant_id = current_setting('prowzi.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('prowzi.tenant_id', true));

CREATE POLICY tenant_isolation_feedback ON prowzi.feedback
  FOR ALL
  USING (tenant_id = current_setting('prowzi.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('prowzi.tenant_id', true));

CREATE POLICY tenant_isolation_users ON prowzi.users
  FOR ALL
  USING (tenant_id = current_setting('prowzi.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('prowzi.tenant_id', true));

-- Create cross-tenant read policy for shared data
CREATE POLICY shared_data_read ON prowzi.events
  FOR SELECT
  USING (
    tenant_id = current_setting('prowzi.tenant_id', true)
    OR (
      tenant_id = 'shared'
      AND current_setting('prowzi.user_tier', true) IN ('pro', 'elite', 'enterprise')
    )
  );

-- Create admin bypass policy
CREATE POLICY admin_bypass_all ON prowzi.events
  FOR ALL
  USING (current_setting('prowzi.user_role', true) = 'admin');

-- Function to safely set tenant context
CREATE OR REPLACE FUNCTION prowzi.set_tenant_context(
  p_tenant_id TEXT,
  p_user_tier TEXT DEFAULT 'free',
  p_user_role TEXT DEFAULT 'user'
) RETURNS VOID AS $$
BEGIN
  PERFORM set_config('prowzi.tenant_id', p_tenant_id, true);
  PERFORM set_config('prowzi.user_tier', p_user_tier, true);
  PERFORM set_config('prowzi.user_role', p_user_role, true);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Test helper to verify isolation
CREATE OR REPLACE FUNCTION prowzi.test_tenant_isolation(
  p_tenant1 TEXT,
  p_tenant2 TEXT
) RETURNS TABLE(test_name TEXT, passed BOOLEAN, details TEXT) AS $$
BEGIN
  -- Test 1: Can't see other tenant's data
  PERFORM prowzi.set_tenant_context(p_tenant1);
  RETURN QUERY
  SELECT 
    'cross_tenant_read_blocked'::TEXT,
    NOT EXISTS(SELECT 1 FROM prowzi.events WHERE tenant_id = p_tenant2),
    'Tenant ' || p_tenant1 || ' cannot see tenant ' || p_tenant2 || ' data';

  -- Test 2: Can see own data
  RETURN QUERY
  SELECT
    'own_tenant_read_allowed'::TEXT,
    EXISTS(SELECT 1 FROM prowzi.events WHERE tenant_id = p_tenant1),
    'Tenant ' || p_tenant1 || ' can see own data';

  -- Test 3: Can't insert into other tenant
  BEGIN
    INSERT INTO prowzi.events (tenant_id, event_id) VALUES (p_tenant2, gen_random_uuid());
    RETURN QUERY SELECT 'cross_tenant_write_blocked'::TEXT, FALSE, 'Should not be able to insert';
  EXCEPTION WHEN others THEN
    RETURN QUERY SELECT 'cross_tenant_write_blocked'::TEXT, TRUE, 'Insert correctly blocked';
  END;
END;
$$ LANGUAGE plpgsql;
