-- Create custom aggregation functions for time-series
CREATE OR REPLACE FUNCTION first_value_agg(anyelement, anyelement)
RETURNS anyelement
LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT $1;
$$;

CREATE AGGREGATE first(anyelement) (
    SFUNC = first_value_agg,
    STYPE = anyelement
);

-- Optimized continuous aggregate for brief generation metrics
CREATE MATERIALIZED VIEW brief_generation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', created_at) AS minute,
    domain,
    source,
    COUNT(*) AS event_count,
    COUNT(DISTINCT mission_id) AS active_missions,
    AVG(EXTRACT(EPOCH FROM (brief_generated_at - created_at))) AS avg_latency_seconds,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY 
        EXTRACT(EPOCH FROM (brief_generated_at - created_at))
    ) AS p95_latency_seconds,
    SUM(CASE WHEN impact_level = 'critical' THEN 1 ELSE 0 END) AS critical_briefs,
    AVG((evi_scores->>'total')::float) AS avg_evi_score
FROM events e
LEFT JOIN briefs b ON e.event_id = ANY(b.event_ids)
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY minute, domain, source
WITH NO DATA;

-- Create custom index for complex queries
CREATE INDEX idx_events_composite ON events (
    domain,
    source,
    created_at DESC,
    (evi_scores->>'total')::float DESC
) WHERE (evi_scores->>'total')::float > 0.7;

-- Function for efficient mission performance analysis
CREATE OR REPLACE FUNCTION analyze_mission_performance(
    p_mission_id UUID,
    p_start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
    p_end_time TIMESTAMPTZ DEFAULT NOW()
) RETURNS TABLE (
    metric_name TEXT,
    metric_value NUMERIC,
    comparison_to_baseline NUMERIC,
    percentile_rank NUMERIC
) AS $$
DECLARE
    v_baseline RECORD;
BEGIN
    -- Get baseline metrics for similar missions
    SELECT 
        AVG(total_events) AS avg_events,
        AVG(unique_findings) AS avg_findings,
        AVG(total_tokens) AS avg_tokens,
        AVG(duration_hours) AS avg_duration
    INTO v_baseline
    FROM mission_summaries
    WHERE mission_type = (
        SELECT config->>'type' 
        FROM missions 
        WHERE id = p_mission_id
    )
    AND completed_at > NOW() - INTERVAL '30 days';

    -- Return comparative analysis
    RETURN QUERY
    WITH mission_metrics AS (
        SELECT
            COUNT(*) AS total_events,
            COUNT(DISTINCT payload->>'hash') AS unique_findings,
            SUM((resource_usage->>'tokens')::int) AS total_tokens,
            EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 3600 AS duration_hours,
            AVG((evi_scores->>'total')::float) AS avg_quality
        FROM events
        WHERE mission_id = p_mission_id
        AND created_at BETWEEN p_start_time AND p_end_time
    )
    SELECT 
        'Total Events Processed'::TEXT,
        total_events::NUMERIC,
        ROUND(((total_events - v_baseline.avg_events) / v_baseline.avg_events * 100)::numeric, 2),
        PERCENT_RANK() OVER (ORDER BY total_events)::NUMERIC
    FROM mission_metrics

    UNION ALL

    SELECT 
        'Unique Findings'::TEXT,
        unique_findings::NUMERIC,
        ROUND(((unique_findings - v_baseline.avg_findings) / v_baseline.avg_findings * 100)::numeric, 2),
        PERCENT_RANK() OVER (ORDER BY unique_findings)::NUMERIC
    FROM mission_metrics

    -- Add more metrics as needed
    ;
END;
$$ LANGUAGE plpgsql;
