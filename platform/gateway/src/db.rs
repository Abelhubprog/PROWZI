
use sqlx::{Pool, Postgres, Row};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Mission {
    pub id: Uuid,
    pub name: String,
    pub prompt: String,
    pub status: String,
    pub plan: serde_json::Value,
    pub config: serde_json::Value,
    pub resource_usage: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub user_id: Option<Uuid>,
    pub tenant_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub status: String,
    pub mission_id: Option<Uuid>,
    pub config: serde_json::Value,
    pub metrics: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub resource_usage: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Event {
    pub event_id: Uuid,
    pub mission_id: Option<Uuid>,
    pub domain: String,
    pub source: String,
    pub topic_hints: Vec<String>,
    pub payload: serde_json::Value,
    pub metadata: serde_json::Value,
    pub evi_scores: Option<serde_json::Value>,
    pub band: Option<String>,
    pub created_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Brief {
    pub brief_id: Uuid,
    pub mission_id: Option<Uuid>,
    pub headline: String,
    pub content: serde_json::Value,
    pub event_ids: Vec<Uuid>,
    pub impact_level: String,
    pub confidence_score: f64,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub user_feedback: serde_json::Value,
}

pub struct Database {
    pool: Pool<Postgres>,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(20)
            .connect(database_url)
            .await?;

        Ok(Database { pool })
    }

    // Mission operations
    pub async fn create_mission(&self, mission: &Mission) -> Result<Mission, sqlx::Error> {
        let row = sqlx::query!(
            r#"
            INSERT INTO missions (id, name, prompt, status, plan, config, resource_usage, user_id, tenant_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, name, prompt, status, plan, config, resource_usage, created_at, updated_at, completed_at, user_id, tenant_id
            "#,
            mission.id,
            mission.name,
            mission.prompt,
            mission.status,
            mission.plan,
            mission.config,
            mission.resource_usage,
            mission.user_id,
            mission.tenant_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(Mission {
            id: row.id,
            name: row.name,
            prompt: row.prompt,
            status: row.status,
            plan: row.plan,
            config: row.config,
            resource_usage: row.resource_usage,
            created_at: row.created_at,
            updated_at: row.updated_at,
            completed_at: row.completed_at,
            user_id: row.user_id,
            tenant_id: row.tenant_id,
        })
    }

    pub async fn get_mission(&self, id: Uuid) -> Result<Option<Mission>, sqlx::Error> {
        let row = sqlx::query!(
            "SELECT * FROM missions WHERE id = $1",
            id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Mission {
            id: r.id,
            name: r.name,
            prompt: r.prompt,
            status: r.status,
            plan: r.plan,
            config: r.config,
            resource_usage: r.resource_usage,
            created_at: r.created_at,
            updated_at: r.updated_at,
            completed_at: r.completed_at,
            user_id: r.user_id,
            tenant_id: r.tenant_id,
        }))
    }

    pub async fn list_missions(&self, user_id: Option<Uuid>, limit: i64, offset: i64) -> Result<Vec<Mission>, sqlx::Error> {
        let rows = match user_id {
            Some(uid) => {
                sqlx::query!(
                    "SELECT * FROM missions WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                    uid,
                    limit,
                    offset
                )
                .fetch_all(&self.pool)
                .await?
            }
            None => {
                sqlx::query!(
                    "SELECT * FROM missions ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit,
                    offset
                )
                .fetch_all(&self.pool)
                .await?
            }
        };

        Ok(rows.into_iter().map(|r| Mission {
            id: r.id,
            name: r.name,
            prompt: r.prompt,
            status: r.status,
            plan: r.plan,
            config: r.config,
            resource_usage: r.resource_usage,
            created_at: r.created_at,
            updated_at: r.updated_at,
            completed_at: r.completed_at,
            user_id: r.user_id,
            tenant_id: r.tenant_id,
        }).collect())
    }

    pub async fn update_mission_status(&self, id: Uuid, status: &str) -> Result<(), sqlx::Error> {
        sqlx::query!(
            "UPDATE missions SET status = $1, updated_at = NOW() WHERE id = $2",
            status,
            id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // Agent operations
    pub async fn create_agent(&self, agent: &Agent) -> Result<Agent, sqlx::Error> {
        let row = sqlx::query!(
            r#"
            INSERT INTO agents (id, name, type, status, mission_id, config, metrics, resource_usage)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, name, type, status, mission_id, config, metrics, created_at, updated_at, last_heartbeat, resource_usage
            "#,
            agent.id,
            agent.name,
            agent.agent_type,
            agent.status,
            agent.mission_id,
            agent.config,
            agent.metrics,
            agent.resource_usage
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(Agent {
            id: row.id,
            name: row.name,
            agent_type: row.type,
            status: row.status,
            mission_id: row.mission_id,
            config: row.config,
            metrics: row.metrics,
            created_at: row.created_at,
            updated_at: row.updated_at,
            last_heartbeat: row.last_heartbeat,
            resource_usage: row.resource_usage,
        })
    }

    pub async fn get_agents_by_mission(&self, mission_id: Uuid) -> Result<Vec<Agent>, sqlx::Error> {
        let rows = sqlx::query!(
            "SELECT * FROM agents WHERE mission_id = $1 ORDER BY created_at",
            mission_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| Agent {
            id: r.id,
            name: r.name,
            agent_type: r.type,
            status: r.status,
            mission_id: r.mission_id,
            config: r.config,
            metrics: r.metrics,
            created_at: r.created_at,
            updated_at: r.updated_at,
            last_heartbeat: r.last_heartbeat,
            resource_usage: r.resource_usage,
        }).collect())
    }

    pub async fn update_agent_heartbeat(&self, agent_id: &str) -> Result<(), sqlx::Error> {
        sqlx::query!(
            "UPDATE agents SET last_heartbeat = NOW(), updated_at = NOW() WHERE id = $1",
            agent_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // Event operations
    pub async fn create_event(&self, event: &Event) -> Result<Event, sqlx::Error> {
        let row = sqlx::query!(
            r#"
            INSERT INTO events (event_id, mission_id, domain, source, topic_hints, payload, metadata, evi_scores, band)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING event_id, mission_id, domain, source, topic_hints, payload, metadata, evi_scores, band, created_at, processed_at
            "#,
            event.event_id,
            event.mission_id,
            event.domain,
            event.source,
            &event.topic_hints,
            event.payload,
            event.metadata,
            event.evi_scores,
            event.band
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(Event {
            event_id: row.event_id,
            mission_id: row.mission_id,
            domain: row.domain,
            source: row.source,
            topic_hints: row.topic_hints,
            payload: row.payload,
            metadata: row.metadata,
            evi_scores: row.evi_scores,
            band: row.band,
            created_at: row.created_at,
            processed_at: row.processed_at,
        })
    }

    pub async fn get_events_by_mission(&self, mission_id: Uuid, limit: i64) -> Result<Vec<Event>, sqlx::Error> {
        let rows = sqlx::query!(
            "SELECT * FROM events WHERE mission_id = $1 ORDER BY created_at DESC LIMIT $2",
            mission_id,
            limit
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| Event {
            event_id: r.event_id,
            mission_id: r.mission_id,
            domain: r.domain,
            source: r.source,
            topic_hints: r.topic_hints,
            payload: r.payload,
            metadata: r.metadata,
            evi_scores: r.evi_scores,
            band: r.band,
            created_at: r.created_at,
            processed_at: r.processed_at,
        }).collect())
    }

    // Brief operations
    pub async fn create_brief(&self, brief: &Brief) -> Result<Brief, sqlx::Error> {
        let row = sqlx::query!(
            r#"
            INSERT INTO briefs (brief_id, mission_id, headline, content, event_ids, impact_level, confidence_score, expires_at, user_feedback)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING brief_id, mission_id, headline, content, event_ids, impact_level, confidence_score, created_at, expires_at, user_feedback
            "#,
            brief.brief_id,
            brief.mission_id,
            brief.headline,
            brief.content,
            &brief.event_ids,
            brief.impact_level,
            brief.confidence_score,
            brief.expires_at,
            brief.user_feedback
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(Brief {
            brief_id: row.brief_id,
            mission_id: row.mission_id,
            headline: row.headline,
            content: row.content,
            event_ids: row.event_ids,
            impact_level: row.impact_level,
            confidence_score: row.confidence_score,
            created_at: row.created_at,
            expires_at: row.expires_at,
            user_feedback: row.user_feedback,
        })
    }

    pub async fn get_briefs_by_mission(&self, mission_id: Uuid, limit: i64) -> Result<Vec<Brief>, sqlx::Error> {
        let rows = sqlx::query!(
            "SELECT * FROM briefs WHERE mission_id = $1 ORDER BY created_at DESC LIMIT $2",
            mission_id,
            limit
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| Brief {
            brief_id: r.brief_id,
            mission_id: r.mission_id,
            headline: r.headline,
            content: r.content,
            event_ids: r.event_ids,
            impact_level: r.impact_level,
            confidence_score: r.confidence_score,
            created_at: r.created_at,
            expires_at: r.expires_at,
            user_feedback: r.user_feedback,
        }).collect())
    }

    // Analytics and metrics
    pub async fn get_mission_summary(&self, mission_id: Uuid) -> Result<Option<serde_json::Value>, sqlx::Error> {
        let row = sqlx::query!(
            "SELECT * FROM mission_summaries WHERE id = $1",
            mission_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| serde_json::json!({
            "id": r.id,
            "name": r.name,
            "status": r.status,
            "created_at": r.created_at,
            "completed_at": r.completed_at,
            "agent_count": r.agent_count,
            "event_count": r.event_count,
            "brief_count": r.brief_count,
            "total_budget_used": r.total_budget_used,
            "duration_hours": r.duration_hours
        })))
    }

    pub async fn get_system_stats(&self) -> Result<serde_json::Value, sqlx::Error> {
        let missions = sqlx::query!("SELECT COUNT(*) as count FROM missions")
            .fetch_one(&self.pool)
            .await?;

        let agents = sqlx::query!("SELECT COUNT(*) as count FROM agents WHERE status = 'running'")
            .fetch_one(&self.pool)
            .await?;

        let events_today = sqlx::query!(
            "SELECT COUNT(*) as count FROM events WHERE created_at >= CURRENT_DATE"
        )
        .fetch_one(&self.pool)
        .await?;

        let briefs_today = sqlx::query!(
            "SELECT COUNT(*) as count FROM briefs WHERE created_at >= CURRENT_DATE"
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(serde_json::json!({
            "total_missions": missions.count,
            "active_agents": agents.count,
            "events_today": events_today.count,
            "briefs_today": briefs_today.count
        }))
    }
}
