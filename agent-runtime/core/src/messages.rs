
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Core message types for agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    /// Task assignment from orchestrator to agent
    TaskAssignment {
        task_id: String,
        spec: TaskSpec,
        priority: Priority,
    },
    /// Task completion notification
    TaskComplete {
        task_id: String,
        result: TaskResult,
        metrics: TaskMetrics,
    },
    /// Event data enriched with context
    EnrichedEvent(EnrichedEvent),
    /// EVI scoring envelope
    EVIEnvelope(EVIEnvelope),
    /// Brief generation request
    BriefRequest {
        events: Vec<EnrichedEvent>,
        context: BriefContext,
    },
    /// Generated brief
    Brief(Brief),
    /// Agent status update
    StatusUpdate {
        agent_id: String,
        status: AgentStatus,
        metrics: HashMap<String, f64>,
    },
    /// Resource budget update
    BudgetUpdate {
        agent_id: String,
        remaining: ResourceBudget,
    },
    /// Collaboration request between agents
    CollaborationRequest {
        from_agent: String,
        to_agent: String,
        task_id: String,
        data: serde_json::Value,
    },
    /// Shutdown signal
    Shutdown {
        reason: ShutdownReason,
        graceful: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub id: String,
    pub mission_id: Option<String>,
    pub task_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub dependencies: Vec<String>,
    pub resource_limits: ResourceBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub success: bool,
    pub data: serde_json::Value,
    pub error: Option<String>,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    pub duration_ms: u64,
    pub tokens_used: u64,
    pub api_calls: u32,
    pub bytes_processed: u64,
    pub quality_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedEvent {
    pub event_id: String,
    pub mission_id: Option<String>,
    pub timestamp: i64,
    pub domain: Domain,
    pub source: String,
    pub topic_hints: Vec<String>,
    pub payload: EventPayload,
    pub metadata: EventMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayload {
    pub raw: serde_json::Value,
    pub extracted: ExtractedData,
    pub embeddings: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedData {
    pub entities: Vec<Entity>,
    pub metrics: HashMap<String, f64>,
    pub sentiment: Option<f32>,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: EntityType,
    pub confidence: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Token,
    Protocol,
    Person,
    Organization,
    Repository,
    Paper,
    Address,
    Transaction,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    pub content_hash: String,
    pub geo_location: Option<GeoLocation>,
    pub language: String,
    pub processing_time_ms: u64,
    pub data_sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub country: Option<String>,
    pub city: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVIEnvelope {
    pub event_id: String,
    pub scores: EVIScores,
    pub total_evi: f32,
    pub band: Band,
    pub explanations: HashMap<String, String>,
    pub metadata: EVIMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVIScores {
    pub freshness: f32,
    pub novelty: f32,
    pub impact: f32,
    pub confidence: f32,
    pub gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVIMetadata {
    pub calculated_at: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub weights: EVIWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVIWeights {
    pub freshness: f32,
    pub novelty: f32,
    pub impact: f32,
    pub confidence: f32,
    pub gap: f32,
}

impl Default for EVIWeights {
    fn default() -> Self {
        Self {
            freshness: 0.25,
            novelty: 0.25,
            impact: 0.30,
            confidence: 0.15,
            gap: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Band {
    Instant,   // EVI >= 0.8
    SameDay,   // EVI >= 0.6
    Weekly,    // EVI >= 0.3
    Archive,   // EVI < 0.3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brief {
    pub brief_id: String,
    pub mission_id: Option<String>,
    pub headline: String,
    pub content: BriefContent,
    pub event_ids: Vec<String>,
    pub impact_level: ImpactLevel,
    pub confidence_score: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefContent {
    pub summary: String,
    pub key_findings: Vec<String>,
    pub evidence: Vec<Evidence>,
    pub suggested_actions: Vec<String>,
    pub risk_assessment: Option<RiskAssessment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub text: String,
    pub confidence: f32,
    pub source: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub level: RiskLevel,
    pub factors: Vec<String>,
    pub mitigation: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefContext {
    pub mission_objectives: Vec<String>,
    pub user_preferences: HashMap<String, serde_json::Value>,
    pub historical_briefs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub artifact_type: ArtifactType,
    pub data: serde_json::Value,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    Chart,
    Table,
    Image,
    Document,
    Model,
    Dataset,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub tokens: u64,
    pub api_calls: u32,
    pub compute_hours: f32,
    pub memory_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Domain {
    Crypto,
    AI,
    General,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Starting,
    Running,
    Paused,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShutdownReason {
    BudgetExhausted,
    MissionComplete,
    Error(String),
    UserRequested,
    SystemShutdown,
}

impl Message {
    pub fn message_type(&self) -> &'static str {
        match self {
            Message::TaskAssignment { .. } => "task_assignment",
            Message::TaskComplete { .. } => "task_complete",
            Message::EnrichedEvent(_) => "enriched_event",
            Message::EVIEnvelope(_) => "evi_envelope",
            Message::BriefRequest { .. } => "brief_request",
            Message::Brief(_) => "brief",
            Message::StatusUpdate { .. } => "status_update",
            Message::BudgetUpdate { .. } => "budget_update",
            Message::CollaborationRequest { .. } => "collaboration_request",
            Message::Shutdown { .. } => "shutdown",
        }
    }

    pub fn priority(&self) -> Priority {
        match self {
            Message::Shutdown { .. } => Priority::Critical,
            Message::BudgetUpdate { .. } => Priority::High,
            Message::TaskAssignment { priority, .. } => priority.clone(),
            Message::EVIEnvelope(envelope) => match envelope.band {
                Band::Instant => Priority::Critical,
                Band::SameDay => Priority::High,
                Band::Weekly => Priority::Medium,
                Band::Archive => Priority::Low,
            },
            _ => Priority::Medium,
        }
    }
}

/// Message validation and serialization helpers
impl Message {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Message::EnrichedEvent(event) => {
                if event.event_id.is_empty() {
                    return Err("Event ID cannot be empty".to_string());
                }
                if event.timestamp <= 0 {
                    return Err("Invalid timestamp".to_string());
                }
            }
            Message::Brief(brief) => {
                if brief.headline.is_empty() {
                    return Err("Brief headline cannot be empty".to_string());
                }
                if brief.confidence_score < 0.0 || brief.confidence_score > 1.0 {
                    return Err("Confidence score must be between 0 and 1".to_string());
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}
