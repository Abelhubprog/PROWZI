use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorConfig {
    pub actor_id: String,
    pub actor_type: String,
    pub mission_id: Option<String>,
    pub config: HashMap<String, serde_json::Value>,
}

impl ActorConfig {
    pub fn get(&self, key: &str) -> Option<String> {
        self.config.get(key).and_then(|v| v.as_str().map(String::from))
    }
}

#[derive(Debug, Clone)]
pub struct ActorContext {
    pub actor_id: String,
    pub mission_id: Option<String>,
    pub sender: mpsc::Sender<Message>,
    pub budget: Arc<RwLock<Budget>>,
    pub metrics: Arc<RwLock<ActorMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub from: String,
    pub to: String,
    pub msg_type: String,
    pub payload: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    pub mission_id: Option<String>,
}

#[derive(Debug, Default)]
pub struct Budget {
    pub cpu_ms: u64,
    pub memory_mb: u64,
    pub tokens: u64,
    pub bandwidth_mb: u64,
}

#[derive(Debug, Default)]
pub struct ActorMetrics {
    pub messages_processed: u64,
    pub errors: u64,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub tokens_consumed: u64,
    pub cpu_time_ms: u64,
}

#[derive(Debug)]
pub enum ShutdownReason {
    BudgetExhausted,
    MissionComplete,
    Error(String),
    UserRequested,
    SystemShutdown,
}

/// Core trait that all Prowzi agents must implement
#[async_trait::async_trait]
pub trait Actor: Send + Sync {
    /// Initialize the actor with configuration and context
    async fn init(&mut self, config: ActorConfig, ctx: ActorContext) -> Result<(), ActorError>;

    /// Handle incoming messages
    async fn handle(&mut self, msg: Message, ctx: &mut ActorContext) -> Result<(), ActorError>;

    /// Called every second for periodic tasks
    async fn tick(&mut self, ctx: &mut ActorContext) -> Result<(), ActorError>;

    /// Gracefully shutdown the actor
    async fn shutdown(&mut self, reason: ShutdownReason) -> Result<(), ActorError>;

    /// Get actor type identifier
    fn actor_type(&self) -> &str;

    /// Get actor capabilities
    fn capabilities(&self) -> Vec<String>;
}

#[derive(Debug, thiserror::Error)]
pub enum ActorError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Budget exhausted: {resource}")]
    BudgetExhausted { resource: String },

    #[error("Message handling error: {0}")]
    MessageHandling(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl ActorContext {
    pub async fn send_message(&self, to: &str, msg_type: &str, payload: serde_json::Value) -> Result<(), ActorError> {
        let message = Message {
            id: Uuid::new_v4().to_string(),
            from: self.actor_id.clone(),
            to: to.to_string(),
            msg_type: msg_type.to_string(),
            payload,
            timestamp: Utc::now(),
            mission_id: self.mission_id.clone(),
        };

        self.sender.send(message).await
            .map_err(|e| ActorError::MessageHandling(format!("Failed to send message: {}", e)))?;

        Ok(())
    }

    pub async fn consume_tokens(&self, amount: u64) -> Result<(), ActorError> {
        let mut budget = self.budget.write().await;
        if budget.tokens < amount {
            return Err(ActorError::BudgetExhausted { resource: "tokens".to_string() });
        }
        budget.tokens -= amount;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.tokens_consumed += amount;

        Ok(())
    }

    pub async fn record_heartbeat(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.last_heartbeat = Some(Utc::now());
    }

    pub async fn record_error(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.errors += 1;
    }

    pub async fn publish(&self, topic: &str, data: impl Serialize) -> Result<(), ActorError> {
        let payload = serde_json::to_value(data)?;
        self.send_message(topic, "publish", payload).await
    }
}