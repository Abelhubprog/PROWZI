//! Prowzi Memory System
//!
//! Advanced memory architecture for autonomous AI agents with multi-layer hierarchy,
//! knowledge graphs, and collective intelligence capabilities.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use uuid::Uuid;
use tracing::{debug, info, warn, error, instrument};

use crate::{
    actor::{ActorContext, ActorError},
    messages::{Message, EnrichedEvent, EVIEnvelope, Brief, TaskResult, TaskMetrics},
    budget::Budget,
    performance::PerformanceMetrics,
};

/// Core memory system for Prowzi agents
#[derive(Debug, Clone)]
pub struct AgentMemorySystem {
    /// Fast short-term memory for active processing
    working_memory: Arc<RwLock<WorkingMemory>>,
    /// Medium-term episodic memory for experiences
    episodic_memory: Arc<RwLock<EpisodicMemory>>,
    /// Long-term semantic memory for patterns and knowledge
    semantic_memory: Arc<RwLock<SemanticMemory>>,
    /// Shared collective memory across all agents
    collective_memory: Arc<CollectiveMemory>,
    /// Memory consolidation engine
    consolidation_engine: Arc<ConsolidationEngine>,
    /// Memory configuration
    config: MemoryConfig,
}

/// Memory configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub working_memory_capacity: usize,
    pub working_memory_decay_rate: f64,
    pub episodic_retention_days: i64,
    pub semantic_confidence_threshold: f64,
    pub consolidation_interval_sec: u64,
    pub enable_collective_sharing: bool,
    pub vector_dimensions: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            working_memory_capacity: 1000,
            working_memory_decay_rate: 0.1,
            episodic_retention_days: 30,
            semantic_confidence_threshold: 0.7,
            consolidation_interval_sec: 300, // 5 minutes
            enable_collective_sharing: true,
            vector_dimensions: 768,
        }
    }
}

/// Unique identifier for memory items
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Core memory content structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContent {
    pub content_type: MemoryContentType,
    pub data: serde_json::Value,
    pub context: MemoryContext,
    pub embeddings: Option<Vec<f32>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of memory content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryContentType {
    TradingExperience {
        market_context: MarketContext,
        actions_taken: Vec<TradingAction>,
        outcomes: Vec<TradingOutcome>,
        performance_score: f64,
    },
    MarketPattern {
        pattern_type: PatternType,
        indicators: Vec<MarketIndicator>,
        confidence: f64,
        success_rate: f64,
    },
    RiskEvent {
        event_type: RiskEventType,
        severity: RiskSeverity,
        mitigation_actions: Vec<String>,
        lessons_learned: Vec<String>,
    },
    Insight {
        insight_type: InsightType,
        content: String,
        confidence: f64,
        source_events: Vec<String>,
    },
    Communication {
        from_agent: String,
        to_agent: String,
        message_type: String,
        content: serde_json::Value,
        success: bool,
    },
}

/// Memory context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub agent_id: String,
    pub mission_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub market_conditions: Option<MarketConditions>,
    pub performance_state: Option<PerformanceState>,
    pub tags: Vec<String>,
}

/// Market context for trading experiences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub symbol: String,
    pub price: f64,
    pub volume_24h: f64,
    pub volatility: f64,
    pub market_cap: Option<f64>,
    pub sentiment_score: Option<f64>,
}

/// Trading actions taken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAction {
    pub action_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
}

/// Trading outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingOutcome {
    pub outcome_type: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub success: bool,
}

/// Pattern types for market analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    PriceMovement,
    VolumePattern,
    SentimentShift,
    ArbitrageOpportunity,
    LiquidityPattern,
    Custom(String),
}

/// Market indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketIndicator {
    pub name: String,
    pub value: f64,
    pub timeframe: String,
    pub weight: f64,
}

/// Risk event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEventType {
    PositionLoss,
    MarketCrash,
    LiquidityDrop,
    SystemFailure,
    SecurityBreach,
    ExternalShock,
}

/// Risk severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Insight types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    MarketTrend,
    RiskAlert,
    OpportunityDetection,
    StrategyOptimization,
    PerformanceImprovement,
}

/// Current market conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub overall_sentiment: f64,
    pub volatility_index: f64,
    pub liquidity_score: f64,
    pub market_phase: MarketPhase,
}

/// Market phases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketPhase {
    Bull,
    Bear,
    Sideways,
    Volatile,
    Unknown,
}

/// Performance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceState {
    pub recent_success_rate: f64,
    pub average_latency_ms: f64,
    pub resource_utilization: f64,
    pub error_rate: f64,
}

impl AgentMemorySystem {
    /// Create a new memory system instance
    pub async fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        Ok(Self {
            working_memory: Arc::new(RwLock::new(WorkingMemory::new(config.working_memory_capacity, config.working_memory_decay_rate))),
            episodic_memory: Arc::new(RwLock::new(EpisodicMemory::new(config.episodic_retention_days))),
            semantic_memory: Arc::new(RwLock::new(SemanticMemory::new(config.semantic_confidence_threshold))),
            collective_memory: Arc::new(CollectiveMemory::new(config.enable_collective_sharing).await?),
            consolidation_engine: Arc::new(ConsolidationEngine::new(config.consolidation_interval_sec)),
            config,
        })
    }

    /// Store a new experience in memory
    #[instrument(level = "debug", skip(self))]
    pub async fn store_experience(&self, content: MemoryContent) -> Result<MemoryId, MemoryError> {
        let memory_id = MemoryId::new();
        
        // Store in working memory first
        self.working_memory.write().await.store(memory_id.clone(), content.clone()).await?;
        
        // Calculate importance score
        let importance = self.calculate_importance(&content).await?;
        
        // Schedule consolidation if important enough
        if importance > 0.6 {
            self.consolidation_engine
                .schedule_consolidation(memory_id.clone(), content.clone(), importance)
                .await?;
        }
        
        // Share with collective memory if highly significant
        if importance > 0.8 && self.config.enable_collective_sharing {
            let insight = self.extract_insight_from_content(&content).await?;
            self.collective_memory.share_insight(insight).await?;
        }
        
        info!("Stored experience {} with importance score {:.2}", memory_id.0, importance);
        Ok(memory_id)
    }

    /// Retrieve relevant memories for a given context
    #[instrument(level = "debug", skip(self))]
    pub async fn retrieve_relevant_memories(&self, context: &MemoryContext, limit: usize) -> Result<Vec<RetrievedMemory>, MemoryError> {
        let mut all_memories = Vec::new();
        
        // Search working memory
        let working_results = self.working_memory.read().await
            .search_by_context(context, limit).await?;
        all_memories.extend(working_results);
        
        // Search episodic memory
        let episodic_results = self.episodic_memory.read().await
            .search_similar(context, limit).await?;
        all_memories.extend(episodic_results);
        
        // Search semantic memory
        let semantic_results = self.semantic_memory.read().await
            .query_knowledge(context, limit).await?;
        all_memories.extend(semantic_results);
        
        // Rank by relevance and recency
        all_memories.sort_by(|a, b| {
            let score_a = a.relevance_score * self.recency_weight(a.timestamp);
            let score_b = b.relevance_score * self.recency_weight(b.timestamp);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(all_memories.into_iter().take(limit).collect())
    }

    /// Update memory based on feedback
    #[instrument(level = "debug", skip(self))]
    pub async fn update_from_feedback(&self, memory_id: &MemoryId, feedback: MemoryFeedback) -> Result<(), MemoryError> {
        // Update working memory if present
        if let Some(mut memory_item) = self.working_memory.write().await.get_mut(memory_id) {
            memory_item.apply_feedback(&feedback);
        }
        
        // Update episodic memory
        self.episodic_memory.write().await.update_memory(memory_id, &feedback).await?;
        
        // Update semantic patterns based on feedback
        self.semantic_memory.write().await.update_patterns(&feedback).await?;
        
        debug!("Applied feedback to memory {}: {:?}", memory_id.0, feedback);
        Ok(())
    }

    /// Process enriched events into memory
    #[instrument(level = "debug", skip(self))]
    pub async fn process_enriched_event(&self, event: &EnrichedEvent) -> Result<MemoryId, MemoryError> {
        let context = MemoryContext {
            agent_id: "system".to_string(), // Could be extracted from event metadata
            mission_id: event.mission_id.clone(),
            timestamp: Utc::now(),
            market_conditions: self.extract_market_conditions(event),
            performance_state: None,
            tags: event.topic_hints.clone(),
        };

        let content = MemoryContent {
            content_type: self.categorize_event_content(event),
            data: event.payload.raw.clone(),
            context,
            embeddings: event.payload.embeddings.clone(),
            metadata: self.extract_event_metadata(event),
        };

        self.store_experience(content).await
    }

    /// Start background consolidation process
    pub async fn start_consolidation(&self) -> Result<(), MemoryError> {
        let engine = self.consolidation_engine.clone();
        let working_memory = self.working_memory.clone();
        let episodic_memory = self.episodic_memory.clone();
        let semantic_memory = self.semantic_memory.clone();

        tokio::spawn(async move {
            engine.run_consolidation_loop(working_memory, episodic_memory, semantic_memory).await;
        });

        info!("Started memory consolidation background process");
        Ok(())
    }

    /// Get memory system statistics
    pub async fn get_statistics(&self) -> MemoryStatistics {
        let working_stats = self.working_memory.read().await.get_statistics().await;
        let episodic_stats = self.episodic_memory.read().await.get_statistics().await;
        let semantic_stats = self.semantic_memory.read().await.get_statistics().await;
        let collective_stats = self.collective_memory.get_statistics().await;

        MemoryStatistics {
            working_memory: working_stats,
            episodic_memory: episodic_stats,
            semantic_memory: semantic_stats,
            collective_memory: collective_stats,
            total_memories: working_stats.total_items + episodic_stats.total_items + semantic_stats.total_items,
        }
    }

    // Helper methods
    async fn calculate_importance(&self, content: &MemoryContent) -> Result<f64, MemoryError> {
        match &content.content_type {
            MemoryContentType::TradingExperience { performance_score, .. } => {
                Ok(*performance_score)
            }
            MemoryContentType::MarketPattern { confidence, success_rate, .. } => {
                Ok((confidence + success_rate) / 2.0)
            }
            MemoryContentType::RiskEvent { severity, .. } => {
                Ok(match severity {
                    RiskSeverity::Critical => 1.0,
                    RiskSeverity::High => 0.8,
                    RiskSeverity::Medium => 0.6,
                    RiskSeverity::Low => 0.4,
                })
            }
            MemoryContentType::Insight { confidence, .. } => Ok(*confidence),
            MemoryContentType::Communication { success, .. } => {
                Ok(if *success { 0.5 } else { 0.3 })
            }
        }
    }

    fn recency_weight(&self, timestamp: DateTime<Utc>) -> f64 {
        let age_hours = (Utc::now() - timestamp).num_hours() as f64;
        (-age_hours / 168.0).exp() // Decay over a week
    }

    async fn extract_insight_from_content(&self, content: &MemoryContent) -> Result<CollectiveInsight, MemoryError> {
        match &content.content_type {
            MemoryContentType::MarketPattern { pattern_type, confidence, success_rate, .. } => {
                Ok(CollectiveInsight {
                    insight_id: Uuid::new_v4().to_string(),
                    insight_type: InsightType::MarketTrend,
                    content: format!("Pattern detected: {:?} with {:.2}% success rate", pattern_type, success_rate * 100.0),
                    confidence: *confidence,
                    source_agent: content.context.agent_id.clone(),
                    created_at: Utc::now(),
                    validation_count: 0,
                })
            }
            MemoryContentType::TradingExperience { performance_score, .. } if *performance_score > 0.8 => {
                Ok(CollectiveInsight {
                    insight_id: Uuid::new_v4().to_string(),
                    insight_type: InsightType::StrategyOptimization,
                    content: format!("High-performance trading strategy achieved {:.2}% success", performance_score * 100.0),
                    confidence: *performance_score,
                    source_agent: content.context.agent_id.clone(),
                    created_at: Utc::now(),
                    validation_count: 0,
                })
            }
            _ => Err(MemoryError::ExtractionFailed("Cannot extract insight from this content type".to_string()))
        }
    }

    fn extract_market_conditions(&self, event: &EnrichedEvent) -> Option<MarketConditions> {
        // Extract market conditions from event metadata
        event.metadata.get("market_conditions")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    fn categorize_event_content(&self, event: &EnrichedEvent) -> MemoryContentType {
        // Categorize based on event source and content
        if event.source.contains("solana") || event.source.contains("trading") {
            if let Some(sentiment) = event.payload.extracted.sentiment {
                return MemoryContentType::MarketPattern {
                    pattern_type: PatternType::SentimentShift,
                    indicators: vec![MarketIndicator {
                        name: "sentiment".to_string(),
                        value: sentiment as f64,
                        timeframe: "current".to_string(),
                        weight: 1.0,
                    }],
                    confidence: 0.7,
                    success_rate: 0.0, // Will be updated based on outcomes
                };
            }
        }
        
        // Default to insight type
        MemoryContentType::Insight {
            insight_type: InsightType::MarketTrend,
            content: format!("Event from {}: {}", event.source, event.payload.extracted.keywords.join(", ")),
            confidence: 0.5,
            source_events: vec![event.event_id.clone()],
        }
    }

    fn extract_event_metadata(&self, event: &EnrichedEvent) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("event_id".to_string(), serde_json::Value::String(event.event_id.clone()));
        metadata.insert("source".to_string(), serde_json::Value::String(event.source.clone()));
        metadata.insert("domain".to_string(), serde_json::to_value(&event.domain).unwrap_or_default());
        metadata.insert("processing_time_ms".to_string(), serde_json::Value::Number(event.metadata.processing_time_ms.into()));
        metadata
    }
}

/// Memory item retrieved from search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedMemory {
    pub memory_id: MemoryId,
    pub content: MemoryContent,
    pub relevance_score: f64,
    pub timestamp: DateTime<Utc>,
    pub source: MemorySource,
}

/// Source of retrieved memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemorySource {
    Working,
    Episodic,
    Semantic,
    Collective,
}

/// Feedback for memory updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFeedback {
    pub feedback_type: FeedbackType,
    pub value: f64,
    pub context: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Types of feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    SuccessRate,
    PerformanceScore,
    RelevanceRating,
    UsefulnessScore,
    AccuracyRating,
}

/// Memory system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    pub working_memory: WorkingMemoryStats,
    pub episodic_memory: EpisodicMemoryStats,
    pub semantic_memory: SemanticMemoryStats,
    pub collective_memory: CollectiveMemoryStats,
    pub total_memories: usize,
}

/// Working memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryStats {
    pub total_items: usize,
    pub capacity: usize,
    pub average_activation: f64,
    pub eviction_count: u64,
}

/// Episodic memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemoryStats {
    pub total_items: usize,
    pub average_importance: f64,
    pub retention_days: i64,
    pub cleanup_count: u64,
}

/// Semantic memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemoryStats {
    pub total_items: usize,
    pub total_patterns: usize,
    pub average_confidence: f64,
    pub pattern_success_rate: f64,
}

/// Collective memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveMemoryStats {
    pub total_insights: usize,
    pub validated_insights: usize,
    pub validation_rate: f64,
    pub sharing_enabled: bool,
}

/// Collective insight shared across agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveInsight {
    pub insight_id: String,
    pub insight_type: InsightType,
    pub content: String,
    pub confidence: f64,
    pub source_agent: String,
    pub created_at: DateTime<Utc>,
    pub validation_count: u32,
}

/// Memory system errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Retrieval error: {0}")]
    Retrieval(String),
    
    #[error("Consolidation error: {0}")]
    Consolidation(String),
    
    #[error("Extraction failed: {0}")]
    ExtractionFailed(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// Import memory layer implementations
pub mod working_memory;
pub mod episodic_memory;
pub mod semantic_memory;
pub mod collective_memory;
pub mod consolidation;

pub use working_memory::*;
pub use episodic_memory::*;
pub use semantic_memory::*;
pub use collective_memory::*;
pub use consolidation::*;