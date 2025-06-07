
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedEvent {
    pub event_id: String,
    pub mission_id: Option<String>,
    pub timestamp: i64,
    pub domain: String,
    pub source: String,
    pub topic_hints: Vec<String>,
    pub payload: serde_json::Value,
    pub metadata: EventMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    pub content_hash: String,
    pub language: String,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVIEnvelope {
    pub event_id: String,
    pub scores: EVIScores,
    pub total_evi: f32,
    pub band: String,
    pub explanations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVIScores {
    pub freshness: f32,
    pub novelty: f32,
    pub impact: f32,
    pub confidence: f32,
    pub gap: f32,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct BandThresholds {
    pub instant: f32,
    pub same_day: f32,
    pub weekly: f32,
}

impl Default for BandThresholds {
    fn default() -> Self {
        Self {
            instant: 0.8,
            same_day: 0.6,
            weekly: 0.3,
        }
    }
}

pub struct EvaluatorService {
    weights: Arc<RwLock<EVIWeights>>,
    thresholds: Arc<RwLock<BandThresholds>>,
    processed_events: Arc<RwLock<HashMap<String, Instant>>>,
    metrics: Arc<RwLock<EvaluatorMetrics>>,
}

#[derive(Debug, Default)]
pub struct EvaluatorMetrics {
    pub total_processed: u64,
    pub instant_band: u64,
    pub same_day_band: u64,
    pub weekly_band: u64,
    pub archive_band: u64,
    pub avg_processing_time: f64,
}

impl EvaluatorService {
    pub fn new() -> Self {
        Self {
            weights: Arc::new(RwLock::new(EVIWeights::default())),
            thresholds: Arc::new(RwLock::new(BandThresholds::default())),
            processed_events: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(EvaluatorMetrics::default())),
        }
    }

    pub async fn calculate_evi(&self, event: &EnrichedEvent) -> EVIEnvelope {
        let start_time = Instant::now();
        
        // Calculate individual scores
        let scores = EVIScores {
            freshness: self.calculate_freshness(event).await,
            novelty: self.calculate_novelty(event).await,
            impact: self.calculate_impact(event).await,
            confidence: self.calculate_confidence(event).await,
            gap: self.calculate_gap(event).await,
        };

        // Calculate weighted total
        let weights = self.weights.read().await;
        let total_evi = scores.freshness * weights.freshness
            + scores.novelty * weights.novelty
            + scores.impact * weights.impact
            + scores.confidence * weights.confidence
            + scores.gap * weights.gap;

        // Determine band
        let thresholds = self.thresholds.read().await;
        let band = if total_evi >= thresholds.instant {
            "instant"
        } else if total_evi >= thresholds.same_day {
            "same_day"
        } else if total_evi >= thresholds.weekly {
            "weekly"
        } else {
            "archive"
        };

        // Generate explanations
        let explanations = self.generate_explanations(&scores, event);

        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        let mut metrics = self.metrics.write().await;
        metrics.total_processed += 1;
        match band {
            "instant" => metrics.instant_band += 1,
            "same_day" => metrics.same_day_band += 1,
            "weekly" => metrics.weekly_band += 1,
            _ => metrics.archive_band += 1,
        }
        metrics.avg_processing_time = 
            (metrics.avg_processing_time * (metrics.total_processed - 1) as f64 + processing_time)
            / metrics.total_processed as f64;

        EVIEnvelope {
            event_id: event.event_id.clone(),
            scores,
            total_evi,
            band: band.to_string(),
            explanations,
        }
    }

    async fn calculate_freshness(&self, event: &EnrichedEvent) -> f32 {
        let now = chrono::Utc::now().timestamp_millis();
        let age_ms = now - event.timestamp;
        let age_hours = age_ms as f32 / (1000.0 * 60.0 * 60.0);

        // Exponential decay with 24-hour half-life
        let half_life_hours = 24.0;
        let decay_factor = -(age_hours / half_life_hours).ln();
        (2.0_f32).powf(-age_hours / half_life_hours).min(1.0).max(0.0)
    }

    async fn calculate_novelty(&self, event: &EnrichedEvent) -> f32 {
        // Check if we've seen similar content recently
        let processed = self.processed_events.read().await;
        
        // Simple novelty based on content hash uniqueness
        if processed.contains_key(&event.metadata.content_hash) {
            0.1 // Low novelty for duplicate content
        } else {
            // Check for similar events in the same domain/source
            let similar_count = processed.keys()
                .filter(|key| key.starts_with(&format!("{}:{}", event.domain, event.source)))
                .count();
            
            // Diminishing novelty based on frequency
            if similar_count == 0 {
                1.0 // Completely novel
            } else {
                (1.0 / (1.0 + similar_count as f32 * 0.1)).max(0.1)
            }
        }
    }

    async fn calculate_impact(&self, event: &EnrichedEvent) -> f32 {
        // Domain-specific impact calculation
        match event.domain.as_str() {
            "crypto" => self.calculate_crypto_impact(event).await,
            "ai" => self.calculate_ai_impact(event).await,
            _ => 0.5, // Default moderate impact
        }
    }

    async fn calculate_crypto_impact(&self, event: &EnrichedEvent) -> f32 {
        let mut impact = 0.5;

        // Check for high-impact crypto events
        if event.topic_hints.contains(&"token_launch".to_string()) {
            impact += 0.3;
        }
        if event.topic_hints.contains(&"whale_movement".to_string()) {
            impact += 0.4;
        }
        if event.topic_hints.contains(&"exploit".to_string()) {
            impact += 0.5;
        }

        // Check payload for value indicators
        if let Some(payload) = event.payload.as_object() {
            if let Some(amount) = payload.get("amount").and_then(|v| v.as_f64()) {
                if amount > 1_000_000.0 {
                    impact += 0.2;
                }
            }
        }

        impact.min(1.0)
    }

    async fn calculate_ai_impact(&self, event: &EnrichedEvent) -> f32 {
        let mut impact = 0.5;

        // Check for high-impact AI events
        if event.topic_hints.contains(&"model_release".to_string()) {
            impact += 0.3;
        }
        if event.topic_hints.contains(&"breakthrough".to_string()) {
            impact += 0.4;
        }
        if event.topic_hints.contains(&"security_vulnerability".to_string()) {
            impact += 0.5;
        }

        // Check for GitHub star velocity
        if event.source == "github_events" {
            if let Some(payload) = event.payload.as_object() {
                if let Some(stars) = payload.get("star_count").and_then(|v| v.as_u64()) {
                    if stars > 1000 {
                        impact += 0.2;
                    }
                }
            }
        }

        impact.min(1.0)
    }

    async fn calculate_confidence(&self, event: &EnrichedEvent) -> f32 {
        let mut confidence = 0.5;

        // Source reliability
        match event.source.as_str() {
            "solana_mempool" | "ethereum_mempool" => confidence += 0.3,
            "github_events" => confidence += 0.2,
            "arxiv" => confidence += 0.4,
            "twitter" => confidence += 0.1,
            _ => confidence += 0.0,
        }

        // Data completeness
        if let Some(payload) = event.payload.as_object() {
            let field_count = payload.len();
            confidence += (field_count as f32 / 10.0).min(0.2);
        }

        confidence.min(1.0)
    }

    async fn calculate_gap(&self, event: &EnrichedEvent) -> f32 {
        // Simple gap calculation based on topic coverage
        // In a real implementation, this would check knowledge graph density
        if event.topic_hints.is_empty() {
            0.5
        } else {
            // More specific topics indicate potential gaps
            let specificity = event.topic_hints.len() as f32;
            (0.1 * specificity).min(1.0)
        }
    }

    fn generate_explanations(&self, scores: &EVIScores, event: &EnrichedEvent) -> HashMap<String, String> {
        let mut explanations = HashMap::new();

        explanations.insert(
            "freshness".to_string(),
            format!("Event is {:.1} hours old", 
                (chrono::Utc::now().timestamp_millis() - event.timestamp) as f32 / (1000.0 * 60.0 * 60.0))
        );

        explanations.insert(
            "novelty".to_string(),
            if scores.novelty > 0.8 {
                "Highly novel content".to_string()
            } else if scores.novelty > 0.5 {
                "Moderately novel content".to_string()
            } else {
                "Similar content seen recently".to_string()
            }
        );

        explanations.insert(
            "impact".to_string(),
            format!("Domain: {}, Source: {}", event.domain, event.source)
        );

        explanations.insert(
            "confidence".to_string(),
            format!("Based on source reliability and data completeness")
        );

        explanations
    }

    pub async fn cleanup_old_events(&self) {
        let mut processed = self.processed_events.write().await;
        let now = Instant::now();
        
        // Remove events older than 24 hours
        processed.retain(|_, timestamp| {
            now.duration_since(*timestamp) < Duration::from_secs(24 * 60 * 60)
        });
    }
}

// HTTP handlers
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "evaluator",
        "version": "1.0.0"
    }))
}

async fn get_metrics(State(service): State<Arc<EvaluatorService>>) -> Json<EvaluatorMetrics> {
    let metrics = service.metrics.read().await;
    Json(metrics.clone())
}

async fn evaluate_event(
    State(service): State<Arc<EvaluatorService>>,
    Json(event): Json<EnrichedEvent>,
) -> Json<EVIEnvelope> {
    let envelope = service.calculate_evi(&event).await;
    
    // Store event hash for novelty calculation
    {
        let mut processed = service.processed_events.write().await;
        processed.insert(event.metadata.content_hash.clone(), Instant::now());
    }
    
    Json(envelope)
}

async fn update_weights(
    State(service): State<Arc<EvaluatorService>>,
    Json(new_weights): Json<EVIWeights>,
) -> StatusCode {
    let mut weights = service.weights.write().await;
    *weights = new_weights;
    StatusCode::OK
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    
    let service = Arc::new(EvaluatorService::new());
    
    // Start cleanup task
    let cleanup_service = service.clone();
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(3600)); // Every hour
        loop {
            interval.tick().await;
            cleanup_service.cleanup_old_events().await;
        }
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(get_metrics))
        .route("/evaluate", post(evaluate_event))
        .route("/weights", post(update_weights))
        .with_state(service);

    let addr = "0.0.0.0:8081".parse().unwrap();
    println!("EVI Evaluator listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
