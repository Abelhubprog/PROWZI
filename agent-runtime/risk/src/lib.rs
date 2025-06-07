//! # Prowzi Risk Management System
//! 
//! Advanced risk management system for autonomous AI agents with real-time monitoring,
//! circuit breakers, and adaptive protection strategies.

use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use solana_sdk::pubkey::Pubkey;

pub mod config;
pub mod engine;
pub mod models;
pub mod metrics;
pub mod protection;
pub mod circuit_breaker;
pub mod portfolio;
pub mod monitoring;
pub mod neural;
pub mod utils;

// Re-export core types
pub use engine::RiskEngine;
pub use models::*;
pub use protection::ProtectionEngine;
pub use circuit_breaker::CircuitBreaker;
pub use config::RiskConfig;

/// Main Risk Management System API
pub struct RiskManager {
    config: RiskConfig,
    engine: Arc<RiskEngine>,
    protection: Arc<ProtectionEngine>,
    circuit_breaker: Arc<CircuitBreaker>,
    metrics: Arc<metrics::MetricsCollector>,
}

impl RiskManager {
    /// Create a new risk management system
    pub async fn new(config: RiskConfig) -> Result<Self, RiskError> {
        let metrics = Arc::new(metrics::MetricsCollector::new(&config.metrics)?);
        
        let engine = Arc::new(RiskEngine::new(config.clone(), metrics.clone()).await?);
        let protection = Arc::new(ProtectionEngine::new(config.clone(), metrics.clone()).await?);
        let circuit_breaker = Arc::new(CircuitBreaker::new(config.clone(), metrics.clone()).await?);

        Ok(Self {
            config,
            engine,
            protection,
            circuit_breaker,
            metrics,
        })
    }

    /// Assess risk for a trading decision
    pub async fn assess_risk(&self, request: RiskAssessmentRequest) -> Result<RiskAssessment, RiskError> {
        // Circuit breaker check
        if self.circuit_breaker.is_triggered().await {
            return Ok(RiskAssessment {
                decision: RiskDecision::Reject,
                reason: "Circuit breaker activated".to_string(),
                confidence: 1.0,
                metrics: self.get_current_metrics().await?,
                timestamp: Utc::now(),
            });
        }

        // Perform risk assessment
        let assessment = self.engine.assess_risk(request).await?;
        
        // Update metrics
        self.metrics.record_assessment(&assessment).await?;
        
        // Check for circuit breaker triggers
        self.circuit_breaker.check_conditions(&assessment).await?;

        Ok(assessment)
    }

    /// Get real-time risk metrics
    pub async fn get_risk_metrics(&self) -> Result<RiskMetrics, RiskError> {
        self.engine.get_current_metrics().await
    }

    /// Update position information
    pub async fn update_position(&self, update: PositionUpdate) -> Result<(), RiskError> {
        self.engine.update_position(update).await
    }

    /// Get protection recommendations
    pub async fn get_protection_strategy(&self, position: &Position) -> Result<ProtectionStrategy, RiskError> {
        self.protection.generate_strategy(position).await
    }

    /// Emergency shutdown
    pub async fn emergency_shutdown(&self, reason: String) -> Result<(), RiskError> {
        tracing::warn!("Emergency shutdown triggered: {}", reason);
        self.circuit_breaker.trigger_emergency(reason).await
    }

    async fn get_current_metrics(&self) -> Result<RiskMetrics, RiskError> {
        self.engine.get_current_metrics().await
    }
}

/// Risk assessment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentRequest {
    pub id: Uuid,
    pub position: Position,
    pub trade_intent: TradeIntent,
    pub market_context: MarketContext,
    pub timestamp: DateTime<Utc>,
}

/// Risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub decision: RiskDecision,
    pub reason: String,
    pub confidence: f64,
    pub metrics: RiskMetrics,
    pub timestamp: DateTime<Utc>,
}

/// Risk decision types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskDecision {
    Approve,
    ApproveWithLimits {
        max_size: f64,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
    },
    Reject,
    Defer {
        until: DateTime<Utc>,
        reason: String,
    },
}

/// Trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub id: Uuid,
    pub symbol: String,
    pub size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub timestamp: DateTime<Utc>,
}

/// Trade intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeIntent {
    pub symbol: String,
    pub side: TradeSide,
    pub size: f64,
    pub price: Option<f64>,
    pub order_type: OrderType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Market context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub volatility: f64,
    pub liquidity: f64,
    pub spread: f64,
    pub volume: f64,
    pub trend: MarketTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketTrend {
    Bullish,
    Bearish,
    Sideways,
    Unknown,
}

/// Position update information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    pub position_id: Uuid,
    pub new_size: Option<f64>,
    pub new_price: Option<f64>,
    pub realized_pnl: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub var_1d: f64,
    pub var_7d: f64,
    pub expected_shortfall: f64,
    pub max_drawdown: f64,
    pub portfolio_beta: f64,
    pub concentration_risk: f64,
    pub liquidity_risk: f64,
    pub tail_risk: f64,
}

/// Protection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectionStrategy {
    pub stop_loss: Option<StopLoss>,
    pub take_profit: Option<TakeProfit>,
    pub position_sizing: PositionSizing,
    pub hedging: Option<HedgingStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopLoss {
    pub price: f64,
    pub percentage: f64,
    pub strategy_type: StopLossType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopLossType {
    Fixed,
    Trailing,
    Dynamic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfit {
    pub price: f64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizing {
    pub max_size: f64,
    pub recommended_size: f64,
    pub risk_per_trade: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgingStrategy {
    pub hedge_ratio: f64,
    pub hedge_instrument: String,
    pub dynamic_adjustment: bool,
}

/// Error types
#[derive(thiserror::Error, Debug)]
pub enum RiskError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Assessment error: {0}")]
    Assessment(String),
    
    #[error("Position error: {0}")]
    Position(String),
    
    #[error("Circuit breaker error: {0}")]
    CircuitBreaker(String),
    
    #[error("Protection error: {0}")]
    Protection(String),
    
    #[error("Metrics error: {0}")]
    Metrics(String),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for risk operations
pub type RiskResult<T> = Result<T, RiskError>;

/// Trait for risk assessment providers
#[async_trait]
pub trait RiskAssessor: Send + Sync {
    async fn assess(&self, request: RiskAssessmentRequest) -> RiskResult<RiskAssessment>;
    async fn update_position(&self, update: PositionUpdate) -> RiskResult<()>;
    async fn get_metrics(&self) -> RiskResult<RiskMetrics>;
}

/// Trait for protection strategy providers
#[async_trait]
pub trait ProtectionProvider: Send + Sync {
    async fn generate_strategy(&self, position: &Position) -> RiskResult<ProtectionStrategy>;
    async fn adjust_strategy(&self, position: &Position, market_change: f64) -> RiskResult<ProtectionStrategy>;
}

/// Trait for circuit breaker implementations
#[async_trait]
pub trait CircuitBreakerProvider: Send + Sync {
    async fn check_conditions(&self, assessment: &RiskAssessment) -> RiskResult<bool>;
    async fn is_triggered(&self) -> bool;
    async fn reset(&self) -> RiskResult<()>;
    async fn trigger_emergency(&self, reason: String) -> RiskResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_manager_creation() {
        let config = RiskConfig::default();
        let manager = RiskManager::new(config).await.unwrap();
        
        let metrics = manager.get_risk_metrics().await.unwrap();
        assert!(metrics.var_1d >= 0.0);
    }

    #[tokio::test]
    async fn test_risk_assessment() {
        let config = RiskConfig::default();
        let manager = RiskManager::new(config).await.unwrap();
        
        let request = RiskAssessmentRequest {
            id: Uuid::new_v4(),
            position: Position {
                id: Uuid::new_v4(),
                symbol: "SOL/USDC".to_string(),
                size: 100.0,
                entry_price: 50.0,
                current_price: 52.0,
                unrealized_pnl: 200.0,
                timestamp: Utc::now(),
            },
            trade_intent: TradeIntent {
                symbol: "SOL/USDC".to_string(),
                side: TradeSide::Buy,
                size: 50.0,
                price: Some(52.0),
                order_type: OrderType::Limit,
            },
            market_context: MarketContext {
                volatility: 0.25,
                liquidity: 1000000.0,
                spread: 0.01,
                volume: 50000.0,
                trend: MarketTrend::Bullish,
            },
            timestamp: Utc::now(),
        };
        
        let assessment = manager.assess_risk(request).await.unwrap();
        assert!(matches!(assessment.decision, RiskDecision::Approve | RiskDecision::ApproveWithLimits { .. }));
    }
}