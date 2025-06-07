//! Advanced risk assessment engine with real-time calculations

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use parking_lot::Mutex;
use dashmap::DashMap;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

use crate::{
    config::RiskConfig,
    models::*,
    metrics::MetricsCollector,
    RiskError, RiskResult,
    RiskAssessmentRequest, RiskAssessment, RiskDecision, 
    PositionUpdate, RiskMetrics, Position,
};

/// High-performance risk assessment engine
pub struct RiskEngine {
    config: RiskConfig,
    metrics: Arc<MetricsCollector>,
    portfolio_state: Arc<RwLock<PortfolioState>>,
    market_data: Arc<RwLock<MarketDataCache>>,
    calculators: RiskCalculators,
    state_manager: Arc<StateManager>,
}

impl RiskEngine {
    /// Create new risk engine
    pub async fn new(config: RiskConfig, metrics: Arc<MetricsCollector>) -> RiskResult<Self> {
        let portfolio_state = Arc::new(RwLock::new(PortfolioState::new()));
        let market_data = Arc::new(RwLock::new(MarketDataCache::new()));
        let calculators = RiskCalculators::new(&config)?;
        let state_manager = Arc::new(StateManager::new(&config).await?);

        Ok(Self {
            config,
            metrics,
            portfolio_state,
            market_data,
            calculators,
            state_manager,
        })
    }

    /// Assess risk for a trading decision
    pub async fn assess_risk(&self, request: RiskAssessmentRequest) -> RiskResult<RiskAssessment> {
        let start_time = std::time::Instant::now();
        
        debug!("Starting risk assessment for request: {}", request.id);

        // Get current portfolio state
        let portfolio = self.portfolio_state.read().await;
        let market_data = self.market_data.read().await;

        // Calculate risk metrics
        let var_metrics = self.calculators.var.calculate_var(&portfolio, &request).await?;
        let concentration_risk = self.calculators.concentration.calculate(&portfolio, &request)?;
        let liquidity_risk = self.calculators.liquidity.calculate(&portfolio, &request, &market_data)?;
        let volatility_risk = self.calculators.volatility.calculate(&request, &market_data)?;
        let correlation_risk = self.calculators.correlation.calculate(&portfolio, &request)?;

        // Aggregate risk score
        let risk_score = self.calculate_aggregate_risk_score(
            &var_metrics,
            concentration_risk,
            liquidity_risk,
            volatility_risk,
            correlation_risk,
        )?;

        // Make risk decision
        let decision = self.make_risk_decision(&risk_score, &request).await?;

        let assessment = RiskAssessment {
            decision,
            reason: self.generate_assessment_reason(&risk_score)?,
            confidence: risk_score.confidence,
            metrics: RiskMetrics {
                var_1d: var_metrics.var_1d,
                var_7d: var_metrics.var_7d,
                expected_shortfall: var_metrics.expected_shortfall,
                max_drawdown: portfolio.metrics.max_drawdown,
                portfolio_beta: portfolio.metrics.beta,
                concentration_risk,
                liquidity_risk,
                tail_risk: var_metrics.tail_risk,
            },
            timestamp: Utc::now(),
        };

        // Record metrics
        let duration = start_time.elapsed();
        self.metrics.record_assessment_duration(duration).await?;
        self.metrics.record_risk_decision(&assessment.decision).await?;

        info!("Risk assessment completed in {:?}: {:?}", duration, assessment.decision);

        Ok(assessment)
    }

    /// Update position information
    pub async fn update_position(&self, update: PositionUpdate) -> RiskResult<()> {
        let mut portfolio = self.portfolio_state.write().await;
        
        if let Some(position) = portfolio.positions.get_mut(&update.position_id) {
            if let Some(new_size) = update.new_size {
                position.size = new_size;
            }
            if let Some(new_price) = update.new_price {
                position.current_price = new_price;
                position.unrealized_pnl = (new_price - position.entry_price) * position.size;
            }
            if let Some(realized_pnl) = update.realized_pnl {
                portfolio.metrics.total_realized_pnl += realized_pnl;
            }
            
            position.timestamp = update.timestamp;
        }

        // Recalculate portfolio metrics
        self.recalculate_portfolio_metrics(&mut portfolio).await?;

        debug!("Position updated: {}", update.position_id);
        Ok(())
    }

    /// Get current risk metrics
    pub async fn get_current_metrics(&self) -> RiskResult<RiskMetrics> {
        let portfolio = self.portfolio_state.read().await;
        let market_data = self.market_data.read().await;

        // Calculate current VaR
        let var_metrics = self.calculators.var.calculate_portfolio_var(&portfolio).await?;

        Ok(RiskMetrics {
            var_1d: var_metrics.var_1d,
            var_7d: var_metrics.var_7d,
            expected_shortfall: var_metrics.expected_shortfall,
            max_drawdown: portfolio.metrics.max_drawdown,
            portfolio_beta: portfolio.metrics.beta,
            concentration_risk: self.calculators.concentration.calculate_portfolio(&portfolio)?,
            liquidity_risk: self.calculators.liquidity.calculate_portfolio(&portfolio, &market_data)?,
            tail_risk: var_metrics.tail_risk,
        })
    }

    /// Calculate aggregate risk score
    fn calculate_aggregate_risk_score(
        &self,
        var_metrics: &VaRMetrics,
        concentration_risk: f64,
        liquidity_risk: f64,
        volatility_risk: f64,
        correlation_risk: f64,
    ) -> RiskResult<AggregateRiskScore> {
        let weights = &self.config.assessment.scoring_weights;

        let weighted_score = 
            var_metrics.var_1d * weights.var_weight +
            volatility_risk * weights.volatility_weight +
            liquidity_risk * weights.liquidity_weight +
            concentration_risk * weights.concentration_weight +
            correlation_risk * weights.correlation_weight;

        // Calculate confidence based on data quality
        let confidence = self.calculate_confidence(
            var_metrics,
            liquidity_risk,
            volatility_risk,
        )?;

        Ok(AggregateRiskScore {
            total_score: weighted_score,
            components: RiskComponents {
                var_risk: var_metrics.var_1d,
                concentration_risk,
                liquidity_risk,
                volatility_risk,
                correlation_risk,
            },
            confidence,
            timestamp: Utc::now(),
        })
    }

    /// Make risk decision based on risk score
    async fn make_risk_decision(
        &self,
        risk_score: &AggregateRiskScore,
        request: &RiskAssessmentRequest,
    ) -> RiskResult<RiskDecision> {
        // Check hard limits first
        if risk_score.components.var_risk > self.config.assessment.max_var_1d {
            return Ok(RiskDecision::Reject);
        }

        if risk_score.components.concentration_risk > self.config.assessment.max_position_concentration {
            return Ok(RiskDecision::Reject);
        }

        if risk_score.components.liquidity_risk > (1.0 - self.config.assessment.min_liquidity_ratio) {
            return Ok(RiskDecision::Reject);
        }

        // Check for position limits with adjustments
        if risk_score.total_score > 0.8 {
            // High risk - approve with strict limits
            let max_size = request.trade_intent.size * 0.5; // Reduce size by 50%
            let stop_loss = Some(request.trade_intent.price.unwrap_or(0.0) * 0.95); // 5% stop loss
            
            return Ok(RiskDecision::ApproveWithLimits {
                max_size,
                stop_loss,
                take_profit: Some(request.trade_intent.price.unwrap_or(0.0) * 1.10), // 10% take profit
            });
        }

        if risk_score.total_score > 0.6 {
            // Moderate risk - approve with moderate limits
            let max_size = request.trade_intent.size * 0.75; // Reduce size by 25%
            let stop_loss = Some(request.trade_intent.price.unwrap_or(0.0) * 0.97); // 3% stop loss
            
            return Ok(RiskDecision::ApproveWithLimits {
                max_size,
                stop_loss,
                take_profit: None,
            });
        }

        // Low risk - approve without limits
        Ok(RiskDecision::Approve)
    }

    /// Generate assessment reason
    fn generate_assessment_reason(&self, risk_score: &AggregateRiskScore) -> RiskResult<String> {
        let mut reasons = Vec::new();

        if risk_score.components.var_risk > self.config.assessment.max_var_1d * 0.8 {
            reasons.push("High VaR detected");
        }

        if risk_score.components.concentration_risk > self.config.assessment.max_position_concentration * 0.8 {
            reasons.push("High concentration risk");
        }

        if risk_score.components.liquidity_risk > 0.7 {
            reasons.push("Low liquidity conditions");
        }

        if risk_score.components.volatility_risk > self.config.assessment.max_volatility * 0.8 {
            reasons.push("High volatility detected");
        }

        if reasons.is_empty() {
            Ok("Risk assessment passed all checks".to_string())
        } else {
            Ok(reasons.join("; "))
        }
    }

    /// Calculate confidence score
    fn calculate_confidence(
        &self,
        var_metrics: &VaRMetrics,
        liquidity_risk: f64,
        volatility_risk: f64,
    ) -> RiskResult<f64> {
        // Base confidence on data quality and model reliability
        let mut confidence = 1.0;

        // Reduce confidence for high volatility (less predictable)
        confidence *= (1.0 - volatility_risk * 0.3);

        // Reduce confidence for low liquidity (less reliable pricing)
        confidence *= (1.0 - liquidity_risk * 0.2);

        // Reduce confidence if VaR confidence intervals are wide
        if var_metrics.confidence_interval > 0.1 {
            confidence *= 0.8;
        }

        Ok(confidence.max(0.1).min(1.0)) // Clamp between 0.1 and 1.0
    }

    /// Recalculate portfolio metrics
    async fn recalculate_portfolio_metrics(&self, portfolio: &mut PortfolioState) -> RiskResult<()> {
        // Calculate total portfolio value
        let total_value: f64 = portfolio.positions.values()
            .map(|pos| pos.current_price * pos.size.abs())
            .sum();

        portfolio.metrics.total_value = total_value;

        // Calculate total PnL
        let total_unrealized_pnl: f64 = portfolio.positions.values()
            .map(|pos| pos.unrealized_pnl)
            .sum();

        portfolio.metrics.total_unrealized_pnl = total_unrealized_pnl;

        // Update drawdown tracking
        let current_equity = total_value + total_unrealized_pnl + portfolio.metrics.total_realized_pnl;
        
        if current_equity > portfolio.metrics.peak_equity {
            portfolio.metrics.peak_equity = current_equity;
        }

        let current_drawdown = (portfolio.metrics.peak_equity - current_equity) / portfolio.metrics.peak_equity;
        portfolio.metrics.current_drawdown = current_drawdown;
        
        if current_drawdown > portfolio.metrics.max_drawdown {
            portfolio.metrics.max_drawdown = current_drawdown;
        }

        // Calculate portfolio beta (simplified)
        portfolio.metrics.beta = self.calculate_portfolio_beta(portfolio).await?;

        debug!("Portfolio metrics recalculated: value={}, drawdown={:.2}%", 
               total_value, current_drawdown * 100.0);

        Ok(())
    }

    /// Calculate portfolio beta
    async fn calculate_portfolio_beta(&self, portfolio: &PortfolioState) -> RiskResult<f64> {
        // Simplified beta calculation - in production would use more sophisticated methods
        let mut weighted_beta = 0.0;
        let total_value = portfolio.metrics.total_value;

        for position in portfolio.positions.values() {
            let weight = (position.current_price * position.size.abs()) / total_value;
            // Assume beta of 1.0 for crypto assets (relative to market)
            weighted_beta += weight * 1.0;
        }

        Ok(weighted_beta)
    }
}

/// Portfolio state management
#[derive(Debug)]
pub struct PortfolioState {
    pub positions: HashMap<Uuid, Position>,
    pub metrics: PortfolioMetrics,
    pub last_updated: DateTime<Utc>,
}

impl PortfolioState {
    fn new() -> Self {
        Self {
            positions: HashMap::new(),
            metrics: PortfolioMetrics::default(),
            last_updated: Utc::now(),
        }
    }
}

/// Portfolio metrics
#[derive(Debug, Default)]
pub struct PortfolioMetrics {
    pub total_value: f64,
    pub total_realized_pnl: f64,
    pub total_unrealized_pnl: f64,
    pub peak_equity: f64,
    pub current_drawdown: f64,
    pub max_drawdown: f64,
    pub beta: f64,
}

/// Market data cache
#[derive(Debug)]
pub struct MarketDataCache {
    pub prices: HashMap<String, f64>,
    pub volatilities: HashMap<String, f64>,
    pub volumes: HashMap<String, f64>,
    pub last_updated: DateTime<Utc>,
}

impl MarketDataCache {
    fn new() -> Self {
        Self {
            prices: HashMap::new(),
            volatilities: HashMap::new(),
            volumes: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

/// Risk calculators
pub struct RiskCalculators {
    pub var: VaRCalculator,
    pub concentration: ConcentrationCalculator,
    pub liquidity: LiquidityCalculator,
    pub volatility: VolatilityCalculator,
    pub correlation: CorrelationCalculator,
}

impl RiskCalculators {
    fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            var: VaRCalculator::new(config)?,
            concentration: ConcentrationCalculator::new(config)?,
            liquidity: LiquidityCalculator::new(config)?,
            volatility: VolatilityCalculator::new(config)?,
            correlation: CorrelationCalculator::new(config)?,
        })
    }
}

/// State manager for persistence
pub struct StateManager {
    config: RiskConfig,
}

impl StateManager {
    async fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;

    #[tokio::test]
    async fn test_risk_engine_creation() {
        let config = RiskConfig::default();
        let metrics = Arc::new(MetricsCollector::new(&config.metrics).unwrap());
        
        let engine = RiskEngine::new(config, metrics).await.unwrap();
        let current_metrics = engine.get_current_metrics().await.unwrap();
        
        assert!(current_metrics.var_1d >= 0.0);
    }

    #[tokio::test]
    async fn test_position_update() {
        let config = RiskConfig::default();
        let metrics = Arc::new(MetricsCollector::new(&config.metrics).unwrap());
        let engine = RiskEngine::new(config, metrics).await.unwrap();

        let position_id = Uuid::new_v4();
        let update = PositionUpdate {
            position_id,
            new_size: Some(100.0),
            new_price: Some(50.0),
            realized_pnl: Some(10.0),
            timestamp: Utc::now(),
        };

        // This will not find the position but shouldn't error
        engine.update_position(update).await.unwrap();
    }
}