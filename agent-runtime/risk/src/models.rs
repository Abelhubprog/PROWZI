//! Risk management data models and calculators

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{config::RiskConfig, RiskError, RiskResult, RiskAssessmentRequest, Position};
use crate::engine::{PortfolioState, MarketDataCache};

/// VaR (Value at Risk) calculator
pub struct VaRCalculator {
    confidence_level: f64,
    time_horizon: u32,
}

impl VaRCalculator {
    pub fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            confidence_level: 0.95, // 95% confidence level
            time_horizon: 1,        // 1 day horizon
        })
    }

    pub async fn calculate_var(
        &self,
        portfolio: &PortfolioState,
        request: &RiskAssessmentRequest,
    ) -> RiskResult<VaRMetrics> {
        // Simplified VaR calculation using historical simulation
        let portfolio_volatility = self.calculate_portfolio_volatility(portfolio).await?;
        let portfolio_value = portfolio.metrics.total_value;
        
        // VaR = Portfolio Value * Volatility * Z-score * sqrt(time_horizon)
        let z_score = self.get_z_score(self.confidence_level);
        let var_1d = portfolio_value * portfolio_volatility * z_score;
        let var_7d = var_1d * (7.0_f64).sqrt();
        
        // Expected Shortfall (Conditional VaR)
        let expected_shortfall = var_1d * 1.3; // Simplified calculation
        
        // Tail risk (beyond VaR)
        let tail_risk = self.calculate_tail_risk(portfolio_volatility, portfolio_value)?;

        Ok(VaRMetrics {
            var_1d,
            var_7d,
            expected_shortfall,
            tail_risk,
            confidence_level: self.confidence_level,
            confidence_interval: portfolio_volatility * 0.1, // Simplified
            timestamp: Utc::now(),
        })
    }

    pub async fn calculate_portfolio_var(&self, portfolio: &PortfolioState) -> RiskResult<VaRMetrics> {
        let portfolio_volatility = self.calculate_portfolio_volatility(portfolio).await?;
        let portfolio_value = portfolio.metrics.total_value;
        
        let z_score = self.get_z_score(self.confidence_level);
        let var_1d = portfolio_value * portfolio_volatility * z_score;
        let var_7d = var_1d * (7.0_f64).sqrt();
        let expected_shortfall = var_1d * 1.3;
        let tail_risk = self.calculate_tail_risk(portfolio_volatility, portfolio_value)?;

        Ok(VaRMetrics {
            var_1d,
            var_7d,
            expected_shortfall,
            tail_risk,
            confidence_level: self.confidence_level,
            confidence_interval: portfolio_volatility * 0.1,
            timestamp: Utc::now(),
        })
    }

    async fn calculate_portfolio_volatility(&self, portfolio: &PortfolioState) -> RiskResult<f64> {
        if portfolio.positions.is_empty() {
            return Ok(0.0);
        }

        // Simplified volatility calculation
        // In production, would use historical returns and correlation matrix
        let mut weighted_volatility = 0.0;
        let total_value = portfolio.metrics.total_value;

        for position in portfolio.positions.values() {
            let weight = (position.current_price * position.size.abs()) / total_value;
            // Assume 25% annual volatility for crypto assets
            let asset_volatility = 0.25 / (252.0_f64).sqrt(); // Daily volatility
            weighted_volatility += weight * weight * asset_volatility * asset_volatility;
        }

        Ok(weighted_volatility.sqrt())
    }

    fn get_z_score(&self, confidence_level: f64) -> f64 {
        // Z-scores for common confidence levels
        match confidence_level {
            x if (x - 0.90).abs() < 0.001 => 1.282,
            x if (x - 0.95).abs() < 0.001 => 1.645,
            x if (x - 0.99).abs() < 0.001 => 2.326,
            _ => 1.645, // Default to 95%
        }
    }

    fn calculate_tail_risk(&self, volatility: f64, value: f64) -> RiskResult<f64> {
        // Simplified tail risk calculation
        Ok(value * volatility * 3.0) // 3-sigma event
    }
}

/// VaR metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRMetrics {
    pub var_1d: f64,
    pub var_7d: f64,
    pub expected_shortfall: f64,
    pub tail_risk: f64,
    pub confidence_level: f64,
    pub confidence_interval: f64,
    pub timestamp: DateTime<Utc>,
}

/// Concentration risk calculator
pub struct ConcentrationCalculator {
    max_position_concentration: f64,
    max_sector_concentration: f64,
}

impl ConcentrationCalculator {
    pub fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            max_position_concentration: config.assessment.max_position_concentration,
            max_sector_concentration: config.assessment.max_sector_concentration,
        })
    }

    pub fn calculate(
        &self,
        portfolio: &PortfolioState,
        request: &RiskAssessmentRequest,
    ) -> RiskResult<f64> {
        // Calculate concentration risk for the new position
        let total_value = portfolio.metrics.total_value + 
                         (request.trade_intent.size * request.trade_intent.price.unwrap_or(0.0));
        
        if total_value == 0.0 {
            return Ok(0.0);
        }

        let new_position_value = request.trade_intent.size * request.trade_intent.price.unwrap_or(0.0);
        let position_concentration = new_position_value / total_value;

        // Return normalized concentration risk (0-1 scale)
        Ok(position_concentration / self.max_position_concentration)
    }

    pub fn calculate_portfolio(&self, portfolio: &PortfolioState) -> RiskResult<f64> {
        if portfolio.positions.is_empty() || portfolio.metrics.total_value == 0.0 {
            return Ok(0.0);
        }

        // Calculate maximum position concentration
        let max_position_value = portfolio.positions.values()
            .map(|pos| pos.current_price * pos.size.abs())
            .fold(0.0, f64::max);

        let concentration = max_position_value / portfolio.metrics.total_value;
        Ok(concentration / self.max_position_concentration)
    }
}

/// Liquidity risk calculator
pub struct LiquidityCalculator {
    min_liquidity_ratio: f64,
}

impl LiquidityCalculator {
    pub fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            min_liquidity_ratio: config.assessment.min_liquidity_ratio,
        })
    }

    pub fn calculate(
        &self,
        portfolio: &PortfolioState,
        request: &RiskAssessmentRequest,
        market_data: &MarketDataCache,
    ) -> RiskResult<f64> {
        // Get liquidity data for the requested symbol
        let volume = market_data.volumes.get(&request.trade_intent.symbol)
            .copied()
            .unwrap_or(1000000.0); // Default volume

        let trade_size_value = request.trade_intent.size * request.trade_intent.price.unwrap_or(0.0);
        
        // Liquidity risk increases as trade size approaches market volume
        let liquidity_impact = trade_size_value / volume;
        
        // Return normalized liquidity risk (0-1 scale)
        Ok(liquidity_impact.min(1.0))
    }

    pub fn calculate_portfolio(
        &self,
        portfolio: &PortfolioState,
        market_data: &MarketDataCache,
    ) -> RiskResult<f64> {
        if portfolio.positions.is_empty() {
            return Ok(0.0);
        }

        let mut total_liquidity_risk = 0.0;
        let mut total_weight = 0.0;

        for position in portfolio.positions.values() {
            let volume = market_data.volumes.get(&position.symbol)
                .copied()
                .unwrap_or(1000000.0);

            let position_value = position.current_price * position.size.abs();
            let weight = position_value / portfolio.metrics.total_value;
            let liquidity_impact = position_value / volume;

            total_liquidity_risk += weight * liquidity_impact;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            Ok((total_liquidity_risk / total_weight).min(1.0))
        } else {
            Ok(0.0)
        }
    }
}

/// Volatility risk calculator
pub struct VolatilityCalculator {
    max_volatility: f64,
}

impl VolatilityCalculator {
    pub fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            max_volatility: config.assessment.max_volatility,
        })
    }

    pub fn calculate(
        &self,
        request: &RiskAssessmentRequest,
        market_data: &MarketDataCache,
    ) -> RiskResult<f64> {
        // Get volatility for the requested symbol
        let volatility = market_data.volatilities.get(&request.trade_intent.symbol)
            .copied()
            .unwrap_or(0.25); // Default 25% volatility

        // Also consider market context volatility
        let market_volatility = request.market_context.volatility;
        let combined_volatility = (volatility + market_volatility) / 2.0;

        // Return normalized volatility risk (0-1 scale)
        Ok((combined_volatility / self.max_volatility).min(1.0))
    }
}

/// Correlation risk calculator
pub struct CorrelationCalculator {
    max_correlation: f64,
}

impl CorrelationCalculator {
    pub fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            max_correlation: config.assessment.max_correlation,
        })
    }

    pub fn calculate(
        &self,
        portfolio: &PortfolioState,
        request: &RiskAssessmentRequest,
    ) -> RiskResult<f64> {
        if portfolio.positions.is_empty() {
            return Ok(0.0);
        }

        // Simplified correlation calculation
        // In production, would use historical correlation matrix
        let mut max_correlation = 0.0;

        for position in portfolio.positions.values() {
            // Assume high correlation for same-sector assets
            let correlation = if position.symbol.contains("SOL") && 
                                request.trade_intent.symbol.contains("SOL") {
                0.8 // High correlation
            } else if position.symbol.contains(&request.trade_intent.symbol[..3]) {
                0.6 // Medium correlation
            } else {
                0.3 // Low correlation
            };

            max_correlation = max_correlation.max(correlation);
        }

        // Return normalized correlation risk (0-1 scale)
        Ok(max_correlation / self.max_correlation)
    }
}

/// Aggregate risk score
#[derive(Debug, Clone)]
pub struct AggregateRiskScore {
    pub total_score: f64,
    pub components: RiskComponents,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
}

/// Risk components breakdown
#[derive(Debug, Clone)]
pub struct RiskComponents {
    pub var_risk: f64,
    pub concentration_risk: f64,
    pub liquidity_risk: f64,
    pub volatility_risk: f64,
    pub correlation_risk: f64,
}

/// Risk factor for individual assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub symbol: String,
    pub beta: f64,
    pub volatility: f64,
    pub correlation_to_market: f64,
    pub liquidity_score: f64,
    pub last_updated: DateTime<Utc>,
}

/// Market regime detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Normal,
    HighVolatility,
    Crisis,
    Recovery,
}

/// Risk scenario for stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskScenario {
    pub name: String,
    pub description: String,
    pub shock_factors: HashMap<String, f64>,
    pub probability: f64,
    pub time_horizon_days: u32,
}

/// Stress test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    pub scenario: RiskScenario,
    pub portfolio_impact: f64,
    pub position_impacts: HashMap<Uuid, f64>,
    pub metrics_after_shock: VaRMetrics,
    pub timestamp: DateTime<Utc>,
}

/// Risk limit definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimit {
    pub id: Uuid,
    pub name: String,
    pub limit_type: RiskLimitType,
    pub threshold: f64,
    pub currency: String,
    pub scope: RiskLimitScope,
    pub action: RiskLimitAction,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLimitType {
    VaR,
    Concentration,
    Drawdown,
    Leverage,
    Volatility,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLimitScope {
    Portfolio,
    Position(Uuid),
    Symbol(String),
    Sector(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLimitAction {
    Alert,
    Block,
    Reduce,
    Liquidate,
}

/// Real-time risk monitoring event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvent {
    pub id: Uuid,
    pub event_type: RiskEventType,
    pub severity: RiskSeverity,
    pub description: String,
    pub affected_positions: Vec<Uuid>,
    pub metrics_snapshot: Option<VaRMetrics>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEventType {
    LimitBreach,
    ThresholdWarning,
    MarketShock,
    PositionAlert,
    SystemAlert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;

    #[test]
    fn test_var_calculator() {
        let config = RiskConfig::default();
        let calculator = VaRCalculator::new(&config).unwrap();
        assert_eq!(calculator.confidence_level, 0.95);
    }

    #[test]
    fn test_risk_components() {
        let components = RiskComponents {
            var_risk: 0.1,
            concentration_risk: 0.2,
            liquidity_risk: 0.15,
            volatility_risk: 0.25,
            correlation_risk: 0.18,
        };

        assert!(components.var_risk > 0.0);
        assert!(components.concentration_risk > 0.0);
    }
}