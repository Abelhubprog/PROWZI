//! Risk Management Module for Prowzi Trading Agents
//! 
//! Provides comprehensive risk controls, position sizing, and safety mechanisms
//! for autonomous trading operations with focus on $10 minimum trades and Solana optimization.

use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Risk parameters for trading operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameters {
    /// Maximum position size as percentage of portfolio
    pub max_position_size_percent: f64,
    /// Maximum daily loss threshold
    pub max_daily_loss_usd: f64,
    /// Maximum number of concurrent positions
    pub max_concurrent_positions: u32,
    /// Minimum trade size in USD (default: $10 for Solana optimization)
    pub min_trade_size_usd: f64,
    /// Maximum trade size in USD
    pub max_trade_size_usd: f64,
    /// Stop loss percentage
    pub stop_loss_percent: f64,
    /// Take profit percentage
    pub take_profit_percent: f64,
    /// Maximum slippage tolerance
    pub max_slippage_percent: f64,
    /// Risk-reward ratio minimum
    pub min_risk_reward_ratio: f64,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            max_position_size_percent: 5.0,
            max_daily_loss_usd: 100.0,
            max_concurrent_positions: 10,
            min_trade_size_usd: 10.0, // Solana-optimized minimum
            max_trade_size_usd: 1000.0,
            stop_loss_percent: 2.0,
            take_profit_percent: 4.0,
            max_slippage_percent: 0.5,
            min_risk_reward_ratio: 2.0,
        }
    }
}

/// Risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub assessment_id: String,
    pub agent_id: String,
    pub risk_score: f64, // 0.0 (low risk) to 1.0 (high risk)
    pub risk_level: RiskLevel,
    pub recommendations: Vec<String>,
    pub blockers: Vec<String>,
    pub position_size_recommendation: f64,
    pub assessed_at: DateTime<Utc>,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Trading position for risk tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub position_id: String,
    pub agent_id: String,
    pub symbol: String,
    pub size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub opened_at: DateTime<Utc>,
    pub risk_score: f64,
}

/// Portfolio metrics for risk calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    pub total_value_usd: f64,
    pub available_cash_usd: f64,
    pub total_positions: u32,
    pub daily_pnl_usd: f64,
    pub max_drawdown_percent: f64,
    pub sharpe_ratio: f64,
    pub var_95_usd: f64, // Value at Risk (95% confidence)
    pub leverage_ratio: f64,
}

/// Risk event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEvent {
    PositionSizeExceeded {
        agent_id: String,
        attempted_size: f64,
        max_allowed: f64,
    },
    DailyLossThresholdReached {
        agent_id: String,
        current_loss: f64,
        threshold: f64,
    },
    StopLossTriggered {
        position_id: String,
        agent_id: String,
        trigger_price: f64,
    },
    SlippageExceeded {
        agent_id: String,
        expected_price: f64,
        actual_price: f64,
        slippage_percent: f64,
    },
    MarginCallWarning {
        agent_id: String,
        current_margin: f64,
        required_margin: f64,
    },
}

/// Risk manager for trading operations
pub struct RiskManager {
    parameters: RiskParameters,
    positions: HashMap<String, Position>,
    portfolio_metrics: PortfolioMetrics,
    risk_events: Vec<(DateTime<Utc>, RiskEvent)>,
}

impl RiskManager {
    pub fn new(parameters: RiskParameters) -> Self {
        Self {
            parameters,
            positions: HashMap::new(),
            portfolio_metrics: PortfolioMetrics {
                total_value_usd: 0.0,
                available_cash_usd: 0.0,
                total_positions: 0,
                daily_pnl_usd: 0.0,
                max_drawdown_percent: 0.0,
                sharpe_ratio: 0.0,
                var_95_usd: 0.0,
                leverage_ratio: 0.0,
            },
            risk_events: Vec::new(),
        }
    }

    /// Assess risk for a proposed trade
    pub async fn assess_trade_risk(
        &self,
        agent_id: &str,
        symbol: &str,
        proposed_size: f64,
        entry_price: f64,
    ) -> Result<RiskAssessment, RiskError> {
        let mut risk_score = 0.0;
        let mut recommendations = Vec::new();
        let mut blockers = Vec::new();

        // Check minimum trade size
        let trade_value = proposed_size * entry_price;
        if trade_value < self.parameters.min_trade_size_usd {
            blockers.push(format!(
                "Trade value ${:.2} below minimum ${:.2}",
                trade_value, self.parameters.min_trade_size_usd
            ));
            risk_score += 0.8; // High risk for below minimum trades
        }

        // Check maximum trade size
        if trade_value > self.parameters.max_trade_size_usd {
            blockers.push(format!(
                "Trade value ${:.2} exceeds maximum ${:.2}",
                trade_value, self.parameters.max_trade_size_usd
            ));
        }

        // Check position size limits
        let position_percent = (trade_value / self.portfolio_metrics.total_value_usd) * 100.0;
        if position_percent > self.parameters.max_position_size_percent {
            blockers.push(format!(
                "Position size {:.2}% exceeds maximum {:.2}%",
                position_percent, self.parameters.max_position_size_percent
            ));
            risk_score += 0.6;
        }

        // Check concurrent position limits
        let agent_positions = self.positions.values()
            .filter(|p| p.agent_id == agent_id)
            .count() as u32;
        
        if agent_positions >= self.parameters.max_concurrent_positions {
            blockers.push(format!(
                "Agent has {} positions, maximum is {}",
                agent_positions, self.parameters.max_concurrent_positions
            ));
        }

        // Check daily loss limits
        if self.portfolio_metrics.daily_pnl_usd <= -self.parameters.max_daily_loss_usd {
            blockers.push("Daily loss threshold reached".to_string());
            risk_score += 0.9;
        }

        // Generate recommendations
        if risk_score > 0.3 {
            recommendations.push("Consider reducing position size".to_string());
        }
        if self.portfolio_metrics.var_95_usd > trade_value * 0.1 {
            recommendations.push("High portfolio VaR detected".to_string());
        }

        // Determine risk level
        let risk_level = match risk_score {
            s if s < 0.3 => RiskLevel::Low,
            s if s < 0.6 => RiskLevel::Medium,
            s if s < 0.8 => RiskLevel::High,
            _ => RiskLevel::Critical,
        };

        // Calculate recommended position size
        let max_risk_per_trade = self.portfolio_metrics.total_value_usd * 0.02; // 2% risk per trade
        let stop_loss_distance = entry_price * (self.parameters.stop_loss_percent / 100.0);
        let position_size_recommendation = if stop_loss_distance > 0.0 {
            (max_risk_per_trade / stop_loss_distance).min(proposed_size)
        } else {
            proposed_size * 0.5 // Conservative fallback
        };

        Ok(RiskAssessment {
            assessment_id: Uuid::new_v4().to_string(),
            agent_id: agent_id.to_string(),
            risk_score,
            risk_level,
            recommendations,
            blockers,
            position_size_recommendation,
            assessed_at: Utc::now(),
        })
    }

    /// Add a new position for tracking
    pub fn add_position(&mut self, position: Position) {
        self.positions.insert(position.position_id.clone(), position);
        self.update_portfolio_metrics();
    }

    /// Update position with current market data
    pub fn update_position(
        &mut self,
        position_id: &str,
        current_price: f64,
    ) -> Result<(), RiskError> {
        let position = self.positions.get_mut(position_id)
            .ok_or(RiskError::PositionNotFound)?;

        position.current_price = current_price;
        position.unrealized_pnl = (current_price - position.entry_price) * position.size;

        // Check stop loss
        if let Some(stop_loss) = position.stop_loss {
            if (position.size > 0.0 && current_price <= stop_loss) ||
               (position.size < 0.0 && current_price >= stop_loss) {
                self.record_risk_event(RiskEvent::StopLossTriggered {
                    position_id: position_id.to_string(),
                    agent_id: position.agent_id.clone(),
                    trigger_price: current_price,
                });
            }
        }

        self.update_portfolio_metrics();
        Ok(())
    }

    /// Close a position
    pub fn close_position(&mut self, position_id: &str) -> Result<Position, RiskError> {
        self.positions.remove(position_id)
            .ok_or(RiskError::PositionNotFound)
            .map(|pos| {
                self.update_portfolio_metrics();
                pos
            })
    }

    /// Check if trade is allowed based on current risk state
    pub fn is_trade_allowed(&self, assessment: &RiskAssessment) -> bool {
        assessment.blockers.is_empty() && assessment.risk_level != RiskLevel::Critical
    }

    /// Record a risk event
    pub fn record_risk_event(&mut self, event: RiskEvent) {
        self.risk_events.push((Utc::now(), event));
        
        // Keep only recent events (last 24 hours)
        let cutoff = Utc::now() - Duration::hours(24);
        self.risk_events.retain(|(timestamp, _)| *timestamp > cutoff);
    }

    /// Get recent risk events
    pub fn get_recent_risk_events(&self) -> &[(DateTime<Utc>, RiskEvent)] {
        &self.risk_events
    }

    /// Update portfolio metrics
    fn update_portfolio_metrics(&mut self) {
        let total_unrealized_pnl: f64 = self.positions.values()
            .map(|p| p.unrealized_pnl)
            .sum();

        self.portfolio_metrics.total_positions = self.positions.len() as u32;
        
        // TODO: Implement more sophisticated portfolio metrics calculation
        // This is a simplified version for the initial implementation
    }

    /// Get current risk parameters
    pub fn get_parameters(&self) -> &RiskParameters {
        &self.parameters
    }

    /// Update risk parameters
    pub fn update_parameters(&mut self, parameters: RiskParameters) {
        self.parameters = parameters;
    }

    /// Get portfolio metrics
    pub fn get_portfolio_metrics(&self) -> &PortfolioMetrics {
        &self.portfolio_metrics
    }

    /// Calculate Value at Risk (VaR)
    pub fn calculate_var(&self, confidence_level: f64, time_horizon_days: u32) -> f64 {
        // Simplified VaR calculation
        // In production, this would use historical price data and Monte Carlo simulation
        let portfolio_volatility = 0.02; // 2% daily volatility assumption
        let z_score = match confidence_level {
            0.95 => 1.645,
            0.99 => 2.326,
            _ => 1.645,
        };
        
        self.portfolio_metrics.total_value_usd * portfolio_volatility * z_score * (time_horizon_days as f64).sqrt()
    }
}

/// Risk management errors
#[derive(Debug, thiserror::Error)]
pub enum RiskError {
    #[error("Position not found")]
    PositionNotFound,

    #[error("Risk threshold exceeded: {0}")]
    ThresholdExceeded(String),

    #[error("Invalid risk parameters: {0}")]
    InvalidParameters(String),

    #[error("Insufficient portfolio data")]
    InsufficientData,

    #[error("Market data unavailable")]
    MarketDataUnavailable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_assessment_minimum_trade() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());
        risk_manager.portfolio_metrics.total_value_usd = 10000.0;

        // Test trade below minimum ($10)
        let assessment = risk_manager.assess_trade_risk(
            "test_agent",
            "SOL/USD",
            0.1, // Small size
            50.0, // $5 trade value
        ).await.unwrap();

        assert!(!assessment.blockers.is_empty());
        assert!(assessment.risk_score > 0.5);
    }

    #[tokio::test]
    async fn test_risk_assessment_valid_trade() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());
        risk_manager.portfolio_metrics.total_value_usd = 10000.0;

        // Test valid trade above minimum
        let assessment = risk_manager.assess_trade_risk(
            "test_agent",
            "SOL/USD",
            1.0, // Size
            50.0, // $50 trade value (above $10 minimum)
        ).await.unwrap();

        assert!(assessment.blockers.is_empty());
        assert_eq!(assessment.risk_level, RiskLevel::Low);
    }

    #[test]
    fn test_position_management() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());

        let position = Position {
            position_id: "pos_1".to_string(),
            agent_id: "agent_1".to_string(),
            symbol: "SOL/USD".to_string(),
            size: 1.0,
            entry_price: 100.0,
            current_price: 100.0,
            unrealized_pnl: 0.0,
            stop_loss: Some(95.0),
            take_profit: Some(110.0),
            opened_at: Utc::now(),
            risk_score: 0.3,
        };

        risk_manager.add_position(position);
        assert_eq!(risk_manager.positions.len(), 1);

        // Update position price
        risk_manager.update_position("pos_1", 105.0).unwrap();
        let position = &risk_manager.positions["pos_1"];
        assert_eq!(position.unrealized_pnl, 5.0);

        // Close position
        let closed_position = risk_manager.close_position("pos_1").unwrap();
        assert_eq!(closed_position.position_id, "pos_1");
        assert!(risk_manager.positions.is_empty());
    }

    #[test]
    fn test_var_calculation() {
        let mut risk_manager = RiskManager::new(RiskParameters::default());
        risk_manager.portfolio_metrics.total_value_usd = 100000.0;

        let var_95 = risk_manager.calculate_var(0.95, 1);
        assert!(var_95 > 0.0);
        assert!(var_95 < risk_manager.portfolio_metrics.total_value_usd);
    }
}
