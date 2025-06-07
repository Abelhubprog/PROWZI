//! Configuration management for the risk system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration for the risk management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Risk assessment configuration
    pub assessment: AssessmentConfig,
    
    /// Position management configuration
    pub position: PositionConfig,
    
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    
    /// Protection engine configuration
    pub protection: ProtectionConfig,
    
    /// Metrics and monitoring configuration
    pub metrics: MetricsConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Neural network configuration (optional)
    pub neural: Option<NeuralConfig>,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            assessment: AssessmentConfig::default(),
            position: PositionConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            protection: ProtectionConfig::default(),
            metrics: MetricsConfig::default(),
            database: DatabaseConfig::default(),
            neural: None,
        }
    }
}

/// Risk assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentConfig {
    /// Maximum value at risk (percentage of portfolio)
    pub max_var_1d: f64,
    pub max_var_7d: f64,
    
    /// Maximum expected shortfall
    pub max_expected_shortfall: f64,
    
    /// Maximum drawdown threshold
    pub max_drawdown: f64,
    
    /// Concentration risk limits
    pub max_position_concentration: f64,
    pub max_sector_concentration: f64,
    
    /// Liquidity requirements
    pub min_liquidity_ratio: f64,
    
    /// Volatility thresholds
    pub max_volatility: f64,
    
    /// Correlation limits
    pub max_correlation: f64,
    
    /// Risk scoring weights
    pub scoring_weights: ScoringWeights,
}

impl Default for AssessmentConfig {
    fn default() -> Self {
        Self {
            max_var_1d: 0.02,      // 2% daily VaR
            max_var_7d: 0.05,      // 5% weekly VaR
            max_expected_shortfall: 0.03,
            max_drawdown: 0.10,    // 10% max drawdown
            max_position_concentration: 0.20, // 20% per position
            max_sector_concentration: 0.40,   // 40% per sector
            min_liquidity_ratio: 0.10,        // 10% minimum liquidity
            max_volatility: 0.50,             // 50% max volatility
            max_correlation: 0.80,            // 80% max correlation
            scoring_weights: ScoringWeights::default(),
        }
    }
}

/// Risk scoring weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub var_weight: f64,
    pub volatility_weight: f64,
    pub liquidity_weight: f64,
    pub concentration_weight: f64,
    pub correlation_weight: f64,
    pub momentum_weight: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            var_weight: 0.25,
            volatility_weight: 0.20,
            liquidity_weight: 0.15,
            concentration_weight: 0.20,
            correlation_weight: 0.10,
            momentum_weight: 0.10,
        }
    }
}

/// Position management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionConfig {
    /// Maximum position size (percentage of portfolio)
    pub max_position_size: f64,
    
    /// Maximum number of positions
    pub max_positions: usize,
    
    /// Minimum position size (USD)
    pub min_position_size: f64,
    
    /// Position sizing algorithm
    pub sizing_algorithm: PositionSizingAlgorithm,
    
    /// Rebalancing configuration
    pub rebalancing: RebalancingConfig,
}

impl Default for PositionConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.20,  // 20% max per position
            max_positions: 50,
            min_position_size: 100.0, // $100 minimum
            sizing_algorithm: PositionSizingAlgorithm::KellyOptimal,
            rebalancing: RebalancingConfig::default(),
        }
    }
}

/// Position sizing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizingAlgorithm {
    Fixed,
    PercentageRisk,
    KellyOptimal,
    VolatilityAdjusted,
    SharpOptimal,
}

/// Rebalancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingConfig {
    pub enabled: bool,
    pub frequency_hours: u64,
    pub threshold_drift: f64,
    pub min_trade_size: f64,
}

impl Default for RebalancingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency_hours: 24,    // Daily rebalancing
            threshold_drift: 0.05,  // 5% drift threshold
            min_trade_size: 50.0,   // $50 minimum trade
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    
    /// Loss thresholds that trigger circuit breaker
    pub loss_thresholds: LossThresholds,
    
    /// Volatility thresholds
    pub volatility_thresholds: VolatilityThresholds,
    
    /// Time-based thresholds
    pub time_thresholds: TimeThresholds,
    
    /// Recovery conditions
    pub recovery: RecoveryConfig,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            loss_thresholds: LossThresholds::default(),
            volatility_thresholds: VolatilityThresholds::default(),
            time_thresholds: TimeThresholds::default(),
            recovery: RecoveryConfig::default(),
        }
    }
}

/// Loss thresholds for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossThresholds {
    /// Daily loss threshold (percentage)
    pub daily_loss: f64,
    
    /// Weekly loss threshold (percentage)
    pub weekly_loss: f64,
    
    /// Monthly loss threshold (percentage)
    pub monthly_loss: f64,
    
    /// Consecutive losses threshold
    pub consecutive_losses: usize,
}

impl Default for LossThresholds {
    fn default() -> Self {
        Self {
            daily_loss: 0.05,      // 5% daily loss
            weekly_loss: 0.10,     // 10% weekly loss
            monthly_loss: 0.20,    // 20% monthly loss
            consecutive_losses: 5,  // 5 consecutive losses
        }
    }
}

/// Volatility thresholds for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityThresholds {
    /// Market volatility threshold
    pub market_volatility: f64,
    
    /// Portfolio volatility threshold
    pub portfolio_volatility: f64,
    
    /// Volatility spike threshold
    pub volatility_spike: f64,
}

impl Default for VolatilityThresholds {
    fn default() -> Self {
        Self {
            market_volatility: 0.50,    // 50% market volatility
            portfolio_volatility: 0.40, // 40% portfolio volatility
            volatility_spike: 2.0,       // 2x volatility spike
        }
    }
}

/// Time-based thresholds for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeThresholds {
    /// Maximum trading hours per day
    pub max_trading_hours: u64,
    
    /// Cooldown period after circuit breaker (minutes)
    pub cooldown_minutes: u64,
    
    /// Maximum number of trades per hour
    pub max_trades_per_hour: usize,
}

impl Default for TimeThresholds {
    fn default() -> Self {
        Self {
            max_trading_hours: 18,      // 18 hours max trading
            cooldown_minutes: 30,       // 30 minute cooldown
            max_trades_per_hour: 100,   // 100 trades per hour
        }
    }
}

/// Recovery configuration for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Automatic recovery enabled
    pub auto_recovery: bool,
    
    /// Minimum recovery time (minutes)
    pub min_recovery_time: u64,
    
    /// Recovery conditions
    pub conditions: Vec<RecoveryCondition>,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            min_recovery_time: 60,  // 1 hour minimum
            conditions: vec![
                RecoveryCondition::VolatilityNormalized,
                RecoveryCondition::MarketStable,
                RecoveryCondition::TimeElapsed,
            ],
        }
    }
}

/// Recovery conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryCondition {
    VolatilityNormalized,
    MarketStable,
    TimeElapsed,
    ManualApproval,
}

/// Protection engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectionConfig {
    /// Stop loss configuration
    pub stop_loss: StopLossConfig,
    
    /// Take profit configuration
    pub take_profit: TakeProfitConfig,
    
    /// Dynamic hedging configuration
    pub hedging: HedgingConfig,
    
    /// Risk limits
    pub limits: RiskLimits,
}

impl Default for ProtectionConfig {
    fn default() -> Self {
        Self {
            stop_loss: StopLossConfig::default(),
            take_profit: TakeProfitConfig::default(),
            hedging: HedgingConfig::default(),
            limits: RiskLimits::default(),
        }
    }
}

/// Stop loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopLossConfig {
    pub enabled: bool,
    pub default_percentage: f64,
    pub trailing_enabled: bool,
    pub trailing_percentage: f64,
    pub dynamic_adjustment: bool,
}

impl Default for StopLossConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_percentage: 0.05,  // 5% stop loss
            trailing_enabled: true,
            trailing_percentage: 0.03, // 3% trailing
            dynamic_adjustment: true,
        }
    }
}

/// Take profit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfitConfig {
    pub enabled: bool,
    pub default_percentage: f64,
    pub partial_taking: bool,
    pub scaling_levels: Vec<f64>,
}

impl Default for TakeProfitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_percentage: 0.15,  // 15% take profit
            partial_taking: true,
            scaling_levels: vec![0.05, 0.10, 0.15, 0.20], // Multiple levels
        }
    }
}

/// Hedging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgingConfig {
    pub enabled: bool,
    pub hedge_ratio: f64,
    pub instruments: Vec<String>,
    pub dynamic_adjustment: bool,
    pub correlation_threshold: f64,
}

impl Default for HedgingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hedge_ratio: 0.50,     // 50% hedge ratio
            instruments: vec!["SOL-PERP".to_string()],
            dynamic_adjustment: true,
            correlation_threshold: 0.70, // 70% correlation threshold
        }
    }
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_portfolio_risk: f64,
    pub max_sector_risk: f64,
    pub max_correlation_risk: f64,
    pub max_leverage: f64,
    pub limits_by_asset: HashMap<String, AssetLimits>,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_portfolio_risk: 0.20,     // 20% max portfolio risk
            max_sector_risk: 0.40,        // 40% max sector risk
            max_correlation_risk: 0.80,   // 80% max correlation risk
            max_leverage: 2.0,            // 2x max leverage
            limits_by_asset: HashMap::new(),
        }
    }
}

/// Asset-specific limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetLimits {
    pub max_position_size: f64,
    pub max_daily_volume: f64,
    pub min_liquidity: f64,
    pub max_volatility: f64,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub collection_interval_seconds: u64,
    pub retention_days: u64,
    pub prometheus_endpoint: Option<String>,
    pub detailed_logging: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_seconds: 30, // 30 second intervals
            retention_days: 90,              // 90 days retention
            prometheus_endpoint: None,
            detailed_logging: false,
        }
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout_seconds: u64,
    pub query_timeout_seconds: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgresql://localhost/prowzi_risk".to_string(),
            max_connections: 20,
            connection_timeout_seconds: 30,
            query_timeout_seconds: 60,
        }
    }
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    pub enabled: bool,
    pub model_path: Option<String>,
    pub training_data_path: Option<String>,
    pub update_frequency_hours: u64,
    pub confidence_threshold: f64,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            training_data_path: None,
            update_frequency_hours: 24,  // Daily updates
            confidence_threshold: 0.80,  // 80% confidence threshold
        }
    }
}

impl RiskConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("PROWZI_RISK"))
            .build()?;
            
        settings.try_deserialize()
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate assessment config
        if self.assessment.max_var_1d <= 0.0 || self.assessment.max_var_1d > 1.0 {
            return Err("Invalid max_var_1d: must be between 0.0 and 1.0".to_string());
        }
        
        if self.assessment.max_drawdown <= 0.0 || self.assessment.max_drawdown > 1.0 {
            return Err("Invalid max_drawdown: must be between 0.0 and 1.0".to_string());
        }
        
        // Validate position config
        if self.position.max_positions == 0 {
            return Err("Invalid max_positions: must be greater than 0".to_string());
        }
        
        if self.position.min_position_size <= 0.0 {
            return Err("Invalid min_position_size: must be greater than 0".to_string());
        }
        
        // Validate circuit breaker config
        if self.circuit_breaker.enabled {
            if self.circuit_breaker.loss_thresholds.daily_loss <= 0.0 {
                return Err("Invalid daily_loss: must be greater than 0".to_string());
            }
        }
        
        // Validate database config
        if self.database.max_connections == 0 {
            return Err("Invalid max_connections: must be greater than 0".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RiskConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = RiskConfig::default();
        
        // Test invalid VaR
        config.assessment.max_var_1d = 1.5;
        assert!(config.validate().is_err());
        
        // Test invalid positions
        config.assessment.max_var_1d = 0.02;
        config.position.max_positions = 0;
        assert!(config.validate().is_err());
    }
}