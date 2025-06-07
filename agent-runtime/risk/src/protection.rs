//! Advanced protection engine for position management

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

use crate::{
    config::RiskConfig,
    models::*,
    metrics::MetricsCollector,
    RiskError, RiskResult, Position, ProtectionStrategy,
    StopLoss, TakeProfit, PositionSizing, HedgingStrategy,
    StopLossType,
};

/// Advanced protection engine with dynamic strategy optimization
pub struct ProtectionEngine {
    config: RiskConfig,
    metrics: Arc<MetricsCollector>,
    strategy_cache: Arc<RwLock<HashMap<Uuid, ProtectionStrategy>>>,
    optimizer: ProtectionOptimizer,
    monitor: ProtectionMonitor,
}

impl ProtectionEngine {
    /// Create new protection engine
    pub async fn new(config: RiskConfig, metrics: Arc<MetricsCollector>) -> RiskResult<Self> {
        let strategy_cache = Arc::new(RwLock::new(HashMap::new()));
        let optimizer = ProtectionOptimizer::new(&config)?;
        let monitor = ProtectionMonitor::new(&config).await?;

        Ok(Self {
            config,
            metrics,
            strategy_cache,
            optimizer,
            monitor,
        })
    }

    /// Generate optimal protection strategy for a position
    pub async fn generate_strategy(&self, position: &Position) -> RiskResult<ProtectionStrategy> {
        debug!("Generating protection strategy for position: {}", position.id);

        // Calculate position risk metrics
        let risk_metrics = self.calculate_position_risk(position).await?;
        
        // Generate stop loss strategy
        let stop_loss = self.generate_stop_loss(position, &risk_metrics).await?;
        
        // Generate take profit strategy
        let take_profit = self.generate_take_profit(position, &risk_metrics).await?;
        
        // Calculate optimal position sizing
        let position_sizing = self.calculate_position_sizing(position, &risk_metrics).await?;
        
        // Generate hedging strategy if needed
        let hedging = self.generate_hedging_strategy(position, &risk_metrics).await?;

        let strategy = ProtectionStrategy {
            stop_loss,
            take_profit,
            position_sizing,
            hedging,
        };

        // Cache the strategy
        let mut cache = self.strategy_cache.write().await;
        cache.insert(position.id, strategy.clone());

        // Start monitoring the position
        self.monitor.start_monitoring(position, &strategy).await?;

        info!("Generated protection strategy for position {}", position.id);
        Ok(strategy)
    }

    /// Update protection strategy based on market changes
    pub async fn update_strategy(
        &self, 
        position: &Position, 
        market_change: f64
    ) -> RiskResult<ProtectionStrategy> {
        let mut cache = self.strategy_cache.write().await;
        
        if let Some(current_strategy) = cache.get(&position.id) {
            // Optimize existing strategy
            let updated_strategy = self.optimizer
                .optimize_strategy(position, current_strategy, market_change)
                .await?;
            
            cache.insert(position.id, updated_strategy.clone());
            
            info!("Updated protection strategy for position {} due to {:.2}% market change", 
                  position.id, market_change * 100.0);
            
            Ok(updated_strategy)
        } else {
            // Generate new strategy if none exists
            drop(cache);
            self.generate_strategy(position).await
        }
    }

    /// Calculate position-specific risk metrics
    async fn calculate_position_risk(&self, position: &Position) -> RiskResult<PositionRiskMetrics> {
        let volatility = self.estimate_asset_volatility(&position.symbol).await?;
        let liquidity_score = self.estimate_liquidity_score(&position.symbol).await?;
        let beta = self.estimate_asset_beta(&position.symbol).await?;
        
        // Calculate position-specific VaR
        let position_value = position.current_price * position.size.abs();
        let var_1d = position_value * volatility * 1.645; // 95% confidence
        
        // Calculate correlation risk
        let correlation_risk = self.calculate_correlation_risk(position).await?;
        
        Ok(PositionRiskMetrics {
            volatility,
            liquidity_score,
            beta,
            var_1d,
            correlation_risk,
            unrealized_pnl_percent: position.unrealized_pnl / (position.entry_price * position.size.abs()),
        })
    }

    /// Generate optimal stop loss strategy
    async fn generate_stop_loss(
        &self,
        position: &Position,
        risk_metrics: &PositionRiskMetrics,
    ) -> RiskResult<Option<StopLoss>> {
        if !self.config.protection.stop_loss.enabled {
            return Ok(None);
        }

        // Dynamic stop loss based on volatility and risk
        let base_percentage = self.config.protection.stop_loss.default_percentage;
        let volatility_adjustment = risk_metrics.volatility * 0.5; // Scale volatility impact
        let adjusted_percentage = (base_percentage + volatility_adjustment).min(0.15); // Cap at 15%

        let stop_loss_price = if position.size > 0.0 {
            // Long position - stop loss below entry
            position.current_price * (1.0 - adjusted_percentage)
        } else {
            // Short position - stop loss above entry
            position.current_price * (1.0 + adjusted_percentage)
        };

        // Choose stop loss type based on market conditions
        let strategy_type = if risk_metrics.volatility > 0.3 {
            StopLossType::Dynamic // High volatility - use dynamic
        } else if self.config.protection.stop_loss.trailing_enabled {
            StopLossType::Trailing // Normal conditions - use trailing
        } else {
            StopLossType::Fixed // Conservative - use fixed
        };

        Ok(Some(StopLoss {
            price: stop_loss_price,
            percentage: adjusted_percentage,
            strategy_type,
        }))
    }

    /// Generate take profit strategy
    async fn generate_take_profit(
        &self,
        position: &Position,
        risk_metrics: &PositionRiskMetrics,
    ) -> RiskResult<Option<TakeProfit>> {
        if !self.config.protection.take_profit.enabled {
            return Ok(None);
        }

        // Dynamic take profit based on risk-reward ratio
        let base_percentage = self.config.protection.take_profit.default_percentage;
        
        // Adjust based on current P&L and risk
        let pnl_adjustment = if risk_metrics.unrealized_pnl_percent > 0.05 {
            0.05 // Increase take profit if already profitable
        } else {
            0.0
        };

        let adjusted_percentage = base_percentage + pnl_adjustment;

        let take_profit_price = if position.size > 0.0 {
            // Long position - take profit above current
            position.current_price * (1.0 + adjusted_percentage)
        } else {
            // Short position - take profit below current
            position.current_price * (1.0 - adjusted_percentage)
        };

        Ok(Some(TakeProfit {
            price: take_profit_price,
            percentage: adjusted_percentage,
        }))
    }

    /// Calculate optimal position sizing
    async fn calculate_position_sizing(
        &self,
        position: &Position,
        risk_metrics: &PositionRiskMetrics,
    ) -> RiskResult<PositionSizing> {
        let account_balance = 100000.0; // Placeholder - would come from account service
        
        // Kelly Criterion for optimal sizing
        let win_rate = 0.55; // Placeholder - would come from historical data
        let avg_win = 0.08;   // 8% average win
        let avg_loss = 0.04;  // 4% average loss
        
        let kelly_fraction = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_win;
        let kelly_fraction = kelly_fraction.max(0.0).min(0.25); // Cap at 25%

        // Adjust for volatility
        let volatility_adjusted_fraction = kelly_fraction * (1.0 - risk_metrics.volatility);
        
        // Calculate risk per trade (percentage of account)
        let risk_per_trade = self.config.protection.stop_loss.default_percentage;
        
        // Calculate position size based on risk
        let max_loss_per_trade = account_balance * risk_per_trade;
        let stop_loss_distance = (position.current_price - position.current_price * 0.95).abs();
        let position_size_by_risk = max_loss_per_trade / stop_loss_distance;

        // Use the more conservative of Kelly and risk-based sizing
        let recommended_size = position_size_by_risk.min(account_balance * volatility_adjusted_fraction);
        let max_size = account_balance * self.config.position.max_position_size;

        Ok(PositionSizing {
            max_size,
            recommended_size: recommended_size.min(max_size),
            risk_per_trade,
        })
    }

    /// Generate hedging strategy if needed
    async fn generate_hedging_strategy(
        &self,
        position: &Position,
        risk_metrics: &PositionRiskMetrics,
    ) -> RiskResult<Option<HedgingStrategy>> {
        if !self.config.protection.hedging.enabled {
            return Ok(None);
        }

        // Only hedge if position is large enough and risky enough
        let position_value = position.current_price * position.size.abs();
        if position_value < 10000.0 || risk_metrics.correlation_risk < 0.7 {
            return Ok(None);
        }

        // Calculate optimal hedge ratio
        let base_hedge_ratio = self.config.protection.hedging.hedge_ratio;
        let risk_adjusted_ratio = base_hedge_ratio * risk_metrics.volatility;

        // Select hedge instrument
        let hedge_instrument = if position.symbol.contains("SOL") {
            "SOL-PERP".to_string() // Hedge SOL positions with SOL perpetual
        } else {
            "BTC-PERP".to_string() // Default to BTC hedge
        };

        Ok(Some(HedgingStrategy {
            hedge_ratio: risk_adjusted_ratio,
            hedge_instrument,
            dynamic_adjustment: self.config.protection.hedging.dynamic_adjustment,
        }))
    }

    /// Estimate asset volatility
    async fn estimate_asset_volatility(&self, symbol: &str) -> RiskResult<f64> {
        // Simplified volatility estimation
        // In production, would use historical price data
        match symbol {
            s if s.contains("BTC") => Ok(0.40), // 40% volatility for BTC
            s if s.contains("SOL") => Ok(0.60), // 60% volatility for SOL
            s if s.contains("ETH") => Ok(0.50), // 50% volatility for ETH
            _ => Ok(0.70), // 70% for other altcoins
        }
    }

    /// Estimate liquidity score
    async fn estimate_liquidity_score(&self, symbol: &str) -> RiskResult<f64> {
        // Simplified liquidity scoring (0-1, higher is better)
        match symbol {
            s if s.contains("BTC") => Ok(0.95),
            s if s.contains("ETH") => Ok(0.90),
            s if s.contains("SOL") => Ok(0.85),
            _ => Ok(0.60),
        }
    }

    /// Estimate asset beta
    async fn estimate_asset_beta(&self, symbol: &str) -> RiskResult<f64> {
        // Simplified beta estimation relative to crypto market
        match symbol {
            s if s.contains("BTC") => Ok(1.0), // BTC as market proxy
            s if s.contains("ETH") => Ok(1.2),
            s if s.contains("SOL") => Ok(1.5),
            _ => Ok(2.0), // Higher beta for altcoins
        }
    }

    /// Calculate correlation risk
    async fn calculate_correlation_risk(&self, position: &Position) -> RiskResult<f64> {
        // Simplified correlation risk calculation
        // In production, would use correlation matrix with other positions
        
        // Assume moderate correlation for crypto assets
        match position.symbol.as_str() {
            s if s.contains("BTC") => Ok(0.6),
            s if s.contains("ETH") => Ok(0.7),
            s if s.contains("SOL") => Ok(0.8),
            _ => Ok(0.9), // High correlation for other altcoins
        }
    }
}

/// Position risk metrics
#[derive(Debug, Clone)]
pub struct PositionRiskMetrics {
    pub volatility: f64,
    pub liquidity_score: f64,
    pub beta: f64,
    pub var_1d: f64,
    pub correlation_risk: f64,
    pub unrealized_pnl_percent: f64,
}

/// Protection strategy optimizer
pub struct ProtectionOptimizer {
    config: RiskConfig,
}

impl ProtectionOptimizer {
    fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    async fn optimize_strategy(
        &self,
        position: &Position,
        current_strategy: &ProtectionStrategy,
        market_change: f64,
    ) -> RiskResult<ProtectionStrategy> {
        let mut optimized = current_strategy.clone();

        // Adjust stop loss based on market movement
        if let Some(ref mut stop_loss) = optimized.stop_loss {
            if market_change.abs() > 0.05 { // 5% market movement
                // Tighten stop loss in volatile conditions
                stop_loss.percentage *= 0.8;
                stop_loss.price = if position.size > 0.0 {
                    position.current_price * (1.0 - stop_loss.percentage)
                } else {
                    position.current_price * (1.0 + stop_loss.percentage)
                };
            }
        }

        // Adjust take profit based on momentum
        if let Some(ref mut take_profit) = optimized.take_profit {
            if market_change > 0.03 { // Positive momentum
                // Increase take profit target
                take_profit.percentage *= 1.2;
                take_profit.price = if position.size > 0.0 {
                    position.current_price * (1.0 + take_profit.percentage)
                } else {
                    position.current_price * (1.0 - take_profit.percentage)
                };
            }
        }

        debug!("Optimized protection strategy for {:.2}% market change", market_change * 100.0);
        Ok(optimized)
    }
}

/// Protection monitoring system
pub struct ProtectionMonitor {
    config: RiskConfig,
}

impl ProtectionMonitor {
    async fn new(config: &RiskConfig) -> RiskResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    async fn start_monitoring(
        &self,
        position: &Position,
        strategy: &ProtectionStrategy,
    ) -> RiskResult<()> {
        debug!("Started monitoring protection for position: {}", position.id);
        
        // In production, would start background monitoring tasks
        // For now, just log the monitoring start
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;
    use chrono::Utc;

    #[tokio::test]
    async fn test_protection_engine_creation() {
        let config = RiskConfig::default();
        let metrics = Arc::new(crate::metrics::MetricsCollector::new(&config.metrics).unwrap());
        
        let engine = ProtectionEngine::new(config, metrics).await.unwrap();
        
        let position = Position {
            id: Uuid::new_v4(),
            symbol: "SOL/USDC".to_string(),
            size: 100.0,
            entry_price: 50.0,
            current_price: 52.0,
            unrealized_pnl: 200.0,
            timestamp: Utc::now(),
        };
        
        let strategy = engine.generate_strategy(&position).await.unwrap();
        assert!(strategy.stop_loss.is_some());
    }

    #[tokio::test]
    async fn test_position_sizing() {
        let config = RiskConfig::default();
        let metrics = Arc::new(crate::metrics::MetricsCollector::new(&config.metrics).unwrap());
        let engine = ProtectionEngine::new(config, metrics).await.unwrap();
        
        let position = Position {
            id: Uuid::new_v4(),
            symbol: "BTC/USDC".to_string(),
            size: 1.0,
            entry_price: 50000.0,
            current_price: 51000.0,
            unrealized_pnl: 1000.0,
            timestamp: Utc::now(),
        };
        
        let risk_metrics = engine.calculate_position_risk(&position).await.unwrap();
        let sizing = engine.calculate_position_sizing(&position, &risk_metrics).await.unwrap();
        
        assert!(sizing.recommended_size > 0.0);
        assert!(sizing.recommended_size <= sizing.max_size);
    }
}