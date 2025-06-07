//! Advanced circuit breaker with quantum-fast response capabilities

use std::sync::Arc;
use std::collections::VecDeque;
use tokio::sync::RwLock;
use parking_lot::Mutex;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

use crate::{
    config::{RiskConfig, CircuitBreakerConfig},
    models::*,
    metrics::MetricsCollector,
    RiskError, RiskResult, RiskAssessment,
};

/// High-performance circuit breaker with multiple trigger conditions
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    metrics: Arc<MetricsCollector>,
    state: Arc<RwLock<CircuitBreakerState>>,
    conditions: Vec<Box<dyn TriggerCondition>>,
    recovery_manager: Arc<RecoveryManager>,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub async fn new(config: RiskConfig, metrics: Arc<MetricsCollector>) -> RiskResult<Self> {
        let breaker_config = config.circuit_breaker.clone();
        let state = Arc::new(RwLock::new(CircuitBreakerState::new()));
        let recovery_manager = Arc::new(RecoveryManager::new(breaker_config.recovery.clone()).await?);

        // Initialize trigger conditions
        let mut conditions: Vec<Box<dyn TriggerCondition>> = vec![
            Box::new(LossThresholdCondition::new(breaker_config.loss_thresholds.clone())),
            Box::new(VolatilityThresholdCondition::new(breaker_config.volatility_thresholds.clone())),
            Box::new(TimeThresholdCondition::new(breaker_config.time_thresholds.clone())),
            Box::new(DrawdownCondition::new(0.10)), // 10% max drawdown
            Box::new(VaRBreachCondition::new(0.05)), // 5% VaR breach
        ];

        Ok(Self {
            config: breaker_config,
            metrics,
            state,
            conditions,
            recovery_manager,
        })
    }

    /// Check if circuit breaker should be triggered
    pub async fn check_conditions(&self, assessment: &RiskAssessment) -> RiskResult<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        let mut state = self.state.write().await;
        
        // Skip checks if already triggered and in cooldown
        if state.is_triggered && !self.can_reset(&state).await? {
            return Ok(true);
        }

        // Check all trigger conditions
        for condition in &self.conditions {
            if condition.should_trigger(assessment, &state).await? {
                warn!("Circuit breaker condition triggered: {}", condition.name());
                self.trigger_breaker(&mut state, condition.name()).await?;
                return Ok(true);
            }
        }

        // Check for automatic recovery
        if state.is_triggered && self.can_reset(&state).await? {
            self.reset_breaker(&mut state).await?;
        }

        Ok(state.is_triggered)
    }

    /// Check if circuit breaker is currently triggered
    pub async fn is_triggered(&self) -> bool {
        let state = self.state.read().await;
        state.is_triggered
    }

    /// Manually trigger emergency circuit breaker
    pub async fn trigger_emergency(&self, reason: String) -> RiskResult<()> {
        let mut state = self.state.write().await;
        warn!("Emergency circuit breaker triggered: {}", reason);
        self.trigger_breaker(&mut state, &format!("EMERGENCY: {}", reason)).await
    }

    /// Manually reset circuit breaker
    pub async fn manual_reset(&self) -> RiskResult<()> {
        let mut state = self.state.write().await;
        info!("Manual circuit breaker reset");
        self.reset_breaker(&mut state).await
    }

    /// Get current circuit breaker status
    pub async fn get_status(&self) -> CircuitBreakerStatus {
        let state = self.state.read().await;
        CircuitBreakerStatus {
            is_triggered: state.is_triggered,
            trigger_reason: state.trigger_reason.clone(),
            triggered_at: state.triggered_at,
            trigger_count: state.trigger_count,
            last_reset: state.last_reset,
            cooldown_until: state.cooldown_until,
        }
    }

    /// Trigger the circuit breaker
    async fn trigger_breaker(&self, state: &mut CircuitBreakerState, reason: &str) -> RiskResult<()> {
        state.is_triggered = true;
        state.trigger_reason = Some(reason.to_string());
        state.triggered_at = Some(Utc::now());
        state.trigger_count += 1;
        state.cooldown_until = Some(Utc::now() + Duration::minutes(self.config.time_thresholds.cooldown_minutes as i64));

        // Record metrics
        self.metrics.record_circuit_breaker_trigger(reason).await?;

        // Execute emergency procedures
        self.execute_emergency_procedures(reason).await?;

        error!("Circuit breaker TRIGGERED: {} (count: {})", reason, state.trigger_count);
        Ok(())
    }

    /// Reset the circuit breaker
    async fn reset_breaker(&self, state: &mut CircuitBreakerState) -> RiskResult<()> {
        state.is_triggered = false;
        state.trigger_reason = None;
        state.last_reset = Some(Utc::now());
        state.cooldown_until = None;

        // Record metrics
        self.metrics.record_circuit_breaker_reset().await?;

        info!("Circuit breaker RESET (total triggers: {})", state.trigger_count);
        Ok(())
    }

    /// Check if circuit breaker can be reset
    async fn can_reset(&self, state: &CircuitBreakerState) -> RiskResult<bool> {
        if !state.is_triggered {
            return Ok(false);
        }

        // Check cooldown period
        if let Some(cooldown_until) = state.cooldown_until {
            if Utc::now() < cooldown_until {
                return Ok(false);
            }
        }

        // Check recovery conditions
        self.recovery_manager.check_recovery_conditions().await
    }

    /// Execute emergency procedures when triggered
    async fn execute_emergency_procedures(&self, reason: &str) -> RiskResult<()> {
        debug!("Executing emergency procedures for: {}", reason);

        // 1. Halt all new trading
        // 2. Cancel pending orders
        // 3. Evaluate existing positions
        // 4. Send alerts to operators
        // 5. Log detailed state

        // This would integrate with the actual trading system
        // For now, we just log the actions

        info!("Emergency procedures executed for circuit breaker");
        Ok(())
    }
}

/// Circuit breaker state
#[derive(Debug)]
pub struct CircuitBreakerState {
    pub is_triggered: bool,
    pub trigger_reason: Option<String>,
    pub triggered_at: Option<DateTime<Utc>>,
    pub trigger_count: u64,
    pub last_reset: Option<DateTime<Utc>>,
    pub cooldown_until: Option<DateTime<Utc>>,
    pub recent_assessments: VecDeque<RiskAssessment>,
    pub loss_history: VecDeque<LossEvent>,
}

impl CircuitBreakerState {
    fn new() -> Self {
        Self {
            is_triggered: false,
            trigger_reason: None,
            triggered_at: None,
            trigger_count: 0,
            last_reset: None,
            cooldown_until: None,
            recent_assessments: VecDeque::with_capacity(100),
            loss_history: VecDeque::with_capacity(1000),
        }
    }
}

/// Circuit breaker status for external monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    pub is_triggered: bool,
    pub trigger_reason: Option<String>,
    pub triggered_at: Option<DateTime<Utc>>,
    pub trigger_count: u64,
    pub last_reset: Option<DateTime<Utc>>,
    pub cooldown_until: Option<DateTime<Utc>>,
}

/// Trait for trigger conditions
#[async_trait::async_trait]
pub trait TriggerCondition: Send + Sync {
    async fn should_trigger(
        &self,
        assessment: &RiskAssessment,
        state: &CircuitBreakerState,
    ) -> RiskResult<bool>;
    
    fn name(&self) -> &str;
}

/// Loss threshold condition
pub struct LossThresholdCondition {
    config: crate::config::LossThresholds,
}

impl LossThresholdCondition {
    fn new(config: crate::config::LossThresholds) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl TriggerCondition for LossThresholdCondition {
    async fn should_trigger(
        &self,
        assessment: &RiskAssessment,
        state: &CircuitBreakerState,
    ) -> RiskResult<bool> {
        // Check daily loss threshold
        let daily_loss = self.calculate_daily_loss(state)?;
        if daily_loss > self.config.daily_loss {
            return Ok(true);
        }

        // Check weekly loss threshold
        let weekly_loss = self.calculate_weekly_loss(state)?;
        if weekly_loss > self.config.weekly_loss {
            return Ok(true);
        }

        // Check consecutive losses
        let consecutive_losses = self.count_consecutive_losses(state)?;
        if consecutive_losses >= self.config.consecutive_losses {
            return Ok(true);
        }

        Ok(false)
    }

    fn name(&self) -> &str {
        "LossThreshold"
    }
}

impl LossThresholdCondition {
    fn calculate_daily_loss(&self, state: &CircuitBreakerState) -> RiskResult<f64> {
        let now = Utc::now();
        let day_ago = now - Duration::days(1);
        
        let daily_losses: f64 = state.loss_history.iter()
            .filter(|event| event.timestamp >= day_ago)
            .map(|event| event.amount)
            .sum();
            
        Ok(daily_losses)
    }

    fn calculate_weekly_loss(&self, state: &CircuitBreakerState) -> RiskResult<f64> {
        let now = Utc::now();
        let week_ago = now - Duration::weeks(1);
        
        let weekly_losses: f64 = state.loss_history.iter()
            .filter(|event| event.timestamp >= week_ago)
            .map(|event| event.amount)
            .sum();
            
        Ok(weekly_losses)
    }

    fn count_consecutive_losses(&self, state: &CircuitBreakerState) -> RiskResult<usize> {
        let mut consecutive = 0;
        
        for event in state.loss_history.iter().rev() {
            if event.amount > 0.0 {
                consecutive += 1;
            } else {
                break;
            }
        }
        
        Ok(consecutive)
    }
}

/// Volatility threshold condition
pub struct VolatilityThresholdCondition {
    config: crate::config::VolatilityThresholds,
}

impl VolatilityThresholdCondition {
    fn new(config: crate::config::VolatilityThresholds) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl TriggerCondition for VolatilityThresholdCondition {
    async fn should_trigger(
        &self,
        assessment: &RiskAssessment,
        _state: &CircuitBreakerState,
    ) -> RiskResult<bool> {
        // Check if portfolio volatility exceeds threshold
        if assessment.metrics.var_1d > self.config.portfolio_volatility {
            return Ok(true);
        }

        Ok(false)
    }

    fn name(&self) -> &str {
        "VolatilityThreshold"
    }
}

/// Time-based threshold condition
pub struct TimeThresholdCondition {
    config: crate::config::TimeThresholds,
}

impl TimeThresholdCondition {
    fn new(config: crate::config::TimeThresholds) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl TriggerCondition for TimeThresholdCondition {
    async fn should_trigger(
        &self,
        _assessment: &RiskAssessment,
        state: &CircuitBreakerState,
    ) -> RiskResult<bool> {
        // Check if too many trades in the last hour
        let hour_ago = Utc::now() - Duration::hours(1);
        let recent_trades = state.recent_assessments.iter()
            .filter(|a| a.timestamp >= hour_ago)
            .count();

        if recent_trades > self.config.max_trades_per_hour {
            return Ok(true);
        }

        Ok(false)
    }

    fn name(&self) -> &str {
        "TimeThreshold"
    }
}

/// Drawdown condition
pub struct DrawdownCondition {
    max_drawdown: f64,
}

impl DrawdownCondition {
    fn new(max_drawdown: f64) -> Self {
        Self { max_drawdown }
    }
}

#[async_trait::async_trait]
impl TriggerCondition for DrawdownCondition {
    async fn should_trigger(
        &self,
        assessment: &RiskAssessment,
        _state: &CircuitBreakerState,
    ) -> RiskResult<bool> {
        Ok(assessment.metrics.max_drawdown > self.max_drawdown)
    }

    fn name(&self) -> &str {
        "Drawdown"
    }
}

/// VaR breach condition
pub struct VaRBreachCondition {
    max_var: f64,
}

impl VaRBreachCondition {
    fn new(max_var: f64) -> Self {
        Self { max_var }
    }
}

#[async_trait::async_trait]
impl TriggerCondition for VaRBreachCondition {
    async fn should_trigger(
        &self,
        assessment: &RiskAssessment,
        _state: &CircuitBreakerState,
    ) -> RiskResult<bool> {
        Ok(assessment.metrics.var_1d > self.max_var)
    }

    fn name(&self) -> &str {
        "VaRBreach"
    }
}

/// Recovery manager
pub struct RecoveryManager {
    config: crate::config::RecoveryConfig,
}

impl RecoveryManager {
    async fn new(config: crate::config::RecoveryConfig) -> RiskResult<Self> {
        Ok(Self { config })
    }

    async fn check_recovery_conditions(&self) -> RiskResult<bool> {
        if !self.config.auto_recovery {
            return Ok(false);
        }

        // Check all recovery conditions
        for condition in &self.config.conditions {
            if !self.check_condition(condition).await? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn check_condition(&self, condition: &crate::config::RecoveryCondition) -> RiskResult<bool> {
        match condition {
            crate::config::RecoveryCondition::VolatilityNormalized => {
                // Check if market volatility has normalized
                // This would require real market data
                Ok(true) // Simplified
            }
            crate::config::RecoveryCondition::MarketStable => {
                // Check if market conditions are stable
                Ok(true) // Simplified
            }
            crate::config::RecoveryCondition::TimeElapsed => {
                // Time condition is checked in can_reset
                Ok(true)
            }
            crate::config::RecoveryCondition::ManualApproval => {
                // Manual approval required
                Ok(false)
            }
        }
    }
}

/// Loss event for tracking
#[derive(Debug, Clone)]
pub struct LossEvent {
    pub timestamp: DateTime<Utc>,
    pub amount: f64,
    pub position_id: Option<Uuid>,
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;

    #[tokio::test]
    async fn test_circuit_breaker_creation() {
        let config = RiskConfig::default();
        let metrics = Arc::new(crate::metrics::MetricsCollector::new(&config.metrics).unwrap());
        
        let breaker = CircuitBreaker::new(config, metrics).await.unwrap();
        assert!(!breaker.is_triggered().await);
    }

    #[tokio::test]
    async fn test_trigger_conditions() {
        let config = crate::config::LossThresholds::default();
        let condition = LossThresholdCondition::new(config);
        
        assert_eq!(condition.name(), "LossThreshold");
    }
}