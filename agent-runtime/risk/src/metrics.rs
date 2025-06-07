//! Comprehensive metrics collection and monitoring for risk management

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use parking_lot::Mutex;
use chrono::{DateTime, Utc};
use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge, 
    register_counter, register_gauge, register_histogram, 
    register_int_counter, register_int_gauge,
    Encoder, TextEncoder,
};
use tracing::{info, warn, error, debug};

use crate::{
    config::MetricsConfig,
    RiskError, RiskResult, RiskDecision, RiskAssessment,
};

/// Comprehensive metrics collector for risk management
pub struct MetricsCollector {
    config: MetricsConfig,
    
    // Risk assessment metrics
    assessments_total: IntCounter,
    assessment_duration: Histogram,
    risk_score_distribution: Histogram,
    
    // Decision metrics
    decisions_approve: IntCounter,
    decisions_reject: IntCounter,
    decisions_approve_with_limits: IntCounter,
    decisions_defer: IntCounter,
    
    // Risk metrics gauges
    current_var_1d: Gauge,
    current_var_7d: Gauge,
    current_drawdown: Gauge,
    portfolio_value: Gauge,
    portfolio_beta: Gauge,
    
    // Circuit breaker metrics
    circuit_breaker_triggers: IntCounter,
    circuit_breaker_resets: IntCounter,
    circuit_breaker_status: IntGauge,
    
    // Position metrics
    active_positions: IntGauge,
    total_exposure: Gauge,
    concentration_risk: Gauge,
    liquidity_risk: Gauge,
    
    // Performance metrics
    calculation_errors: IntCounter,
    model_accuracy: Gauge,
    prediction_confidence: Histogram,
    
    // System metrics
    memory_usage: Gauge,
    cpu_usage: Gauge,
    
    // Custom metrics storage
    custom_metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    
    // Alert tracking
    alert_history: Arc<Mutex<Vec<AlertEvent>>>,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(config: &MetricsConfig) -> RiskResult<Self> {
        // Register Prometheus metrics
        let assessments_total = register_int_counter!(
            "prowzi_risk_assessments_total",
            "Total number of risk assessments performed"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let assessment_duration = register_histogram!(
            "prowzi_risk_assessment_duration_seconds",
            "Time taken to perform risk assessments",
            vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let risk_score_distribution = register_histogram!(
            "prowzi_risk_score_distribution",
            "Distribution of risk scores",
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let decisions_approve = register_int_counter!(
            "prowzi_risk_decisions_approve_total",
            "Total number of approved risk decisions"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let decisions_reject = register_int_counter!(
            "prowzi_risk_decisions_reject_total",
            "Total number of rejected risk decisions"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let decisions_approve_with_limits = register_int_counter!(
            "prowzi_risk_decisions_approve_with_limits_total",
            "Total number of approved with limits risk decisions"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let decisions_defer = register_int_counter!(
            "prowzi_risk_decisions_defer_total",
            "Total number of deferred risk decisions"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let current_var_1d = register_gauge!(
            "prowzi_risk_var_1d",
            "Current 1-day Value at Risk"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let current_var_7d = register_gauge!(
            "prowzi_risk_var_7d",
            "Current 7-day Value at Risk"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let current_drawdown = register_gauge!(
            "prowzi_risk_current_drawdown",
            "Current portfolio drawdown"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let portfolio_value = register_gauge!(
            "prowzi_portfolio_value_usd",
            "Current portfolio value in USD"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let portfolio_beta = register_gauge!(
            "prowzi_portfolio_beta",
            "Current portfolio beta"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let circuit_breaker_triggers = register_int_counter!(
            "prowzi_circuit_breaker_triggers_total",
            "Total number of circuit breaker triggers"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let circuit_breaker_resets = register_int_counter!(
            "prowzi_circuit_breaker_resets_total",
            "Total number of circuit breaker resets"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let circuit_breaker_status = register_int_gauge!(
            "prowzi_circuit_breaker_status",
            "Circuit breaker status (1=triggered, 0=normal)"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let active_positions = register_int_gauge!(
            "prowzi_active_positions",
            "Number of active positions"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let total_exposure = register_gauge!(
            "prowzi_total_exposure_usd",
            "Total portfolio exposure in USD"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let concentration_risk = register_gauge!(
            "prowzi_concentration_risk",
            "Current concentration risk score"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let liquidity_risk = register_gauge!(
            "prowzi_liquidity_risk",
            "Current liquidity risk score"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let calculation_errors = register_int_counter!(
            "prowzi_risk_calculation_errors_total",
            "Total number of risk calculation errors"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let model_accuracy = register_gauge!(
            "prowzi_risk_model_accuracy",
            "Current risk model accuracy"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let prediction_confidence = register_histogram!(
            "prowzi_risk_prediction_confidence",
            "Risk prediction confidence distribution",
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let memory_usage = register_gauge!(
            "prowzi_risk_memory_usage_bytes",
            "Current memory usage in bytes"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        let cpu_usage = register_gauge!(
            "prowzi_risk_cpu_usage_percent",
            "Current CPU usage percentage"
        ).map_err(|e| RiskError::Metrics(e.to_string()))?;

        Ok(Self {
            config: config.clone(),
            assessments_total,
            assessment_duration,
            risk_score_distribution,
            decisions_approve,
            decisions_reject,
            decisions_approve_with_limits,
            decisions_defer,
            current_var_1d,
            current_var_7d,
            current_drawdown,
            portfolio_value,
            portfolio_beta,
            circuit_breaker_triggers,
            circuit_breaker_resets,
            circuit_breaker_status,
            active_positions,
            total_exposure,
            concentration_risk,
            liquidity_risk,
            calculation_errors,
            model_accuracy,
            prediction_confidence,
            memory_usage,
            cpu_usage,
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Record a risk assessment
    pub async fn record_assessment(&self, assessment: &RiskAssessment) -> RiskResult<()> {
        self.assessments_total.inc();
        
        // Record prediction confidence
        self.prediction_confidence.observe(assessment.confidence);
        
        // Update current risk metrics
        self.current_var_1d.set(assessment.metrics.var_1d);
        self.current_var_7d.set(assessment.metrics.var_7d);
        self.current_drawdown.set(assessment.metrics.max_drawdown);
        self.portfolio_beta.set(assessment.metrics.portfolio_beta);
        self.concentration_risk.set(assessment.metrics.concentration_risk);
        self.liquidity_risk.set(assessment.metrics.liquidity_risk);

        if self.config.detailed_logging {
            debug!("Recorded risk assessment metrics: var_1d={:.4}, confidence={:.2}", 
                   assessment.metrics.var_1d, assessment.confidence);
        }

        Ok(())
    }

    /// Record assessment duration
    pub async fn record_assessment_duration(&self, duration: Duration) -> RiskResult<()> {
        self.assessment_duration.observe(duration.as_secs_f64());
        Ok(())
    }

    /// Record risk decision
    pub async fn record_risk_decision(&self, decision: &RiskDecision) -> RiskResult<()> {
        match decision {
            RiskDecision::Approve => self.decisions_approve.inc(),
            RiskDecision::Reject => self.decisions_reject.inc(),
            RiskDecision::ApproveWithLimits { .. } => self.decisions_approve_with_limits.inc(),
            RiskDecision::Defer { .. } => self.decisions_defer.inc(),
        }
        Ok(())
    }

    /// Record circuit breaker trigger
    pub async fn record_circuit_breaker_trigger(&self, reason: &str) -> RiskResult<()> {
        self.circuit_breaker_triggers.inc();
        self.circuit_breaker_status.set(1);
        
        // Record alert
        let alert = AlertEvent {
            timestamp: Utc::now(),
            alert_type: AlertType::CircuitBreakerTriggered,
            severity: AlertSeverity::Critical,
            message: format!("Circuit breaker triggered: {}", reason),
            metadata: HashMap::new(),
        };
        
        self.alert_history.lock().push(alert);
        
        warn!("Circuit breaker triggered: {}", reason);
        Ok(())
    }

    /// Record circuit breaker reset
    pub async fn record_circuit_breaker_reset(&self) -> RiskResult<()> {
        self.circuit_breaker_resets.inc();
        self.circuit_breaker_status.set(0);
        info!("Circuit breaker reset");
        Ok(())
    }

    /// Update portfolio metrics
    pub async fn update_portfolio_metrics(
        &self, 
        value: f64, 
        positions: usize, 
        exposure: f64
    ) -> RiskResult<()> {
        self.portfolio_value.set(value);
        self.active_positions.set(positions as i64);
        self.total_exposure.set(exposure);
        Ok(())
    }

    /// Record calculation error
    pub async fn record_calculation_error(&self, error: &str) -> RiskResult<()> {
        self.calculation_errors.inc();
        error!("Risk calculation error: {}", error);
        Ok(())
    }

    /// Update model accuracy
    pub async fn update_model_accuracy(&self, accuracy: f64) -> RiskResult<()> {
        self.model_accuracy.set(accuracy);
        Ok(())
    }

    /// Record custom metric
    pub async fn record_custom_metric(&self, name: String, value: MetricValue) -> RiskResult<()> {
        let mut metrics = self.custom_metrics.write().await;
        metrics.insert(name, value);
        Ok(())
    }

    /// Get metrics summary
    pub async fn get_metrics_summary(&self) -> RiskResult<MetricsSummary> {
        Ok(MetricsSummary {
            total_assessments: self.assessments_total.get(),
            approved_decisions: self.decisions_approve.get(),
            rejected_decisions: self.decisions_reject.get(),
            current_var_1d: self.current_var_1d.get(),
            current_drawdown: self.current_drawdown.get(),
            circuit_breaker_triggers: self.circuit_breaker_triggers.get(),
            active_positions: self.active_positions.get(),
            portfolio_value: self.portfolio_value.get(),
            calculation_errors: self.calculation_errors.get(),
            timestamp: Utc::now(),
        })
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus_metrics(&self) -> RiskResult<String> {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)
            .map_err(|e| RiskError::Metrics(e.to_string()))?;
            
        String::from_utf8(buffer)
            .map_err(|e| RiskError::Metrics(e.to_string()))
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, limit: usize) -> Vec<AlertEvent> {
        let alerts = self.alert_history.lock();
        alerts.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Start background metrics collection
    pub async fn start_background_collection(&self) -> RiskResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let collection_interval = Duration::from_secs(self.config.collection_interval_seconds);
        let memory_gauge = self.memory_usage.clone();
        let cpu_gauge = self.cpu_usage.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(collection_interval);
            
            loop {
                interval.tick().await;
                
                // Collect system metrics
                if let Ok(memory) = Self::get_memory_usage() {
                    memory_gauge.set(memory as f64);
                }
                
                if let Ok(cpu) = Self::get_cpu_usage() {
                    cpu_gauge.set(cpu);
                }
            }
        });

        info!("Started background metrics collection with {}s interval", 
              self.config.collection_interval_seconds);
        Ok(())
    }

    /// Get current memory usage
    fn get_memory_usage() -> RiskResult<u64> {
        // Simplified memory usage calculation
        // In production, would use proper system monitoring
        Ok(1024 * 1024 * 100) // 100MB placeholder
    }

    /// Get current CPU usage
    fn get_cpu_usage() -> RiskResult<f64> {
        // Simplified CPU usage calculation
        // In production, would use proper system monitoring
        Ok(5.0) // 5% placeholder
    }
}

/// Custom metric value types
#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Text(String),
}

/// Metrics summary for dashboards
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_assessments: u64,
    pub approved_decisions: u64,
    pub rejected_decisions: u64,
    pub current_var_1d: f64,
    pub current_drawdown: f64,
    pub circuit_breaker_triggers: u64,
    pub active_positions: i64,
    pub portfolio_value: f64,
    pub calculation_errors: u64,
    pub timestamp: DateTime<Utc>,
}

/// Alert event for monitoring
#[derive(Debug, Clone)]
pub struct AlertEvent {
    pub timestamp: DateTime<Utc>,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    RiskThresholdExceeded,
    CircuitBreakerTriggered,
    CalculationError,
    SystemError,
    PerformanceIssue,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Real-time metrics dashboard data
#[derive(Debug, Clone)]
pub struct DashboardMetrics {
    pub current_risk_score: f64,
    pub var_1d: f64,
    pub var_7d: f64,
    pub portfolio_value: f64,
    pub active_positions: i64,
    pub circuit_breaker_status: bool,
    pub recent_decisions: Vec<RiskDecisionSummary>,
    pub alerts: Vec<AlertEvent>,
    pub performance: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct RiskDecisionSummary {
    pub timestamp: DateTime<Utc>,
    pub decision: String,
    pub risk_score: f64,
    pub symbol: String,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub avg_assessment_time_ms: f64,
    pub assessments_per_second: f64,
    pub error_rate: f64,
    pub model_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MetricsConfig;

    #[test]
    fn test_metrics_collector_creation() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(&config).unwrap();
        
        // Test basic metric recording
        assert_eq!(collector.assessments_total.get(), 0);
    }

    #[tokio::test]
    async fn test_custom_metrics() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(&config).unwrap();
        
        collector.record_custom_metric(
            "test_metric".to_string(),
            MetricValue::Gauge(42.0)
        ).await.unwrap();
        
        let metrics = collector.custom_metrics.read().await;
        assert!(metrics.contains_key("test_metric"));
    }

    #[tokio::test]
    async fn test_metrics_summary() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(&config).unwrap();
        
        let summary = collector.get_metrics_summary().await.unwrap();
        assert_eq!(summary.total_assessments, 0);
    }
}