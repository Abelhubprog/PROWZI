//! Real-time risk monitoring and alerting system

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, mpsc, broadcast};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

use crate::{
    config::RiskConfig,
    models::*,
    metrics::{MetricsCollector, AlertEvent, AlertType, AlertSeverity},
    RiskError, RiskResult, RiskAssessment, Position,
};

/// Real-time risk monitoring system
pub struct RiskMonitor {
    config: RiskConfig,
    metrics: Arc<MetricsCollector>,
    alert_sender: broadcast::Sender<AlertEvent>,
    thresholds: Arc<RwLock<MonitoringThresholds>>,
    alert_history: Arc<RwLock<VecDeque<AlertEvent>>>,
    active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
}

impl RiskMonitor {
    /// Create new risk monitor
    pub async fn new(config: RiskConfig, metrics: Arc<MetricsCollector>) -> RiskResult<Self> {
        let (alert_sender, _) = broadcast::channel(1000);
        let thresholds = Arc::new(RwLock::new(MonitoringThresholds::from_config(&config)));
        let alert_history = Arc::new(RwLock::new(VecDeque::with_capacity(10000)));
        let active_alerts = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            metrics,
            alert_sender,
            thresholds,
            alert_history,
            active_alerts,
        })
    }

    /// Start monitoring risk assessments
    pub async fn start_monitoring(&self) -> RiskResult<()> {
        info!("Starting risk monitoring system");

        // Start background monitoring tasks
        self.start_threshold_monitoring().await?;
        self.start_alert_cleanup().await?;
        self.start_health_check().await?;

        Ok(())
    }

    /// Process a new risk assessment
    pub async fn process_assessment(&self, assessment: &RiskAssessment) -> RiskResult<()> {
        // Check all monitoring thresholds
        self.check_var_thresholds(assessment).await?;
        self.check_drawdown_thresholds(assessment).await?;
        self.check_concentration_thresholds(assessment).await?;
        self.check_liquidity_thresholds(assessment).await?;

        // Update monitoring metrics
        self.update_monitoring_metrics(assessment).await?;

        Ok(())
    }

    /// Subscribe to alerts
    pub fn subscribe_alerts(&self) -> broadcast::Receiver<AlertEvent> {
        self.alert_sender.subscribe()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> HashMap<String, ActiveAlert> {
        self.active_alerts.read().await.clone()
    }

    /// Get alert history
    pub async fn get_alert_history(&self, limit: usize) -> Vec<AlertEvent> {
        let history = self.alert_history.read().await;
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Update monitoring thresholds
    pub async fn update_thresholds(&self, thresholds: MonitoringThresholds) -> RiskResult<()> {
        let mut current_thresholds = self.thresholds.write().await;
        *current_thresholds = thresholds;
        info!("Updated monitoring thresholds");
        Ok(())
    }

    /// Check VaR thresholds
    async fn check_var_thresholds(&self, assessment: &RiskAssessment) -> RiskResult<()> {
        let thresholds = self.thresholds.read().await;
        
        // Check 1-day VaR
        if assessment.metrics.var_1d > thresholds.var_1d_critical {
            self.send_alert(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::RiskThresholdExceeded,
                severity: AlertSeverity::Critical,
                message: format!("Critical VaR-1D threshold exceeded: {:.4} > {:.4}", 
                                assessment.metrics.var_1d, thresholds.var_1d_critical),
                metadata: HashMap::new(),
            }).await?;
        } else if assessment.metrics.var_1d > thresholds.var_1d_warning {
            self.send_alert(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::RiskThresholdExceeded,
                severity: AlertSeverity::Warning,
                message: format!("Warning VaR-1D threshold exceeded: {:.4} > {:.4}", 
                                assessment.metrics.var_1d, thresholds.var_1d_warning),
                metadata: HashMap::new(),
            }).await?;
        }

        // Check 7-day VaR
        if assessment.metrics.var_7d > thresholds.var_7d_critical {
            self.send_alert(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::RiskThresholdExceeded,
                severity: AlertSeverity::Critical,
                message: format!("Critical VaR-7D threshold exceeded: {:.4} > {:.4}", 
                                assessment.metrics.var_7d, thresholds.var_7d_critical),
                metadata: HashMap::new(),
            }).await?;
        }

        Ok(())
    }

    /// Check drawdown thresholds
    async fn check_drawdown_thresholds(&self, assessment: &RiskAssessment) -> RiskResult<()> {
        let thresholds = self.thresholds.read().await;

        if assessment.metrics.max_drawdown > thresholds.max_drawdown_critical {
            self.send_alert(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::RiskThresholdExceeded,
                severity: AlertSeverity::Critical,
                message: format!("Critical drawdown threshold exceeded: {:.2}% > {:.2}%", 
                                assessment.metrics.max_drawdown * 100.0, 
                                thresholds.max_drawdown_critical * 100.0),
                metadata: HashMap::new(),
            }).await?;
        } else if assessment.metrics.max_drawdown > thresholds.max_drawdown_warning {
            self.send_alert(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::RiskThresholdExceeded,
                severity: AlertSeverity::Warning,
                message: format!("Warning drawdown threshold exceeded: {:.2}% > {:.2}%", 
                                assessment.metrics.max_drawdown * 100.0, 
                                thresholds.max_drawdown_warning * 100.0),
                metadata: HashMap::new(),
            }).await?;
        }

        Ok(())
    }

    /// Check concentration thresholds
    async fn check_concentration_thresholds(&self, assessment: &RiskAssessment) -> RiskResult<()> {
        let thresholds = self.thresholds.read().await;

        if assessment.metrics.concentration_risk > thresholds.concentration_critical {
            self.send_alert(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::RiskThresholdExceeded,
                severity: AlertSeverity::Critical,
                message: format!("Critical concentration risk: {:.2}% > {:.2}%", 
                                assessment.metrics.concentration_risk * 100.0, 
                                thresholds.concentration_critical * 100.0),
                metadata: HashMap::new(),
            }).await?;
        }

        Ok(())
    }

    /// Check liquidity thresholds
    async fn check_liquidity_thresholds(&self, assessment: &RiskAssessment) -> RiskResult<()> {
        let thresholds = self.thresholds.read().await;

        if assessment.metrics.liquidity_risk > thresholds.liquidity_critical {
            self.send_alert(AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::RiskThresholdExceeded,
                severity: AlertSeverity::Critical,
                message: format!("Critical liquidity risk: {:.2}% > {:.2}%", 
                                assessment.metrics.liquidity_risk * 100.0, 
                                thresholds.liquidity_critical * 100.0),
                metadata: HashMap::new(),
            }).await?;
        }

        Ok(())
    }

    /// Send alert
    async fn send_alert(&self, alert: AlertEvent) -> RiskResult<()> {
        let alert_key = format!("{}_{}", alert.alert_type as u8, alert.message.chars().take(50).collect::<String>());
        
        // Check if similar alert is already active (to prevent spam)
        {
            let active_alerts = self.active_alerts.read().await;
            if let Some(existing) = active_alerts.get(&alert_key) {
                let time_since_last = Utc::now() - existing.first_seen;
                if time_since_last < Duration::minutes(5) {
                    // Don't send duplicate alerts within 5 minutes
                    return Ok(());
                }
            }
        }

        // Add to active alerts
        {
            let mut active_alerts = self.active_alerts.write().await;
            active_alerts.insert(alert_key.clone(), ActiveAlert {
                alert: alert.clone(),
                first_seen: Utc::now(),
                last_seen: Utc::now(),
                count: 1,
            });
        }

        // Add to history
        {
            let mut history = self.alert_history.write().await;
            history.push_back(alert.clone());
            
            // Keep only last 10000 alerts
            if history.len() > 10000 {
                history.pop_front();
            }
        }

        // Send alert to subscribers
        if let Err(_) = self.alert_sender.send(alert.clone()) {
            debug!("No alert subscribers active");
        }

        // Record in metrics
        self.metrics.record_custom_metric(
            "alerts_sent_total".to_string(),
            crate::metrics::MetricValue::Counter(1),
        ).await?;

        match alert.severity {
            AlertSeverity::Critical => error!("CRITICAL ALERT: {}", alert.message),
            AlertSeverity::Error => error!("ERROR ALERT: {}", alert.message),
            AlertSeverity::Warning => warn!("WARNING ALERT: {}", alert.message),
            AlertSeverity::Info => info!("INFO ALERT: {}", alert.message),
        }

        Ok(())
    }

    /// Update monitoring metrics
    async fn update_monitoring_metrics(&self, assessment: &RiskAssessment) -> RiskResult<()> {
        // Record current risk levels
        self.metrics.record_custom_metric(
            "current_var_1d".to_string(),
            crate::metrics::MetricValue::Gauge(assessment.metrics.var_1d),
        ).await?;

        self.metrics.record_custom_metric(
            "current_drawdown".to_string(),
            crate::metrics::MetricValue::Gauge(assessment.metrics.max_drawdown),
        ).await?;

        self.metrics.record_custom_metric(
            "current_concentration_risk".to_string(),
            crate::metrics::MetricValue::Gauge(assessment.metrics.concentration_risk),
        ).await?;

        Ok(())
    }

    /// Start threshold monitoring background task
    async fn start_threshold_monitoring(&self) -> RiskResult<()> {
        let thresholds = self.thresholds.clone();
        let alert_sender = self.alert_sender.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Check for threshold violations that require time-based monitoring
                // This would integrate with real-time data feeds in production
                
                debug!("Threshold monitoring cycle completed");
            }
        });

        Ok(())
    }

    /// Start alert cleanup background task
    async fn start_alert_cleanup(&self) -> RiskResult<()> {
        let active_alerts = self.active_alerts.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes
            
            loop {
                interval.tick().await;
                
                // Clean up old active alerts
                let mut alerts = active_alerts.write().await;
                let cutoff_time = Utc::now() - Duration::hours(1);
                
                alerts.retain(|_, alert| alert.last_seen > cutoff_time);
                
                debug!("Alert cleanup completed, {} active alerts remaining", alerts.len());
            }
        });

        Ok(())
    }

    /// Start health check background task
    async fn start_health_check(&self) -> RiskResult<()> {
        let metrics = self.metrics.clone();
        let alert_sender = self.alert_sender.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(120)); // 2 minutes
            
            loop {
                interval.tick().await;
                
                // Check system health
                if let Err(e) = Self::check_system_health(&metrics, &alert_sender).await {
                    error!("Health check failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Check system health
    async fn check_system_health(
        metrics: &Arc<MetricsCollector>,
        alert_sender: &broadcast::Sender<AlertEvent>,
    ) -> RiskResult<()> {
        // Check memory usage
        if let Ok(memory_usage) = std::fs::read_to_string("/proc/meminfo") {
            if memory_usage.contains("MemAvailable") {
                // Parse available memory and alert if low
                // Simplified check for demo
            }
        }

        // Check if metrics collection is working
        let summary = metrics.get_metrics_summary().await?;
        
        // Alert if no assessments in the last hour (indicates system issues)
        if summary.total_assessments == 0 {
            let alert = AlertEvent {
                timestamp: Utc::now(),
                alert_type: AlertType::SystemError,
                severity: AlertSeverity::Warning,
                message: "No risk assessments processed recently".to_string(),
                metadata: HashMap::new(),
            };
            
            let _ = alert_sender.send(alert);
        }

        Ok(())
    }
}

/// Monitoring thresholds configuration
#[derive(Debug, Clone)]
pub struct MonitoringThresholds {
    pub var_1d_warning: f64,
    pub var_1d_critical: f64,
    pub var_7d_warning: f64,
    pub var_7d_critical: f64,
    pub max_drawdown_warning: f64,
    pub max_drawdown_critical: f64,
    pub concentration_warning: f64,
    pub concentration_critical: f64,
    pub liquidity_warning: f64,
    pub liquidity_critical: f64,
}

impl MonitoringThresholds {
    fn from_config(config: &RiskConfig) -> Self {
        Self {
            var_1d_warning: config.assessment.max_var_1d * 0.8,
            var_1d_critical: config.assessment.max_var_1d,
            var_7d_warning: config.assessment.max_var_7d * 0.8,
            var_7d_critical: config.assessment.max_var_7d,
            max_drawdown_warning: config.assessment.max_drawdown * 0.8,
            max_drawdown_critical: config.assessment.max_drawdown,
            concentration_warning: config.assessment.max_position_concentration * 0.8,
            concentration_critical: config.assessment.max_position_concentration,
            liquidity_warning: (1.0 - config.assessment.min_liquidity_ratio) * 0.8,
            liquidity_critical: 1.0 - config.assessment.min_liquidity_ratio,
        }
    }
}

/// Active alert tracking
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub alert: AlertEvent,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RiskConfig;

    #[tokio::test]
    async fn test_risk_monitor_creation() {
        let config = RiskConfig::default();
        let metrics = Arc::new(crate::metrics::MetricsCollector::new(&config.metrics).unwrap());
        
        let monitor = RiskMonitor::new(config, metrics).await.unwrap();
        let active_alerts = monitor.get_active_alerts().await;
        assert!(active_alerts.is_empty());
    }

    #[tokio::test]
    async fn test_alert_subscription() {
        let config = RiskConfig::default();
        let metrics = Arc::new(crate::metrics::MetricsCollector::new(&config.metrics).unwrap());
        let monitor = RiskMonitor::new(config, metrics).await.unwrap();
        
        let mut receiver = monitor.subscribe_alerts();
        
        // This would normally receive alerts, but we can't easily test without
        // triggering actual alerts in this test environment
        assert!(receiver.try_recv().is_err()); // No alerts yet
    }
}