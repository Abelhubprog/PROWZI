//! Performance Monitoring Module for Prowzi Agent Runtime
//! 
//! Provides comprehensive performance tracking, optimization recommendations,
//! and system health monitoring for autonomous trading agents.

use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Performance metrics for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Agent identifier
    pub agent_id: String,
    /// CPU usage percentage (0.0 - 100.0)
    pub cpu_usage_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: u64,
    /// Network I/O bytes per second
    pub network_io_bps: u64,
    /// Disk I/O bytes per second
    pub disk_io_bps: u64,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Throughput (operations per second)
    pub throughput_ops: f64,
    /// Error rate percentage
    pub error_rate_percent: f64,
    /// Latency percentiles
    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
    pub latency_p99_ms: u64,
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
}

/// System-wide performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformance {
    /// Total CPU usage across all agents
    pub total_cpu_usage_percent: f64,
    /// Total memory usage in MB
    pub total_memory_usage_mb: u64,
    /// Available memory in MB
    pub available_memory_mb: u64,
    /// Total network bandwidth usage
    pub total_network_bps: u64,
    /// Total disk I/O
    pub total_disk_io_bps: u64,
    /// Number of active agents
    pub active_agents: u32,
    /// System load average
    pub load_average: [f64; 3], // 1min, 5min, 15min
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
}

/// Performance alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceAlert {
    HighCpuUsage {
        agent_id: String,
        usage_percent: f64,
        threshold: f64,
    },
    HighMemoryUsage {
        agent_id: String,
        usage_mb: u64,
        available_mb: u64,
    },
    HighLatency {
        agent_id: String,
        latency_ms: u64,
        threshold_ms: u64,
    },
    LowThroughput {
        agent_id: String,
        current_ops: f64,
        expected_ops: f64,
    },
    HighErrorRate {
        agent_id: String,
        error_rate_percent: f64,
        threshold_percent: f64,
    },
    SystemOverload {
        cpu_usage: f64,
        memory_usage: f64,
        active_agents: u32,
    },
    ResourceExhaustion {
        resource: String,
        usage_percent: f64,
    },
}

/// Performance optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_id: String,
    pub agent_id: String,
    pub category: OptimizationCategory,
    pub priority: OptimizationPriority,
    pub description: String,
    pub expected_improvement: String,
    pub implementation_effort: EffortLevel,
    pub created_at: DateTime<Utc>,
}

/// Optimization categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    CpuOptimization,
    MemoryOptimization,
    NetworkOptimization,
    AlgorithmicOptimization,
    ResourceAllocation,
    Caching,
    Parallelization,
}

/// Optimization priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
}

/// Performance thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_cpu_usage_percent: f64,
    pub max_memory_usage_mb: u64,
    pub max_response_time_ms: u64,
    pub min_throughput_ops: f64,
    pub max_error_rate_percent: f64,
    pub max_latency_p95_ms: u64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_cpu_usage_percent: 80.0,
            max_memory_usage_mb: 1024,
            max_response_time_ms: 1000,
            min_throughput_ops: 10.0,
            max_error_rate_percent: 5.0,
            max_latency_p95_ms: 500,
        }
    }
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64, // 0.0 to 1.0
    pub confidence: f64, // 0.0 to 1.0
    pub predicted_value: f64,
    pub time_horizon_minutes: u32,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Performance monitor for the agent runtime
pub struct PerformanceMonitor {
    thresholds: PerformanceThresholds,
    metrics_history: Arc<RwLock<HashMap<String, VecDeque<PerformanceMetrics>>>>,
    system_history: Arc<RwLock<VecDeque<SystemPerformance>>>,
    alerts: Arc<RwLock<Vec<(DateTime<Utc>, PerformanceAlert)>>>,
    recommendations: Arc<RwLock<Vec<OptimizationRecommendation>>>,
    max_history_size: usize,
}

impl PerformanceMonitor {
    pub fn new(thresholds: PerformanceThresholds, max_history_size: usize) -> Self {
        Self {
            thresholds,
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            system_history: Arc::new(RwLock::new(VecDeque::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            recommendations: Arc::new(RwLock::new(Vec::new())),
            max_history_size,
        }
    }

    /// Record performance metrics for an agent
    pub async fn record_metrics(&self, metrics: PerformanceMetrics) -> Result<(), PerformanceError> {
        let agent_id = metrics.agent_id.clone();
        
        // Store metrics in history
        {
            let mut history = self.metrics_history.write().await;
            let agent_history = history.entry(agent_id.clone()).or_insert_with(VecDeque::new);
            agent_history.push_back(metrics.clone());
            
            // Limit history size
            while agent_history.len() > self.max_history_size {
                agent_history.pop_front();
            }
        }

        // Check for alerts
        self.check_thresholds(&metrics).await?;

        // Generate recommendations if needed
        self.generate_recommendations(&agent_id).await?;

        Ok(())
    }

    /// Record system-wide performance metrics
    pub async fn record_system_metrics(&self, system_metrics: SystemPerformance) -> Result<(), PerformanceError> {
        let mut history = self.system_history.write().await;
        history.push_back(system_metrics.clone());

        // Limit history size
        while history.len() > self.max_history_size {
            history.pop_front();
        }

        // Check for system-level alerts
        if system_metrics.total_cpu_usage_percent > 90.0 ||
           system_metrics.available_memory_mb < 512 {
            let alert = PerformanceAlert::SystemOverload {
                cpu_usage: system_metrics.total_cpu_usage_percent,
                memory_usage: ((system_metrics.total_memory_usage_mb as f64 / 
                               (system_metrics.total_memory_usage_mb + system_metrics.available_memory_mb) as f64) * 100.0),
                active_agents: system_metrics.active_agents,
            };
            
            let mut alerts = self.alerts.write().await;
            alerts.push((Utc::now(), alert));
        }

        Ok(())
    }

    /// Get performance metrics for an agent
    pub async fn get_agent_metrics(&self, agent_id: &str, limit: Option<usize>) -> Vec<PerformanceMetrics> {
        let history = self.metrics_history.read().await;
        if let Some(agent_metrics) = history.get(agent_id) {
            let metrics: Vec<PerformanceMetrics> = agent_metrics.iter().cloned().collect();
            if let Some(limit) = limit {
                metrics.into_iter().rev().take(limit).collect()
            } else {
                metrics
            }
        } else {
            Vec::new()
        }
    }

    /// Get system performance history
    pub async fn get_system_metrics(&self, limit: Option<usize>) -> Vec<SystemPerformance> {
        let history = self.system_history.read().await;
        let metrics: Vec<SystemPerformance> = history.iter().cloned().collect();
        if let Some(limit) = limit {
            metrics.into_iter().rev().take(limit).collect()
        } else {
            metrics
        }
    }

    /// Get recent performance alerts
    pub async fn get_alerts(&self, severity_filter: Option<OptimizationPriority>) -> Vec<(DateTime<Utc>, PerformanceAlert)> {
        let alerts = self.alerts.read().await;
        if let Some(_filter) = severity_filter {
            // TODO: Implement severity filtering based on alert type
            alerts.clone()
        } else {
            alerts.clone()
        }
    }

    /// Get optimization recommendations
    pub async fn get_recommendations(&self, agent_id: Option<&str>) -> Vec<OptimizationRecommendation> {
        let recommendations = self.recommendations.read().await;
        if let Some(agent_id) = agent_id {
            recommendations.iter()
                .filter(|rec| rec.agent_id == agent_id)
                .cloned()
                .collect()
        } else {
            recommendations.clone()
        }
    }

    /// Analyze performance trends
    pub async fn analyze_trends(&self, agent_id: &str) -> Vec<PerformanceTrend> {
        let metrics = self.get_agent_metrics(agent_id, Some(50)).await;
        if metrics.len() < 10 {
            return Vec::new(); // Not enough data for trend analysis
        }

        let mut trends = Vec::new();

        // Analyze CPU usage trend
        let cpu_values: Vec<f64> = metrics.iter().map(|m| m.cpu_usage_percent).collect();
        if let Some(trend) = self.calculate_trend("cpu_usage", &cpu_values) {
            trends.push(trend);
        }

        // Analyze response time trend
        let response_times: Vec<f64> = metrics.iter().map(|m| m.response_time_ms as f64).collect();
        if let Some(trend) = self.calculate_trend("response_time", &response_times) {
            trends.push(trend);
        }

        // Analyze throughput trend
        let throughput_values: Vec<f64> = metrics.iter().map(|m| m.throughput_ops).collect();
        if let Some(trend) = self.calculate_trend("throughput", &throughput_values) {
            trends.push(trend);
        }

        trends
    }

    /// Calculate performance summary for an agent
    pub async fn get_performance_summary(&self, agent_id: &str) -> Option<PerformanceSummary> {
        let metrics = self.get_agent_metrics(agent_id, Some(100)).await;
        if metrics.is_empty() {
            return None;
        }

        let avg_cpu = metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / metrics.len() as f64;
        let avg_memory = metrics.iter().map(|m| m.memory_usage_mb).sum::<u64>() / metrics.len() as u64;
        let avg_response_time = metrics.iter().map(|m| m.response_time_ms).sum::<u64>() / metrics.len() as u64;
        let avg_throughput = metrics.iter().map(|m| m.throughput_ops).sum::<f64>() / metrics.len() as f64;
        let avg_error_rate = metrics.iter().map(|m| m.error_rate_percent).sum::<f64>() / metrics.len() as f64;

        Some(PerformanceSummary {
            agent_id: agent_id.to_string(),
            avg_cpu_usage_percent: avg_cpu,
            avg_memory_usage_mb: avg_memory,
            avg_response_time_ms: avg_response_time,
            avg_throughput_ops: avg_throughput,
            avg_error_rate_percent: avg_error_rate,
            total_measurements: metrics.len() as u32,
            time_period_hours: if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
                (last.timestamp - first.timestamp).num_hours() as u32
            } else { 0 },
        })
    }

    /// Check performance thresholds and generate alerts
    async fn check_thresholds(&self, metrics: &PerformanceMetrics) -> Result<(), PerformanceError> {
        let mut alerts_to_add = Vec::new();

        // Check CPU usage
        if metrics.cpu_usage_percent > self.thresholds.max_cpu_usage_percent {
            alerts_to_add.push(PerformanceAlert::HighCpuUsage {
                agent_id: metrics.agent_id.clone(),
                usage_percent: metrics.cpu_usage_percent,
                threshold: self.thresholds.max_cpu_usage_percent,
            });
        }

        // Check memory usage
        if metrics.memory_usage_mb > self.thresholds.max_memory_usage_mb {
            alerts_to_add.push(PerformanceAlert::HighMemoryUsage {
                agent_id: metrics.agent_id.clone(),
                usage_mb: metrics.memory_usage_mb,
                available_mb: 0, // TODO: Get actual available memory
            });
        }

        // Check latency
        if metrics.latency_p95_ms > self.thresholds.max_latency_p95_ms {
            alerts_to_add.push(PerformanceAlert::HighLatency {
                agent_id: metrics.agent_id.clone(),
                latency_ms: metrics.latency_p95_ms,
                threshold_ms: self.thresholds.max_latency_p95_ms,
            });
        }

        // Check throughput
        if metrics.throughput_ops < self.thresholds.min_throughput_ops {
            alerts_to_add.push(PerformanceAlert::LowThroughput {
                agent_id: metrics.agent_id.clone(),
                current_ops: metrics.throughput_ops,
                expected_ops: self.thresholds.min_throughput_ops,
            });
        }

        // Check error rate
        if metrics.error_rate_percent > self.thresholds.max_error_rate_percent {
            alerts_to_add.push(PerformanceAlert::HighErrorRate {
                agent_id: metrics.agent_id.clone(),
                error_rate_percent: metrics.error_rate_percent,
                threshold_percent: self.thresholds.max_error_rate_percent,
            });
        }

        // Add alerts
        if !alerts_to_add.is_empty() {
            let mut alerts = self.alerts.write().await;
            let now = Utc::now();
            for alert in alerts_to_add {
                alerts.push((now, alert));
            }

            // Keep only recent alerts (last 24 hours)
            let cutoff = now - Duration::hours(24);
            alerts.retain(|(timestamp, _)| *timestamp > cutoff);
        }

        Ok(())
    }

    /// Generate optimization recommendations based on performance data
    async fn generate_recommendations(&self, agent_id: &str) -> Result<(), PerformanceError> {
        let recent_metrics = self.get_agent_metrics(agent_id, Some(10)).await;
        if recent_metrics.is_empty() {
            return Ok(());
        }

        let avg_cpu = recent_metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / recent_metrics.len() as f64;
        let avg_memory = recent_metrics.iter().map(|m| m.memory_usage_mb).sum::<u64>() / recent_metrics.len() as u64;
        let avg_latency = recent_metrics.iter().map(|m| m.latency_p95_ms).sum::<u64>() / recent_metrics.len() as u64;

        let mut new_recommendations = Vec::new();

        // CPU optimization recommendations
        if avg_cpu > 70.0 {
            new_recommendations.push(OptimizationRecommendation {
                recommendation_id: Uuid::new_v4().to_string(),
                agent_id: agent_id.to_string(),
                category: OptimizationCategory::CpuOptimization,
                priority: if avg_cpu > 90.0 { OptimizationPriority::Critical } else { OptimizationPriority::High },
                description: "High CPU usage detected. Consider optimizing computational algorithms or implementing caching.".to_string(),
                expected_improvement: format!("Reduce CPU usage by 20-30%"),
                implementation_effort: EffortLevel::Medium,
                created_at: Utc::now(),
            });
        }

        // Memory optimization recommendations
        if avg_memory > 800 {
            new_recommendations.push(OptimizationRecommendation {
                recommendation_id: Uuid::new_v4().to_string(),
                agent_id: agent_id.to_string(),
                category: OptimizationCategory::MemoryOptimization,
                priority: OptimizationPriority::High,
                description: "High memory usage detected. Consider implementing memory pooling or reducing data retention.".to_string(),
                expected_improvement: "Reduce memory usage by 25-40%".to_string(),
                implementation_effort: EffortLevel::Medium,
                created_at: Utc::now(),
            });
        }

        // Latency optimization recommendations
        if avg_latency > 300 {
            new_recommendations.push(OptimizationRecommendation {
                recommendation_id: Uuid::new_v4().to_string(),
                agent_id: agent_id.to_string(),
                category: OptimizationCategory::NetworkOptimization,
                priority: OptimizationPriority::Medium,
                description: "High latency detected. Consider implementing request batching or connection pooling.".to_string(),
                expected_improvement: "Reduce latency by 30-50%".to_string(),
                implementation_effort: EffortLevel::Low,
                created_at: Utc::now(),
            });
        }

        // Add recommendations
        if !new_recommendations.is_empty() {
            let mut recommendations = self.recommendations.write().await;
            recommendations.extend(new_recommendations);

            // Keep only recent recommendations (last 7 days)
            let cutoff = Utc::now() - Duration::days(7);
            recommendations.retain(|rec| rec.created_at > cutoff);
        }

        Ok(())
    }

    /// Calculate trend for a metric
    fn calculate_trend(&self, metric_name: &str, values: &[f64]) -> Option<PerformanceTrend> {
        if values.len() < 5 {
            return None;
        }

        // Simple linear regression for trend calculation
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x_squared_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum.powi(2));
        let intercept = (y_sum - slope * x_sum) / n;

        let trend_direction = if slope > 0.1 {
            TrendDirection::Degrading // For most metrics, increasing is bad
        } else if slope < -0.1 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };

        // Calculate R-squared for confidence
        let y_mean = y_sum / n;
        let ss_tot: f64 = values.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = values.iter().enumerate()
            .map(|(i, &y)| (y - (slope * i as f64 + intercept)).powi(2))
            .sum();
        let r_squared = 1.0 - (ss_res / ss_tot);

        let predicted_value = slope * (values.len() as f64) + intercept;

        Some(PerformanceTrend {
            metric_name: metric_name.to_string(),
            trend_direction,
            trend_strength: slope.abs(),
            confidence: r_squared.max(0.0).min(1.0),
            predicted_value,
            time_horizon_minutes: 60, // 1 hour prediction
        })
    }

    /// Update performance thresholds
    pub fn update_thresholds(&mut self, thresholds: PerformanceThresholds) {
        self.thresholds = thresholds;
    }

    /// Get current thresholds
    pub fn get_thresholds(&self) -> &PerformanceThresholds {
        &self.thresholds
    }
}

/// Performance summary for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub agent_id: String,
    pub avg_cpu_usage_percent: f64,
    pub avg_memory_usage_mb: u64,
    pub avg_response_time_ms: u64,
    pub avg_throughput_ops: f64,
    pub avg_error_rate_percent: f64,
    pub total_measurements: u32,
    pub time_period_hours: u32,
}

/// Performance-related errors
#[derive(Debug, thiserror::Error)]
pub enum PerformanceError {
    #[error("Insufficient data for analysis")]
    InsufficientData,

    #[error("Metric collection failed: {0}")]
    CollectionFailed(String),

    #[error("Invalid threshold configuration: {0}")]
    InvalidThreshold(String),

    #[error("Trend analysis failed: {0}")]
    TrendAnalysisFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_monitoring() {
        let thresholds = PerformanceThresholds::default();
        let monitor = PerformanceMonitor::new(thresholds, 100);

        let metrics = PerformanceMetrics {
            agent_id: "test_agent".to_string(),
            cpu_usage_percent: 45.0,
            memory_usage_mb: 512,
            network_io_bps: 1024,
            disk_io_bps: 512,
            response_time_ms: 150,
            throughput_ops: 25.0,
            error_rate_percent: 1.0,
            latency_p50_ms: 100,
            latency_p95_ms: 200,
            latency_p99_ms: 300,
            timestamp: Utc::now(),
        };

        assert!(monitor.record_metrics(metrics).await.is_ok());

        let retrieved_metrics = monitor.get_agent_metrics("test_agent", None).await;
        assert_eq!(retrieved_metrics.len(), 1);
    }

    #[tokio::test]
    async fn test_threshold_alerts() {
        let mut thresholds = PerformanceThresholds::default();
        thresholds.max_cpu_usage_percent = 50.0; // Lower threshold for testing
        
        let monitor = PerformanceMonitor::new(thresholds, 100);

        let high_cpu_metrics = PerformanceMetrics {
            agent_id: "test_agent".to_string(),
            cpu_usage_percent: 75.0, // Above threshold
            memory_usage_mb: 256,
            network_io_bps: 1024,
            disk_io_bps: 512,
            response_time_ms: 150,
            throughput_ops: 25.0,
            error_rate_percent: 1.0,
            latency_p50_ms: 100,
            latency_p95_ms: 200,
            latency_p99_ms: 300,
            timestamp: Utc::now(),
        };

        assert!(monitor.record_metrics(high_cpu_metrics).await.is_ok());

        let alerts = monitor.get_alerts(None).await;
        assert!(!alerts.is_empty());
    }

    #[tokio::test]
    async fn test_trend_analysis() {
        let thresholds = PerformanceThresholds::default();
        let monitor = PerformanceMonitor::new(thresholds, 100);

        // Record multiple metrics to enable trend analysis
        for i in 0..15 {
            let metrics = PerformanceMetrics {
                agent_id: "test_agent".to_string(),
                cpu_usage_percent: 30.0 + i as f64 * 2.0, // Increasing trend
                memory_usage_mb: 256,
                network_io_bps: 1024,
                disk_io_bps: 512,
                response_time_ms: 150,
                throughput_ops: 25.0,
                error_rate_percent: 1.0,
                latency_p50_ms: 100,
                latency_p95_ms: 200,
                latency_p99_ms: 300,
                timestamp: Utc::now(),
            };
            monitor.record_metrics(metrics).await.unwrap();
        }

        let trends = monitor.analyze_trends("test_agent").await;
        assert!(!trends.is_empty());
        
        // Should detect degrading CPU trend
        let cpu_trend = trends.iter().find(|t| t.metric_name == "cpu_usage");
        assert!(cpu_trend.is_some());
    }

    #[tokio::test]
    async fn test_performance_summary() {
        let thresholds = PerformanceThresholds::default();
        let monitor = PerformanceMonitor::new(thresholds, 100);

        // Record some metrics
        for i in 0..5 {
            let metrics = PerformanceMetrics {
                agent_id: "test_agent".to_string(),
                cpu_usage_percent: 40.0 + i as f64,
                memory_usage_mb: 256 + i * 64,
                network_io_bps: 1024,
                disk_io_bps: 512,
                response_time_ms: 150 + i * 10,
                throughput_ops: 25.0,
                error_rate_percent: 1.0,
                latency_p50_ms: 100,
                latency_p95_ms: 200,
                latency_p99_ms: 300,
                timestamp: Utc::now(),
            };
            monitor.record_metrics(metrics).await.unwrap();
        }

        let summary = monitor.get_performance_summary("test_agent").await;
        assert!(summary.is_some());
        
        let summary = summary.unwrap();
        assert_eq!(summary.agent_id, "test_agent");
        assert_eq!(summary.total_measurements, 5);
        assert!(summary.avg_cpu_usage_percent > 40.0);
    }
}
