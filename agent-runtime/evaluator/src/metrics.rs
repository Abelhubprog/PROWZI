//! Agent Performance Metrics Module
//!
//! This module handles collection, processing, and reporting of agent performance metrics
//! for the Prowzi evaluation system. It tracks agent effectiveness, resource utilization,
//! and success rates across different mission types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive metrics for agent performance evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub agent_id: String,
    pub mission_id: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub performance: PerformanceMetrics,
    pub resources: ResourceMetrics,
    pub quality: QualityMetrics,
    pub errors: ErrorMetrics,
}

/// Performance-related metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub average_completion_time_ms: f64,
    pub throughput_per_hour: f64,
    pub success_rate: f64,
    pub efficiency_score: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub peak_memory_mb: u64,
    pub tokens_consumed: u64,
    pub api_calls_made: u64,
    pub bandwidth_used_mb: f64,
    pub gpu_utilization_percent: Option<f64>,
}

/// Quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityMetrics {
    pub accuracy_score: f64,
    pub confidence_score: f64,
    pub relevance_score: f64,
    pub consistency_score: f64,
    pub hallucination_rate: f64,
    pub user_satisfaction: Option<f64>,
}

/// Error tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub timeout_errors: u64,
    pub api_errors: u64,
    pub validation_errors: u64,
    pub critical_errors: u64,
    pub error_rate: f64,
    pub mean_time_to_recovery_ms: f64,
}

/// Aggregated metrics for multiple agents or missions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub total_agents: u64,
    pub active_agents: u64,
    pub total_missions: u64,
    pub completed_missions: u64,
    pub overall_performance: PerformanceMetrics,
    pub resource_summary: ResourceMetrics,
    pub quality_summary: QualityMetrics,
    pub error_summary: ErrorMetrics,
    pub trend_data: Vec<TrendPoint>,
}

/// Time-series data point for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub metric_type: MetricType,
}

/// Types of metrics for trend tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    SuccessRate,
    EfficiencyScore,
    ErrorRate,
    ResourceUtilization,
    QualityScore,
    Throughput,
}

/// Metrics collector for gathering and processing agent metrics
pub struct MetricsCollector {
    metrics: HashMap<String, AgentMetrics>,
    aggregates: AggregateMetrics,
    collection_interval: Duration,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(collection_interval: Duration) -> Self {
        Self {
            metrics: HashMap::new(),
            aggregates: AggregateMetrics::default(),
            collection_interval,
        }
    }

    /// Record metrics for an agent
    pub fn record_agent_metrics(&mut self, agent_id: String, metrics: AgentMetrics) {
        self.metrics.insert(agent_id, metrics);
        self.update_aggregates();
    }

    /// Get metrics for a specific agent
    pub fn get_agent_metrics(&self, agent_id: &str) -> Option<&AgentMetrics> {
        self.metrics.get(agent_id)
    }

    /// Get all current metrics
    pub fn get_all_metrics(&self) -> &HashMap<String, AgentMetrics> {
        &self.metrics
    }

    /// Get aggregated metrics
    pub fn get_aggregate_metrics(&self) -> &AggregateMetrics {
        &self.aggregates
    }

    /// Update performance metrics for an agent
    pub fn update_performance(&mut self, agent_id: &str, performance: PerformanceMetrics) {
        if let Some(metrics) = self.metrics.get_mut(agent_id) {
            metrics.performance = performance;
            self.update_aggregates();
        }
    }

    /// Update resource metrics for an agent
    pub fn update_resources(&mut self, agent_id: &str, resources: ResourceMetrics) {
        if let Some(metrics) = self.metrics.get_mut(agent_id) {
            metrics.resources = resources;
            self.update_aggregates();
        }
    }

    /// Update quality metrics for an agent
    pub fn update_quality(&mut self, agent_id: &str, quality: QualityMetrics) {
        if let Some(metrics) = self.metrics.get_mut(agent_id) {
            metrics.quality = quality;
            self.update_aggregates();
        }
    }

    /// Record an error for an agent
    pub fn record_error(&mut self, agent_id: &str, error_type: ErrorType) {
        if let Some(metrics) = self.metrics.get_mut(agent_id) {
            metrics.errors.total_errors += 1;
            
            match error_type {
                ErrorType::Timeout => metrics.errors.timeout_errors += 1,
                ErrorType::Api => metrics.errors.api_errors += 1,
                ErrorType::Validation => metrics.errors.validation_errors += 1,
                ErrorType::Critical => metrics.errors.critical_errors += 1,
            }
            
            self.calculate_error_rate(agent_id);
            self.update_aggregates();
        }
    }

    /// Calculate and update the success rate for an agent
    fn calculate_success_rate(&mut self, agent_id: &str) {
        if let Some(metrics) = self.metrics.get_mut(agent_id) {
            let total_tasks = metrics.performance.tasks_completed + metrics.performance.tasks_failed;
            if total_tasks > 0 {
                metrics.performance.success_rate = 
                    metrics.performance.tasks_completed as f64 / total_tasks as f64;
            }
        }
    }

    /// Calculate and update the error rate for an agent
    fn calculate_error_rate(&mut self, agent_id: &str) {
        if let Some(metrics) = self.metrics.get_mut(agent_id) {
            let total_tasks = metrics.performance.tasks_completed + metrics.performance.tasks_failed;
            if total_tasks > 0 {
                metrics.errors.error_rate = 
                    metrics.errors.total_errors as f64 / total_tasks as f64;
            }
        }
    }

    /// Update aggregate metrics based on individual agent metrics
    fn update_aggregates(&mut self) {
        let total_agents = self.metrics.len() as u64;
        let active_agents = self.metrics.values()
            .filter(|m| m.end_time.is_none())
            .count() as u64;

        // Calculate aggregate performance
        let mut total_tasks_completed = 0;
        let mut total_tasks_failed = 0;
        let mut sum_completion_time = 0.0;
        let mut sum_success_rate = 0.0;
        let mut sum_efficiency = 0.0;

        // Calculate aggregate resources
        let mut sum_cpu = 0.0;
        let mut sum_memory = 0;
        let mut sum_tokens = 0;
        let mut sum_api_calls = 0;

        // Calculate aggregate quality
        let mut sum_accuracy = 0.0;
        let mut sum_confidence = 0.0;
        let mut sum_relevance = 0.0;

        // Calculate aggregate errors
        let mut total_errors = 0;
        let mut sum_error_rate = 0.0;

        for metrics in self.metrics.values() {
            // Performance
            total_tasks_completed += metrics.performance.tasks_completed;
            total_tasks_failed += metrics.performance.tasks_failed;
            sum_completion_time += metrics.performance.average_completion_time_ms;
            sum_success_rate += metrics.performance.success_rate;
            sum_efficiency += metrics.performance.efficiency_score;

            // Resources
            sum_cpu += metrics.resources.cpu_usage_percent;
            sum_memory += metrics.resources.memory_usage_mb;
            sum_tokens += metrics.resources.tokens_consumed;
            sum_api_calls += metrics.resources.api_calls_made;

            // Quality
            sum_accuracy += metrics.quality.accuracy_score;
            sum_confidence += metrics.quality.confidence_score;
            sum_relevance += metrics.quality.relevance_score;

            // Errors
            total_errors += metrics.errors.total_errors;
            sum_error_rate += metrics.errors.error_rate;
        }

        let agent_count = total_agents as f64;
        if agent_count > 0.0 {
            self.aggregates.overall_performance = PerformanceMetrics {
                tasks_completed: total_tasks_completed,
                tasks_failed: total_tasks_failed,
                average_completion_time_ms: sum_completion_time / agent_count,
                success_rate: sum_success_rate / agent_count,
                efficiency_score: sum_efficiency / agent_count,
                throughput_per_hour: 0.0, // Would need time-based calculation
            };

            self.aggregates.resource_summary = ResourceMetrics {
                cpu_usage_percent: sum_cpu / agent_count,
                memory_usage_mb: sum_memory / total_agents,
                tokens_consumed: sum_tokens,
                api_calls_made: sum_api_calls,
                ..Default::default()
            };

            self.aggregates.quality_summary = QualityMetrics {
                accuracy_score: sum_accuracy / agent_count,
                confidence_score: sum_confidence / agent_count,
                relevance_score: sum_relevance / agent_count,
                ..Default::default()
            };

            self.aggregates.error_summary = ErrorMetrics {
                total_errors,
                error_rate: sum_error_rate / agent_count,
                ..Default::default()
            };
        }

        self.aggregates.total_agents = total_agents;
        self.aggregates.active_agents = active_agents;
    }

    /// Clear all metrics (useful for testing or reset scenarios)
    pub fn clear_all_metrics(&mut self) {
        self.metrics.clear();
        self.aggregates = AggregateMetrics::default();
    }

    /// Export metrics in JSON format
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.metrics)
    }
}

/// Types of errors that can be recorded
#[derive(Debug, Clone)]
pub enum ErrorType {
    Timeout,
    Api,
    Validation,
    Critical,
}

impl Default for AggregateMetrics {
    fn default() -> Self {
        Self {
            total_agents: 0,
            active_agents: 0,
            total_missions: 0,
            completed_missions: 0,
            overall_performance: PerformanceMetrics::default(),
            resource_summary: ResourceMetrics::default(),
            quality_summary: QualityMetrics::default(),
            error_summary: ErrorMetrics::default(),
            trend_data: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new(Duration::from_secs(60));
        assert_eq!(collector.metrics.len(), 0);
        assert_eq!(collector.aggregates.total_agents, 0);
    }

    #[test]
    fn test_record_agent_metrics() {
        let mut collector = MetricsCollector::new(Duration::from_secs(60));
        
        let metrics = AgentMetrics {
            agent_id: "test-agent".to_string(),
            mission_id: Some("test-mission".to_string()),
            start_time: Utc::now(),
            end_time: None,
            performance: PerformanceMetrics {
                tasks_completed: 10,
                tasks_failed: 2,
                success_rate: 0.83,
                ..Default::default()
            },
            resources: ResourceMetrics::default(),
            quality: QualityMetrics::default(),
            errors: ErrorMetrics::default(),
        };

        collector.record_agent_metrics("test-agent".to_string(), metrics);
        
        assert_eq!(collector.metrics.len(), 1);
        assert_eq!(collector.aggregates.total_agents, 1);
        assert!(collector.get_agent_metrics("test-agent").is_some());
    }

    #[test]
    fn test_error_recording() {
        let mut collector = MetricsCollector::new(Duration::from_secs(60));
        
        let metrics = AgentMetrics {
            agent_id: "test-agent".to_string(),
            mission_id: None,
            start_time: Utc::now(),
            end_time: None,
            performance: PerformanceMetrics::default(),
            resources: ResourceMetrics::default(),
            quality: QualityMetrics::default(),
            errors: ErrorMetrics::default(),
        };

        collector.record_agent_metrics("test-agent".to_string(), metrics);
        collector.record_error("test-agent", ErrorType::Api);
        
        let agent_metrics = collector.get_agent_metrics("test-agent").unwrap();
        assert_eq!(agent_metrics.errors.total_errors, 1);
        assert_eq!(agent_metrics.errors.api_errors, 1);
    }
}