//! Agent Performance Banding Module
//!
//! This module implements a performance banding system for categorizing agents
//! based on their performance metrics. It provides tiered evaluation to identify
//! top performers, average performers, and underperformers for optimization.

use crate::metrics::{AgentMetrics, PerformanceMetrics, QualityMetrics, ResourceMetrics};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance band classifications for agents
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerformanceBand {
    Elite,      // Top 10% performers
    Superior,   // Top 10-25% performers  
    Good,       // Top 25-50% performers
    Average,    // Top 50-75% performers
    Poor,       // Bottom 25% performers
    Critical,   // Severely underperforming agents
}

impl PerformanceBand {
    /// Get the numeric score range for this band (0-100)
    pub fn score_range(&self) -> (f64, f64) {
        match self {
            PerformanceBand::Elite => (90.0, 100.0),
            PerformanceBand::Superior => (75.0, 90.0),
            PerformanceBand::Good => (60.0, 75.0),
            PerformanceBand::Average => (40.0, 60.0),
            PerformanceBand::Poor => (20.0, 40.0),
            PerformanceBand::Critical => (0.0, 20.0),
        }
    }

    /// Get the band description
    pub fn description(&self) -> &'static str {
        match self {
            PerformanceBand::Elite => "Elite performer with exceptional results",
            PerformanceBand::Superior => "Superior performer exceeding expectations",
            PerformanceBand::Good => "Good performer meeting high standards",
            PerformanceBand::Average => "Average performer meeting basic requirements",
            PerformanceBand::Poor => "Poor performer requiring improvement",
            PerformanceBand::Critical => "Critical performer requiring immediate attention",
        }
    }

    /// Get recommended actions for this band
    pub fn recommended_actions(&self) -> Vec<&'static str> {
        match self {
            PerformanceBand::Elite => vec![
                "Consider for leadership roles",
                "Use as training examples",
                "Maintain current configuration"
            ],
            PerformanceBand::Superior => vec![
                "Increase task complexity",
                "Consider for specialized missions",
                "Monitor for elite promotion"
            ],
            PerformanceBand::Good => vec![
                "Optimize resource allocation",
                "Fine-tune parameters",
                "Gradual task complexity increase"
            ],
            PerformanceBand::Average => vec![
                "Review configuration",
                "Provide additional training data",
                "Monitor closely"
            ],
            PerformanceBand::Poor => vec![
                "Reduce task complexity",
                "Increase supervision",
                "Consider retraining"
            ],
            PerformanceBand::Critical => vec![
                "Immediate intervention required",
                "Full configuration review",
                "Consider deactivation"
            ],
        }
    }
}

/// Comprehensive evaluation result for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvaluation {
    pub agent_id: String,
    pub mission_id: Option<String>,
    pub overall_band: PerformanceBand,
    pub overall_score: f64,
    pub component_scores: ComponentScores,
    pub evaluation_time: DateTime<Utc>,
    pub recommendations: Vec<String>,
    pub trend: PerformanceTrend,
}

/// Detailed component scores for different aspects of performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentScores {
    pub efficiency_score: f64,
    pub quality_score: f64,
    pub reliability_score: f64,
    pub resource_score: f64,
    pub consistency_score: f64,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Declining,
    Volatile,
    Insufficient, // Not enough data
}

/// Configuration for banding thresholds and weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandingConfig {
    pub efficiency_weight: f64,
    pub quality_weight: f64,
    pub reliability_weight: f64,
    pub resource_weight: f64,
    pub consistency_weight: f64,
    pub minimum_evaluations: usize,
    pub trend_window_hours: u64,
}

impl Default for BandingConfig {
    fn default() -> Self {
        Self {
            efficiency_weight: 0.3,
            quality_weight: 0.25,
            reliability_weight: 0.2,
            resource_weight: 0.15,
            consistency_weight: 0.1,
            minimum_evaluations: 5,
            trend_window_hours: 24,
        }
    }
}

/// Agent performance evaluator for banding
pub struct PerformanceEvaluator {
    config: BandingConfig,
    historical_evaluations: HashMap<String, Vec<AgentEvaluation>>,
}

impl PerformanceEvaluator {
    /// Create a new performance evaluator
    pub fn new(config: BandingConfig) -> Self {
        Self {
            config,
            historical_evaluations: HashMap::new(),
        }
    }

    /// Evaluate an agent's performance and assign a band
    pub fn evaluate_agent(&mut self, metrics: &AgentMetrics) -> AgentEvaluation {
        let component_scores = self.calculate_component_scores(metrics);
        let overall_score = self.calculate_overall_score(&component_scores);
        let overall_band = self.determine_band(overall_score);
        let trend = self.calculate_trend(&metrics.agent_id);
        let recommendations = self.generate_recommendations(&overall_band, &trend, &component_scores);

        let evaluation = AgentEvaluation {
            agent_id: metrics.agent_id.clone(),
            mission_id: metrics.mission_id.clone(),
            overall_band,
            overall_score,
            component_scores,
            evaluation_time: Utc::now(),
            recommendations,
            trend,
        };

        // Store for historical analysis
        self.historical_evaluations
            .entry(metrics.agent_id.clone())
            .or_insert_with(Vec::new)
            .push(evaluation.clone());

        evaluation
    }

    /// Calculate component scores from agent metrics
    fn calculate_component_scores(&self, metrics: &AgentMetrics) -> ComponentScores {
        ComponentScores {
            efficiency_score: self.calculate_efficiency_score(&metrics.performance),
            quality_score: self.calculate_quality_score(&metrics.quality),
            reliability_score: self.calculate_reliability_score(metrics),
            resource_score: self.calculate_resource_score(&metrics.resources),
            consistency_score: self.calculate_consistency_score(&metrics.agent_id),
        }
    }

    /// Calculate efficiency score based on performance metrics
    fn calculate_efficiency_score(&self, performance: &PerformanceMetrics) -> f64 {
        let success_rate_score = performance.success_rate * 40.0;
        let efficiency_score = performance.efficiency_score * 30.0;
        let throughput_score = (performance.throughput_per_hour / 100.0).min(1.0) * 30.0;
        
        success_rate_score + efficiency_score + throughput_score
    }

    /// Calculate quality score based on quality metrics
    fn calculate_quality_score(&self, quality: &QualityMetrics) -> f64 {
        let accuracy_score = quality.accuracy_score * 30.0;
        let confidence_score = quality.confidence_score * 20.0;
        let relevance_score = quality.relevance_score * 25.0;
        let consistency_score = quality.consistency_score * 15.0;
        let hallucination_penalty = (1.0 - quality.hallucination_rate) * 10.0;
        
        accuracy_score + confidence_score + relevance_score + consistency_score + hallucination_penalty
    }

    /// Calculate reliability score based on error metrics and uptime
    fn calculate_reliability_score(&self, metrics: &AgentMetrics) -> f64 {
        let error_penalty = (1.0 - metrics.errors.error_rate) * 50.0;
        let uptime_score = if metrics.end_time.is_none() { 30.0 } else { 20.0 };
        let recovery_score = if metrics.errors.mean_time_to_recovery_ms > 0.0 {
            (1.0 - (metrics.errors.mean_time_to_recovery_ms / 60000.0).min(1.0)) * 20.0
        } else {
            20.0
        };
        
        error_penalty + uptime_score + recovery_score
    }

    /// Calculate resource efficiency score
    fn calculate_resource_score(&self, resources: &ResourceMetrics) -> f64 {
        // Lower resource usage is better (more efficient)
        let cpu_efficiency = (1.0 - (resources.cpu_usage_percent / 100.0)) * 30.0;
        let memory_efficiency = if resources.memory_usage_mb > 0 {
            (1.0 - (resources.memory_usage_mb as f64 / 2048.0).min(1.0)) * 25.0
        } else {
            25.0
        };
        let token_efficiency = if resources.tokens_consumed > 0 {
            (1.0 - (resources.tokens_consumed as f64 / 10000.0).min(1.0)) * 25.0
        } else {
            25.0
        };
        let bandwidth_efficiency = (1.0 - (resources.bandwidth_used_mb / 100.0).min(1.0)) * 20.0;
        
        cpu_efficiency + memory_efficiency + token_efficiency + bandwidth_efficiency
    }

    /// Calculate consistency score based on historical performance
    fn calculate_consistency_score(&self, agent_id: &str) -> f64 {
        if let Some(evaluations) = self.historical_evaluations.get(agent_id) {
            if evaluations.len() < 2 {
                return 50.0; // Default score for insufficient data
            }

            let scores: Vec<f64> = evaluations.iter()
                .take(10) // Last 10 evaluations
                .map(|e| e.overall_score)
                .collect();
            
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance = scores.iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();
            
            // Lower standard deviation means higher consistency
            (1.0 - (std_dev / 50.0).min(1.0)) * 100.0
        } else {
            50.0 // Default score for new agents
        }
    }

    /// Calculate overall weighted score
    fn calculate_overall_score(&self, component_scores: &ComponentScores) -> f64 {
        component_scores.efficiency_score * self.config.efficiency_weight +
        component_scores.quality_score * self.config.quality_weight +
        component_scores.reliability_score * self.config.reliability_weight +
        component_scores.resource_score * self.config.resource_weight +
        component_scores.consistency_score * self.config.consistency_weight
    }

    /// Determine performance band based on overall score
    fn determine_band(&self, overall_score: f64) -> PerformanceBand {
        match overall_score {
            score if score >= 90.0 => PerformanceBand::Elite,
            score if score >= 75.0 => PerformanceBand::Superior,
            score if score >= 60.0 => PerformanceBand::Good,
            score if score >= 40.0 => PerformanceBand::Average,
            score if score >= 20.0 => PerformanceBand::Poor,
            _ => PerformanceBand::Critical,
        }
    }

    /// Calculate performance trend
    fn calculate_trend(&self, agent_id: &str) -> PerformanceTrend {
        if let Some(evaluations) = self.historical_evaluations.get(agent_id) {
            if evaluations.len() < self.config.minimum_evaluations {
                return PerformanceTrend::Insufficient;
            }

            let recent_scores: Vec<f64> = evaluations.iter()
                .rev()
                .take(self.config.minimum_evaluations)
                .map(|e| e.overall_score)
                .collect();

            if recent_scores.len() < self.config.minimum_evaluations {
                return PerformanceTrend::Insufficient;
            }

            // Simple trend analysis
            let first_half = &recent_scores[0..recent_scores.len()/2];
            let second_half = &recent_scores[recent_scores.len()/2..];
            
            let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
            let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
            
            let variance = recent_scores.iter()
                .map(|score| (*score - first_avg).powi(2))
                .sum::<f64>() / recent_scores.len() as f64;
            
            if variance > 100.0 {
                PerformanceTrend::Volatile
            } else if second_avg > first_avg + 5.0 {
                PerformanceTrend::Improving
            } else if second_avg < first_avg - 5.0 {
                PerformanceTrend::Declining
            } else {
                PerformanceTrend::Stable
            }
        } else {
            PerformanceTrend::Insufficient
        }
    }

    /// Generate recommendations based on band, trend, and component scores
    fn generate_recommendations(
        &self,
        band: &PerformanceBand,
        trend: &PerformanceTrend,
        component_scores: &ComponentScores,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Add band-specific recommendations
        recommendations.extend(
            band.recommended_actions()
                .iter()
                .map(|s| s.to_string())
        );

        // Add trend-specific recommendations
        match trend {
            PerformanceTrend::Improving => {
                recommendations.push("Performance is improving - maintain current trajectory".to_string());
            },
            PerformanceTrend::Declining => {
                recommendations.push("Performance is declining - investigate root causes".to_string());
            },
            PerformanceTrend::Volatile => {
                recommendations.push("Performance is inconsistent - review configuration stability".to_string());
            },
            PerformanceTrend::Stable => {
                recommendations.push("Performance is stable - consider optimization opportunities".to_string());
            },
            PerformanceTrend::Insufficient => {
                recommendations.push("Insufficient data for trend analysis - continue monitoring".to_string());
            },
        }

        // Add component-specific recommendations
        if component_scores.quality_score < 50.0 {
            recommendations.push("Quality score is low - review training data and model parameters".to_string());
        }
        if component_scores.resource_score < 50.0 {
            recommendations.push("Resource efficiency is poor - optimize resource usage".to_string());
        }
        if component_scores.reliability_score < 50.0 {
            recommendations.push("Reliability is concerning - address error handling and stability".to_string());
        }

        recommendations
    }

    /// Get band distribution across all evaluated agents
    pub fn get_band_distribution(&self) -> HashMap<PerformanceBand, usize> {
        let mut distribution = HashMap::new();
        
        for evaluations in self.historical_evaluations.values() {
            if let Some(latest) = evaluations.last() {
                *distribution.entry(latest.overall_band.clone()).or_insert(0) += 1;
            }
        }
        
        distribution
    }

    /// Get historical evaluations for an agent
    pub fn get_agent_history(&self, agent_id: &str) -> Option<&Vec<AgentEvaluation>> {
        self.historical_evaluations.get(agent_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::*;

    fn create_test_metrics() -> AgentMetrics {
        AgentMetrics {
            agent_id: "test-agent".to_string(),
            mission_id: Some("test-mission".to_string()),
            start_time: Utc::now(),
            end_time: None,
            performance: PerformanceMetrics {
                tasks_completed: 100,
                tasks_failed: 5,
                success_rate: 0.95,
                efficiency_score: 0.85,
                throughput_per_hour: 50.0,
                average_completion_time_ms: 1000.0,
            },
            resources: ResourceMetrics {
                cpu_usage_percent: 40.0,
                memory_usage_mb: 512,
                tokens_consumed: 1000,
                bandwidth_used_mb: 10.0,
                ..Default::default()
            },
            quality: QualityMetrics {
                accuracy_score: 0.9,
                confidence_score: 0.85,
                relevance_score: 0.88,
                consistency_score: 0.82,
                hallucination_rate: 0.05,
                ..Default::default()
            },
            errors: ErrorMetrics {
                total_errors: 3,
                error_rate: 0.03,
                mean_time_to_recovery_ms: 5000.0,
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_performance_band_ordering() {
        assert!(PerformanceBand::Elite > PerformanceBand::Superior);
        assert!(PerformanceBand::Superior > PerformanceBand::Good);
        assert!(PerformanceBand::Good > PerformanceBand::Average);
        assert!(PerformanceBand::Average > PerformanceBand::Poor);
        assert!(PerformanceBand::Poor > PerformanceBand::Critical);
    }

    #[test]
    fn test_agent_evaluation() {
        let mut evaluator = PerformanceEvaluator::new(BandingConfig::default());
        let metrics = create_test_metrics();
        
        let evaluation = evaluator.evaluate_agent(&metrics);
        
        assert_eq!(evaluation.agent_id, "test-agent");
        assert!(evaluation.overall_score > 0.0);
        assert!(!evaluation.recommendations.is_empty());
    }

    #[test]
    fn test_band_determination() {
        let evaluator = PerformanceEvaluator::new(BandingConfig::default());
        
        assert_eq!(evaluator.determine_band(95.0), PerformanceBand::Elite);
        assert_eq!(evaluator.determine_band(80.0), PerformanceBand::Superior);
        assert_eq!(evaluator.determine_band(65.0), PerformanceBand::Good);
        assert_eq!(evaluator.determine_band(45.0), PerformanceBand::Average);
        assert_eq!(evaluator.determine_band(25.0), PerformanceBand::Poor);
        assert_eq!(evaluator.determine_band(10.0), PerformanceBand::Critical);
    }

    #[test]
    fn test_component_score_calculation() {
        let evaluator = PerformanceEvaluator::new(BandingConfig::default());
        let metrics = create_test_metrics();
        
        let scores = evaluator.calculate_component_scores(&metrics);
        
        assert!(scores.efficiency_score > 0.0);
        assert!(scores.quality_score > 0.0);
        assert!(scores.reliability_score > 0.0);
        assert!(scores.resource_score > 0.0);
    }
}