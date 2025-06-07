//! Episodic Memory Implementation
//!
//! Medium-term memory for storing trading experiences and episodes with
//! vector similarity search and temporal retrieval capabilities.

use super::{
    MemoryId, MemoryContent, MemoryContext, MemoryError, RetrievedMemory, MemorySource, 
    MemoryFeedback, EpisodicMemoryStats, MarketContext, TradingAction, TradingOutcome
};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// A complete trading episode with context, actions, and outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub episode_id: MemoryId,
    pub agent_id: String,
    pub mission_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub duration_minutes: u32,
    pub context: EpisodeContext,
    pub actions: Vec<TradingAction>,
    pub outcomes: Vec<TradingOutcome>,
    pub emotions: EmotionalState,
    pub learned_patterns: Vec<LearnedPattern>,
    pub importance_score: f64,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
}

/// Extended context for episodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeContext {
    pub market_context: MarketContext,
    pub strategy_used: String,
    pub risk_tolerance: f64,
    pub market_conditions: super::MarketConditions,
    pub performance_expectations: PerformanceExpectations,
    pub external_factors: Vec<String>,
}

/// Emotional state associated with episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f64,        // -1.0 (negative) to 1.0 (positive)
    pub arousal: f64,        // 0.0 (calm) to 1.0 (excited)
    pub confidence: f64,     // 0.0 (unsure) to 1.0 (certain)
    pub stress_level: f64,   // 0.0 (relaxed) to 1.0 (stressed)
}

/// Patterns learned from episodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    pub pattern_id: String,
    pub pattern_type: super::PatternType,
    pub description: String,
    pub conditions: Vec<PatternCondition>,
    pub success_rate: f64,
    pub confidence: f64,
    pub sample_size: u32,
}

/// Conditions that define a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCondition {
    pub variable: String,
    pub operator: String,
    pub value: f64,
    pub tolerance: f64,
}

/// Performance expectations for episodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectations {
    pub expected_return: f64,
    pub max_drawdown: f64,
    pub time_horizon_minutes: u32,
    pub risk_reward_ratio: f64,
}

/// Episodic memory implementation with persistent storage simulation
pub struct EpisodicMemory {
    episodes: RwLock<HashMap<MemoryId, Episode>>,
    retention_days: i64,
    cleanup_count: AtomicU64,
    max_episodes: usize,
    // In a real implementation, this would use PostgreSQL or similar
    // For now, we simulate with in-memory storage with persistence hooks
}

impl EpisodicMemory {
    pub fn new(retention_days: i64) -> Self {
        Self {
            episodes: RwLock::new(HashMap::new()),
            retention_days,
            cleanup_count: AtomicU64::new(0),
            max_episodes: 10000, // Reasonable limit for in-memory simulation
        }
    }

    /// Store an episode from memory content
    pub async fn store_episode(&self, content: MemoryContent) -> Result<MemoryId, MemoryError> {
        let episode = self.convert_content_to_episode(content)?;
        let episode_id = episode.episode_id.clone();
        
        let mut episodes = self.episodes.write().await;
        
        // Check capacity and cleanup old episodes if needed
        if episodes.len() >= self.max_episodes {
            self.cleanup_old_episodes_internal(&mut episodes).await?;
        }
        
        episodes.insert(episode_id.clone(), episode);
        
        info!("Stored episode {} in episodic memory", episode_id.0);
        Ok(episode_id)
    }

    /// Retrieve similar episodes based on context
    pub async fn search_similar(&self, context: &MemoryContext, limit: usize) -> Result<Vec<RetrievedMemory>, MemoryError> {
        let episodes = self.episodes.read().await;
        let mut results = Vec::new();

        for episode in episodes.values() {
            let similarity = self.calculate_episode_similarity(episode, context);
            
            if similarity > 0.3 {
                let retrieved = RetrievedMemory {
                    memory_id: episode.episode_id.clone(),
                    content: self.episode_to_memory_content(episode),
                    relevance_score: similarity,
                    timestamp: episode.timestamp,
                    source: MemorySource::Episodic,
                };
                results.push(retrieved);
            }
        }

        // Sort by relevance and recency
        results.sort_by(|a, b| {
            let score_a = a.relevance_score * self.recency_weight(a.timestamp);
            let score_b = b.relevance_score * self.recency_weight(b.timestamp);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update access tracking for retrieved episodes
        drop(episodes);
        let mut episodes = self.episodes.write().await;
        for result in &results[..limit.min(results.len())] {
            if let Some(episode) = episodes.get_mut(&result.memory_id) {
                episode.access_count += 1;
                episode.last_accessed = Utc::now();
            }
        }

        Ok(results.into_iter().take(limit).collect())
    }

    /// Retrieve successful episodes from a time period
    pub async fn retrieve_successful_episodes(&self, time_period: Duration) -> Result<Vec<Episode>, MemoryError> {
        let episodes = self.episodes.read().await;
        let cutoff_time = Utc::now() - time_period;
        
        let successful_episodes: Vec<Episode> = episodes
            .values()
            .filter(|episode| {
                episode.timestamp >= cutoff_time && 
                episode.importance_score > 0.6 &&
                episode.emotions.valence > 0.3
            })
            .cloned()
            .collect();

        Ok(successful_episodes)
    }

    /// Retrieve episodes by importance score threshold
    pub async fn retrieve_by_significance(&self, min_score: f64, limit: usize) -> Result<Vec<Episode>, MemoryError> {
        let episodes = self.episodes.read().await;
        
        let mut significant_episodes: Vec<Episode> = episodes
            .values()
            .filter(|episode| episode.importance_score >= min_score)
            .cloned()
            .collect();

        significant_episodes.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(significant_episodes.into_iter().take(limit).collect())
    }

    /// Update memory based on feedback
    pub async fn update_memory(&self, memory_id: &MemoryId, feedback: &MemoryFeedback) -> Result<(), MemoryError> {
        let mut episodes = self.episodes.write().await;
        
        if let Some(episode) = episodes.get_mut(memory_id) {
            match feedback.feedback_type {
                super::FeedbackType::SuccessRate => {
                    // Update learned patterns based on success rate feedback
                    for pattern in &mut episode.learned_patterns {
                        pattern.success_rate = (pattern.success_rate + feedback.value) / 2.0;
                    }
                }
                super::FeedbackType::PerformanceScore => {
                    episode.importance_score = (episode.importance_score + feedback.value) / 2.0;
                }
                _ => {}
            }
            
            episode.last_accessed = Utc::now();
            debug!("Updated episode {} with feedback: {:?}", memory_id.0, feedback);
        }

        Ok(())
    }

    /// Cleanup old episodes beyond retention period
    pub async fn cleanup_old_episodes(&self, retention_period: Duration) -> Result<usize, MemoryError> {
        let mut episodes = self.episodes.write().await;
        self.cleanup_old_episodes_internal(&mut episodes).await
    }

    /// Extract patterns from recent episodes
    pub async fn extract_patterns(&self, min_episodes: usize) -> Result<Vec<LearnedPattern>, MemoryError> {
        let episodes = self.episodes.read().await;
        
        if episodes.len() < min_episodes {
            return Ok(Vec::new());
        }

        // Group episodes by similar market conditions
        let mut pattern_groups: HashMap<String, Vec<&Episode>> = HashMap::new();
        
        for episode in episodes.values() {
            let market_key = self.create_market_key(&episode.context.market_conditions);
            pattern_groups.entry(market_key).or_insert_with(Vec::new).push(episode);
        }

        let mut extracted_patterns = Vec::new();
        
        for (market_key, group_episodes) in pattern_groups {
            if group_episodes.len() >= 3 { // Minimum sample size for pattern
                if let Some(pattern) = self.analyze_episode_group(&group_episodes, &market_key) {
                    extracted_patterns.push(pattern);
                }
            }
        }

        Ok(extracted_patterns)
    }

    /// Get memory statistics
    pub async fn get_statistics(&self) -> EpisodicMemoryStats {
        let episodes = self.episodes.read().await;
        let total_items = episodes.len();
        
        let average_importance = if total_items > 0 {
            episodes.values().map(|e| e.importance_score).sum::<f64>() / total_items as f64
        } else {
            0.0
        };

        EpisodicMemoryStats {
            total_items,
            average_importance,
            retention_days: self.retention_days,
            cleanup_count: self.cleanup_count.load(Ordering::Relaxed),
        }
    }

    // Helper methods

    fn convert_content_to_episode(&self, content: MemoryContent) -> Result<Episode, MemoryError> {
        let episode_id = MemoryId::new();
        let now = Utc::now();

        match content.content_type {
            super::MemoryContentType::TradingExperience { 
                market_context, 
                actions_taken, 
                outcomes, 
                performance_score 
            } => {
                let emotions = self.derive_emotions_from_performance(performance_score);
                let learned_patterns = self.extract_patterns_from_experience(&actions_taken, &outcomes);
                
                let episode_context = EpisodeContext {
                    market_context,
                    strategy_used: content.metadata.get("strategy")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    risk_tolerance: content.metadata.get("risk_tolerance")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5),
                    market_conditions: content.context.market_conditions
                        .unwrap_or_else(|| super::MarketConditions {
                            overall_sentiment: 0.0,
                            volatility_index: 0.5,
                            liquidity_score: 0.5,
                            market_phase: super::MarketPhase::Unknown,
                        }),
                    performance_expectations: PerformanceExpectations {
                        expected_return: content.metadata.get("expected_return")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.05),
                        max_drawdown: content.metadata.get("max_drawdown")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.1),
                        time_horizon_minutes: content.metadata.get("time_horizon_minutes")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(60) as u32,
                        risk_reward_ratio: content.metadata.get("risk_reward_ratio")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(2.0),
                    },
                    external_factors: content.context.tags,
                };

                Ok(Episode {
                    episode_id,
                    agent_id: content.context.agent_id,
                    mission_id: content.context.mission_id,
                    timestamp: content.context.timestamp,
                    duration_minutes: 60, // Default duration, could be derived
                    context: episode_context,
                    actions: actions_taken,
                    outcomes,
                    emotions,
                    learned_patterns,
                    importance_score: performance_score,
                    access_count: 0,
                    last_accessed: now,
                })
            }
            _ => {
                // Convert other content types to generic episodes
                let emotions = EmotionalState {
                    valence: 0.0,
                    arousal: 0.5,
                    confidence: 0.5,
                    stress_level: 0.3,
                };

                let episode_context = EpisodeContext {
                    market_context: MarketContext {
                        symbol: "UNKNOWN".to_string(),
                        price: 0.0,
                        volume_24h: 0.0,
                        volatility: 0.5,
                        market_cap: None,
                        sentiment_score: None,
                    },
                    strategy_used: "generic".to_string(),
                    risk_tolerance: 0.5,
                    market_conditions: content.context.market_conditions
                        .unwrap_or_else(|| super::MarketConditions {
                            overall_sentiment: 0.0,
                            volatility_index: 0.5,
                            liquidity_score: 0.5,
                            market_phase: super::MarketPhase::Unknown,
                        }),
                    performance_expectations: PerformanceExpectations {
                        expected_return: 0.0,
                        max_drawdown: 0.1,
                        time_horizon_minutes: 30,
                        risk_reward_ratio: 1.0,
                    },
                    external_factors: content.context.tags,
                };

                Ok(Episode {
                    episode_id,
                    agent_id: content.context.agent_id,
                    mission_id: content.context.mission_id,
                    timestamp: content.context.timestamp,
                    duration_minutes: 30,
                    context: episode_context,
                    actions: Vec::new(),
                    outcomes: Vec::new(),
                    emotions,
                    learned_patterns: Vec::new(),
                    importance_score: 0.5,
                    access_count: 0,
                    last_accessed: now,
                })
            }
        }
    }

    fn episode_to_memory_content(&self, episode: &Episode) -> MemoryContent {
        let context = MemoryContext {
            agent_id: episode.agent_id.clone(),
            mission_id: episode.mission_id.clone(),
            timestamp: episode.timestamp,
            market_conditions: Some(episode.context.market_conditions.clone()),
            performance_state: None,
            tags: episode.context.external_factors.clone(),
        };

        let mut metadata = HashMap::new();
        metadata.insert("strategy".to_string(), serde_json::Value::String(episode.context.strategy_used.clone()));
        metadata.insert("importance_score".to_string(), serde_json::Value::Number(episode.importance_score.into()));
        metadata.insert("access_count".to_string(), serde_json::Value::Number(episode.access_count.into()));

        MemoryContent {
            content_type: super::MemoryContentType::TradingExperience {
                market_context: episode.context.market_context.clone(),
                actions_taken: episode.actions.clone(),
                outcomes: episode.outcomes.clone(),
                performance_score: episode.importance_score,
            },
            data: serde_json::to_value(episode).unwrap_or_default(),
            context,
            embeddings: None,
            metadata,
        }
    }

    fn calculate_episode_similarity(&self, episode: &Episode, context: &MemoryContext) -> f64 {
        let mut similarity = 0.0;
        let mut weight_sum = 0.0;

        // Agent similarity
        if episode.agent_id == context.agent_id {
            similarity += 0.2;
        }
        weight_sum += 0.2;

        // Mission similarity
        if episode.mission_id == context.mission_id && context.mission_id.is_some() {
            similarity += 0.1;
        }
        weight_sum += 0.1;

        // Market conditions similarity
        if let Some(ctx_market) = &context.market_conditions {
            let market_sim = self.calculate_market_conditions_similarity(&episode.context.market_conditions, ctx_market);
            similarity += market_sim * 0.4;
        }
        weight_sum += 0.4;

        // Tag overlap
        let tag_overlap = self.calculate_tag_overlap(&episode.context.external_factors, &context.tags);
        similarity += tag_overlap * 0.3;
        weight_sum += 0.3;

        if weight_sum > 0.0 {
            similarity / weight_sum
        } else {
            0.0
        }
    }

    fn calculate_market_conditions_similarity(&self, mc1: &super::MarketConditions, mc2: &super::MarketConditions) -> f64 {
        let sentiment_sim = 1.0 - (mc1.overall_sentiment - mc2.overall_sentiment).abs();
        let volatility_sim = 1.0 - (mc1.volatility_index - mc2.volatility_index).abs();
        let liquidity_sim = 1.0 - (mc1.liquidity_score - mc2.liquidity_score).abs();
        let phase_sim = if mc1.market_phase == mc2.market_phase { 1.0 } else { 0.0 };
        
        (sentiment_sim + volatility_sim + liquidity_sim + phase_sim) / 4.0
    }

    fn calculate_tag_overlap(&self, tags1: &[String], tags2: &[String]) -> f64 {
        if tags1.is_empty() && tags2.is_empty() {
            return 1.0;
        }
        if tags1.is_empty() || tags2.is_empty() {
            return 0.0;
        }

        let set1: std::collections::HashSet<_> = tags1.iter().collect();
        let set2: std::collections::HashSet<_> = tags2.iter().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    fn recency_weight(&self, timestamp: DateTime<Utc>) -> f64 {
        let age_hours = (Utc::now() - timestamp).num_hours() as f64;
        (-age_hours / (24.0 * 7.0)).exp() // Decay over a week
    }

    fn derive_emotions_from_performance(&self, performance_score: f64) -> EmotionalState {
        EmotionalState {
            valence: (performance_score - 0.5) * 2.0, // Map 0-1 to -1 to 1
            arousal: performance_score.abs() * 2.0 - 1.0, // Higher for extreme performance
            confidence: performance_score,
            stress_level: (1.0 - performance_score).max(0.0),
        }
    }

    fn extract_patterns_from_experience(&self, actions: &[TradingAction], outcomes: &[TradingOutcome]) -> Vec<LearnedPattern> {
        let mut patterns = Vec::new();

        // Simple pattern: action timing and outcomes
        if !actions.is_empty() && !outcomes.is_empty() {
            let success_rate = outcomes.iter().filter(|o| o.success).count() as f64 / outcomes.len() as f64;
            
            patterns.push(LearnedPattern {
                pattern_id: format!("timing_pattern_{}", Uuid::new_v4()),
                pattern_type: super::PatternType::Custom("timing".to_string()),
                description: "Action timing pattern based on market conditions".to_string(),
                conditions: vec![
                    PatternCondition {
                        variable: "action_count".to_string(),
                        operator: "equals".to_string(),
                        value: actions.len() as f64,
                        tolerance: 1.0,
                    }
                ],
                success_rate,
                confidence: if outcomes.len() > 5 { 0.7 } else { 0.4 },
                sample_size: outcomes.len() as u32,
            });
        }

        patterns
    }

    async fn cleanup_old_episodes_internal(&self, episodes: &mut HashMap<MemoryId, Episode>) -> Result<usize, MemoryError> {
        let cutoff_time = Utc::now() - Duration::days(self.retention_days);
        let mut to_remove = Vec::new();

        for (id, episode) in episodes.iter() {
            if episode.timestamp < cutoff_time && episode.importance_score < 0.3 {
                to_remove.push(id.clone());
            }
        }

        let removed_count = to_remove.len();
        for id in to_remove {
            episodes.remove(&id);
        }

        if removed_count > 0 {
            self.cleanup_count.fetch_add(removed_count as u64, Ordering::Relaxed);
            info!("Cleaned up {} old episodes from episodic memory", removed_count);
        }

        Ok(removed_count)
    }

    fn create_market_key(&self, conditions: &super::MarketConditions) -> String {
        format!(
            "{}_{}_{}_{:?}",
            (conditions.overall_sentiment * 10.0).round() as i32,
            (conditions.volatility_index * 10.0).round() as i32,
            (conditions.liquidity_score * 10.0).round() as i32,
            conditions.market_phase
        )
    }

    fn analyze_episode_group(&self, episodes: &[&Episode], market_key: &str) -> Option<LearnedPattern> {
        if episodes.len() < 3 {
            return None;
        }

        let avg_performance: f64 = episodes.iter().map(|e| e.importance_score).sum::<f64>() / episodes.len() as f64;
        let success_count = episodes.iter().filter(|e| e.importance_score > 0.6).count();
        let success_rate = success_count as f64 / episodes.len() as f64;

        if success_rate > 0.6 {
            Some(LearnedPattern {
                pattern_id: format!("market_pattern_{}", market_key),
                pattern_type: super::PatternType::Custom("market_conditions".to_string()),
                description: format!("Successful trading pattern under market conditions: {}", market_key),
                conditions: vec![
                    PatternCondition {
                        variable: "market_key".to_string(),
                        operator: "equals".to_string(),
                        value: 0.0, // Would encode market_key as number in real implementation
                        tolerance: 0.0,
                    }
                ],
                success_rate,
                confidence: if episodes.len() > 10 { 0.8 } else { 0.6 },
                sample_size: episodes.len() as u32,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryContentType, MarketContext};

    #[tokio::test]
    async fn test_episodic_memory_store_retrieve() {
        let episodic_memory = EpisodicMemory::new(30);
        
        let context = MemoryContext {
            agent_id: "test-agent".to_string(),
            mission_id: Some("test-mission".to_string()),
            timestamp: Utc::now(),
            market_conditions: Some(super::MarketConditions {
                overall_sentiment: 0.7,
                volatility_index: 0.3,
                liquidity_score: 0.8,
                market_phase: super::MarketPhase::Bull,
            }),
            performance_state: None,
            tags: vec!["profitable".to_string(), "low-risk".to_string()],
        };

        let content = MemoryContent {
            content_type: MemoryContentType::TradingExperience {
                market_context: MarketContext {
                    symbol: "SOL/USDC".to_string(),
                    price: 100.0,
                    volume_24h: 1000000.0,
                    volatility: 0.3,
                    market_cap: Some(50000000.0),
                    sentiment_score: Some(0.7),
                },
                actions_taken: vec![
                    TradingAction {
                        action_type: "buy".to_string(),
                        parameters: std::collections::HashMap::new(),
                        timestamp: Utc::now(),
                        confidence: 0.8,
                    }
                ],
                outcomes: vec![
                    TradingOutcome {
                        outcome_type: "profit".to_string(),
                        value: 500.0,
                        timestamp: Utc::now(),
                        success: true,
                    }
                ],
                performance_score: 0.85,
            },
            data: serde_json::json!({}),
            context: context.clone(),
            embeddings: None,
            metadata: std::collections::HashMap::new(),
        };

        // Store episode
        let episode_id = episodic_memory.store_episode(content).await.unwrap();

        // Search for similar episodes
        let results = episodic_memory.search_similar(&context, 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].memory_id, episode_id);

        // Retrieve successful episodes
        let successful = episodic_memory.retrieve_successful_episodes(Duration::hours(1)).await.unwrap();
        assert!(!successful.is_empty());
    }

    #[tokio::test]
    async fn test_episode_pattern_extraction() {
        let episodic_memory = EpisodicMemory::new(30);
        
        // Store multiple similar episodes
        for i in 0..5 {
            let context = MemoryContext {
                agent_id: "test-agent".to_string(),
                mission_id: Some("test-mission".to_string()),
                timestamp: Utc::now() - Duration::minutes(i * 10),
                market_conditions: Some(super::MarketConditions {
                    overall_sentiment: 0.7,
                    volatility_index: 0.3,
                    liquidity_score: 0.8,
                    market_phase: super::MarketPhase::Bull,
                }),
                performance_state: None,
                tags: vec!["profitable".to_string()],
            };

            let content = MemoryContent {
                content_type: MemoryContentType::TradingExperience {
                    market_context: MarketContext {
                        symbol: "SOL/USDC".to_string(),
                        price: 100.0 + i as f64,
                        volume_24h: 1000000.0,
                        volatility: 0.3,
                        market_cap: Some(50000000.0),
                        sentiment_score: Some(0.7),
                    },
                    actions_taken: vec![],
                    outcomes: vec![],
                    performance_score: 0.8,
                },
                data: serde_json::json!({}),
                context,
                embeddings: None,
                metadata: std::collections::HashMap::new(),
            };

            episodic_memory.store_episode(content).await.unwrap();
        }

        // Extract patterns
        let patterns = episodic_memory.extract_patterns(3).await.unwrap();
        assert!(!patterns.is_empty());
    }
}