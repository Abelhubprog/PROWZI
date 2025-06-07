//! Working Memory Implementation
//!
//! Fast, short-term memory for active processing with automatic decay and eviction.

use super::{MemoryId, MemoryContent, MemoryContext, MemoryError, RetrievedMemory, MemorySource, MemoryFeedback, WorkingMemoryStats};
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::time::Instant;
use tracing::{debug, warn};

/// Working memory item with activation tracking
#[derive(Debug, Clone)]
pub struct WorkingMemoryItem {
    pub memory_id: MemoryId,
    pub content: MemoryContent,
    pub activation_level: f64,
    pub last_accessed: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub access_count: u64,
    pub associations: Vec<MemoryId>,
}

impl WorkingMemoryItem {
    pub fn new(memory_id: MemoryId, content: MemoryContent) -> Self {
        let now = Utc::now();
        Self {
            memory_id,
            content,
            activation_level: 1.0,
            last_accessed: now,
            created_at: now,
            access_count: 0,
            associations: Vec::new(),
        }
    }

    pub fn access(&mut self) {
        self.activation_level = (self.activation_level + 0.3).min(1.0);
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }

    pub fn decay(&mut self, decay_rate: f64, time_delta_sec: f64) {
        let decay_factor = (-decay_rate * time_delta_sec / 3600.0).exp(); // Per hour decay
        self.activation_level *= decay_factor;
        self.activation_level = self.activation_level.max(0.0);
    }

    pub fn apply_feedback(&mut self, feedback: &MemoryFeedback) {
        match feedback.feedback_type {
            super::FeedbackType::RelevanceRating => {
                self.activation_level = (self.activation_level + feedback.value * 0.2).min(1.0);
            }
            super::FeedbackType::UsefulnessScore => {
                self.activation_level = (self.activation_level + feedback.value * 0.1).min(1.0);
            }
            _ => {}
        }
    }

    pub fn add_association(&mut self, memory_id: MemoryId) {
        if !self.associations.contains(&memory_id) {
            self.associations.push(memory_id);
        }
    }
}

/// Working memory implementation with LRU-style eviction
pub struct WorkingMemory {
    items: DashMap<MemoryId, WorkingMemoryItem>,
    capacity: usize,
    decay_rate: f64,
    activation_threshold: f64,
    eviction_count: AtomicU64,
    last_decay_time: tokio::sync::Mutex<Instant>,
}

impl WorkingMemory {
    pub fn new(capacity: usize, decay_rate: f64) -> Self {
        Self {
            items: DashMap::new(),
            capacity,
            decay_rate,
            activation_threshold: 0.1,
            eviction_count: AtomicU64::new(0),
            last_decay_time: tokio::sync::Mutex::new(Instant::now()),
        }
    }

    /// Store a new memory item
    pub async fn store(&self, memory_id: MemoryId, content: MemoryContent) -> Result<(), MemoryError> {
        // Apply decay before storing
        self.apply_decay().await;

        // Check capacity and evict if necessary
        if self.items.len() >= self.capacity {
            self.evict_least_activated().await?;
        }

        let item = WorkingMemoryItem::new(memory_id.clone(), content);
        self.items.insert(memory_id, item);

        debug!("Stored memory item, current count: {}", self.items.len());
        Ok(())
    }

    /// Retrieve a memory item by ID
    pub async fn get(&self, memory_id: &MemoryId) -> Option<MemoryContent> {
        if let Some(mut item) = self.items.get_mut(memory_id) {
            item.access();
            Some(item.content.clone())
        } else {
            None
        }
    }

    /// Get mutable reference to memory item
    pub fn get_mut(&self, memory_id: &MemoryId) -> Option<dashmap::mapref::one::RefMut<MemoryId, WorkingMemoryItem>> {
        self.items.get_mut(memory_id)
    }

    /// Search for memories by context similarity
    pub async fn search_by_context(&self, context: &MemoryContext, limit: usize) -> Result<Vec<RetrievedMemory>, MemoryError> {
        let mut results = Vec::new();

        for item_ref in self.items.iter() {
            let item = item_ref.value();
            
            // Skip items below activation threshold
            if item.activation_level < self.activation_threshold {
                continue;
            }

            let relevance = self.calculate_context_similarity(&item.content.context, context);
            
            if relevance > 0.3 {
                results.push(RetrievedMemory {
                    memory_id: item.memory_id.clone(),
                    content: item.content.clone(),
                    relevance_score: relevance * item.activation_level,
                    timestamp: item.created_at,
                    source: MemorySource::Working,
                });
            }
        }

        // Sort by relevance score
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

        // Access the items we're returning to boost their activation
        for result in &results[..limit.min(results.len())] {
            if let Some(mut item) = self.items.get_mut(&result.memory_id) {
                item.access();
            }
        }

        Ok(results.into_iter().take(limit).collect())
    }

    /// Get all items that should be considered for consolidation
    pub async fn get_consolidation_candidates(&self, min_importance: f64) -> Result<Vec<WorkingMemoryItem>, MemoryError> {
        let candidates: Vec<WorkingMemoryItem> = self.items
            .iter()
            .filter_map(|item_ref| {
                let item = item_ref.value();
                if item.activation_level >= min_importance {
                    Some(item.clone())
                } else {
                    None
                }
            })
            .collect();

        Ok(candidates)
    }

    /// Remove a memory item
    pub async fn remove(&self, memory_id: &MemoryId) -> Option<WorkingMemoryItem> {
        self.items.remove(memory_id).map(|(_, item)| item)
    }

    /// Apply decay to all items
    async fn apply_decay(&self) {
        let mut last_decay = self.last_decay_time.lock().await;
        let now = Instant::now();
        let time_delta_sec = now.duration_since(*last_decay).as_secs_f64();
        
        if time_delta_sec > 60.0 { // Only decay every minute
            let items_to_remove: Vec<MemoryId> = self.items
                .iter_mut()
                .filter_map(|mut item_ref| {
                    let item = item_ref.value_mut();
                    item.decay(self.decay_rate, time_delta_sec);
                    
                    if item.activation_level < self.activation_threshold {
                        Some(item.memory_id.clone())
                    } else {
                        None
                    }
                })
                .collect();

            // Remove items that have decayed below threshold
            for memory_id in items_to_remove {
                self.items.remove(&memory_id);
            }

            *last_decay = now;
        }
    }

    /// Evict least activated item
    async fn evict_least_activated(&self) -> Result<(), MemoryError> {
        let mut min_activation = f64::MAX;
        let mut evict_id = None;

        for item_ref in self.items.iter() {
            let item = item_ref.value();
            let score = item.activation_level - (Utc::now() - item.last_accessed).num_minutes() as f64 * 0.001;
            
            if score < min_activation {
                min_activation = score;
                evict_id = Some(item.memory_id.clone());
            }
        }

        if let Some(id) = evict_id {
            self.items.remove(&id);
            self.eviction_count.fetch_add(1, Ordering::Relaxed);
            debug!("Evicted memory item {} (activation: {:.3})", id.0, min_activation);
        }

        Ok(())
    }

    /// Calculate similarity between two contexts
    fn calculate_context_similarity(&self, ctx1: &MemoryContext, ctx2: &MemoryContext) -> f64 {
        let mut similarity = 0.0;
        let mut factors = 0.0;

        // Agent similarity
        if ctx1.agent_id == ctx2.agent_id {
            similarity += 0.3;
        }
        factors += 0.3;

        // Mission similarity
        if ctx1.mission_id == ctx2.mission_id && ctx1.mission_id.is_some() {
            similarity += 0.2;
        }
        factors += 0.2;

        // Tag overlap
        let tag_overlap = self.calculate_tag_overlap(&ctx1.tags, &ctx2.tags);
        similarity += tag_overlap * 0.3;
        factors += 0.3;

        // Market conditions similarity
        if let (Some(mc1), Some(mc2)) = (&ctx1.market_conditions, &ctx2.market_conditions) {
            let market_sim = self.calculate_market_similarity(mc1, mc2);
            similarity += market_sim * 0.2;
        }
        factors += 0.2;

        if factors > 0.0 {
            similarity / factors
        } else {
            0.0
        }
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

    fn calculate_market_similarity(&self, mc1: &super::MarketConditions, mc2: &super::MarketConditions) -> f64 {
        let sentiment_sim = 1.0 - (mc1.overall_sentiment - mc2.overall_sentiment).abs();
        let volatility_sim = 1.0 - (mc1.volatility_index - mc2.volatility_index).abs();
        let liquidity_sim = 1.0 - (mc1.liquidity_score - mc2.liquidity_score).abs();
        
        let phase_sim = if mc1.market_phase == mc2.market_phase { 1.0 } else { 0.0 };
        
        (sentiment_sim + volatility_sim + liquidity_sim + phase_sim) / 4.0
    }

    /// Get memory statistics
    pub async fn get_statistics(&self) -> WorkingMemoryStats {
        let total_items = self.items.len();
        let average_activation = if total_items > 0 {
            self.items.iter()
                .map(|item| item.value().activation_level)
                .sum::<f64>() / total_items as f64
        } else {
            0.0
        };

        WorkingMemoryStats {
            total_items,
            capacity: self.capacity,
            average_activation,
            eviction_count: self.eviction_count.load(Ordering::Relaxed),
        }
    }

    /// Clean up forgotten memories (below threshold)
    pub async fn cleanup_forgotten_memories(&self) -> Result<usize, MemoryError> {
        let items_to_remove: Vec<MemoryId> = self.items
            .iter()
            .filter_map(|item_ref| {
                let item = item_ref.value();
                if item.activation_level < self.activation_threshold {
                    Some(item.memory_id.clone())
                } else {
                    None
                }
            })
            .collect();

        let removed_count = items_to_remove.len();
        for memory_id in items_to_remove {
            self.items.remove(&memory_id);
        }

        debug!("Cleaned up {} forgotten memories", removed_count);
        Ok(removed_count)
    }

    /// Create associations between memories
    pub async fn create_associations(&self, memories: &[(MemoryId, f64)]) -> Result<(), MemoryError> {
        // Sort by relevance and create bidirectional associations
        for (i, (id1, score1)) in memories.iter().enumerate() {
            for (id2, score2) in memories.iter().skip(i + 1) {
                let association_strength = (score1 + score2) / 2.0;
                
                if association_strength > 0.6 {
                    if let Some(mut item1) = self.items.get_mut(id1) {
                        item1.add_association(id2.clone());
                    }
                    if let Some(mut item2) = self.items.get_mut(id2) {
                        item2.add_association(id1.clone());
                    }
                }
            }
        }

        Ok(())
    }

    /// Get associated memories for a given memory ID
    pub async fn get_associations(&self, memory_id: &MemoryId) -> Vec<MemoryContent> {
        if let Some(item) = self.items.get(memory_id) {
            let mut associated_content = Vec::new();
            
            for assoc_id in &item.associations {
                if let Some(mut assoc_item) = self.items.get_mut(assoc_id) {
                    assoc_item.access(); // Boost activation for accessed associations
                    associated_content.push(assoc_item.content.clone());
                }
            }
            
            associated_content
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryContentType, InsightType};

    #[tokio::test]
    async fn test_working_memory_store_retrieve() {
        let working_memory = WorkingMemory::new(100, 0.1);
        
        let memory_id = MemoryId::new();
        let context = MemoryContext {
            agent_id: "test-agent".to_string(),
            mission_id: Some("test-mission".to_string()),
            timestamp: Utc::now(),
            market_conditions: None,
            performance_state: None,
            tags: vec!["test".to_string()],
        };
        
        let content = MemoryContent {
            content_type: MemoryContentType::Insight {
                insight_type: InsightType::MarketTrend,
                content: "Test insight".to_string(),
                confidence: 0.8,
                source_events: vec!["event1".to_string()],
            },
            data: serde_json::json!({"test": "data"}),
            context: context.clone(),
            embeddings: None,
            metadata: std::collections::HashMap::new(),
        };

        // Store memory
        working_memory.store(memory_id.clone(), content.clone()).await.unwrap();

        // Retrieve memory
        let retrieved = working_memory.get(&memory_id).await;
        assert!(retrieved.is_some());

        // Search by context
        let results = working_memory.search_by_context(&context, 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].memory_id, memory_id);
    }

    #[tokio::test]
    async fn test_working_memory_capacity_and_eviction() {
        let working_memory = WorkingMemory::new(3, 0.1);
        
        // Fill to capacity
        for i in 0..3 {
            let memory_id = MemoryId::new();
            let context = MemoryContext {
                agent_id: format!("agent-{}", i),
                mission_id: None,
                timestamp: Utc::now(),
                market_conditions: None,
                performance_state: None,
                tags: vec![],
            };
            
            let content = MemoryContent {
                content_type: MemoryContentType::Insight {
                    insight_type: InsightType::MarketTrend,
                    content: format!("Test insight {}", i),
                    confidence: 0.5,
                    source_events: vec![],
                },
                data: serde_json::json!({"index": i}),
                context,
                embeddings: None,
                metadata: std::collections::HashMap::new(),
            };

            working_memory.store(memory_id, content).await.unwrap();
        }

        assert_eq!(working_memory.items.len(), 3);

        // Add one more - should trigger eviction
        let memory_id = MemoryId::new();
        let context = MemoryContext {
            agent_id: "agent-extra".to_string(),
            mission_id: None,
            timestamp: Utc::now(),
            market_conditions: None,
            performance_state: None,
            tags: vec![],
        };
        
        let content = MemoryContent {
            content_type: MemoryContentType::Insight {
                insight_type: InsightType::MarketTrend,
                content: "Extra insight".to_string(),
                confidence: 0.5,
                source_events: vec![],
            },
            data: serde_json::json!({"extra": true}),
            context,
            embeddings: None,
            metadata: std::collections::HashMap::new(),
        };

        working_memory.store(memory_id, content).await.unwrap();
        
        // Should still have 3 items (one evicted)
        assert_eq!(working_memory.items.len(), 3);
        
        let stats = working_memory.get_statistics().await;
        assert_eq!(stats.eviction_count, 1);
    }

    #[tokio::test]
    async fn test_memory_decay() {
        let working_memory = WorkingMemory::new(100, 1.0); // High decay rate
        
        let memory_id = MemoryId::new();
        let context = MemoryContext {
            agent_id: "test-agent".to_string(),
            mission_id: None,
            timestamp: Utc::now(),
            market_conditions: None,
            performance_state: None,
            tags: vec![],
        };
        
        let content = MemoryContent {
            content_type: MemoryContentType::Insight {
                insight_type: InsightType::MarketTrend,
                content: "Decaying insight".to_string(),
                confidence: 0.5,
                source_events: vec![],
            },
            data: serde_json::json!({}),
            context,
            embeddings: None,
            metadata: std::collections::HashMap::new(),
        };

        working_memory.store(memory_id.clone(), content).await.unwrap();
        
        // Manually trigger decay
        if let Some(mut item) = working_memory.items.get_mut(&memory_id) {
            item.decay(1.0, 7200.0); // 2 hours of decay
            assert!(item.activation_level < 1.0);
        }
    }
}