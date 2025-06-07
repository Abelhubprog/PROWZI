//! Memory Consolidation Engine
//!
//! Manages the transfer of memories between layers and performs knowledge distillation.

use super::{MemoryId, MemoryContent, MemoryError, WorkingMemory, EpisodicMemory, SemanticMemory};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio::time::{sleep, Duration as TokioDuration};
use tracing::{debug, info, warn};

/// Consolidation request
#[derive(Debug, Clone)]
pub struct ConsolidationRequest {
    pub memory_id: MemoryId,
    pub content: MemoryContent,
    pub importance: f64,
    pub timestamp: DateTime<Utc>,
}

/// Consolidation engine for managing memory transfers
pub struct ConsolidationEngine {
    consolidation_interval: u64,
    pending_requests: Arc<RwLock<VecDeque<ConsolidationRequest>>>,
    consolidation_sender: mpsc::UnboundedSender<ConsolidationRequest>,
    consolidation_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<ConsolidationRequest>>>>,
}

impl ConsolidationEngine {
    /// Create new consolidation engine
    pub fn new(consolidation_interval_sec: u64) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        Self {
            consolidation_interval: consolidation_interval_sec,
            pending_requests: Arc::new(RwLock::new(VecDeque::new())),
            consolidation_sender: sender,
            consolidation_receiver: Arc::new(RwLock::new(Some(receiver))),
        }
    }

    /// Schedule a memory for consolidation
    pub async fn schedule_consolidation(
        &self,
        memory_id: MemoryId,
        content: MemoryContent,
        importance: f64,
    ) -> Result<(), MemoryError> {
        let request = ConsolidationRequest {
            memory_id,
            content,
            importance,
            timestamp: Utc::now(),
        };

        self.consolidation_sender.send(request)
            .map_err(|e| MemoryError::Consolidation(format!("Failed to schedule consolidation: {}", e)))?;

        Ok(())
    }

    /// Run the consolidation loop
    pub async fn run_consolidation_loop(
        &self,
        working_memory: Arc<RwLock<WorkingMemory>>,
        episodic_memory: Arc<RwLock<EpisodicMemory>>,
        semantic_memory: Arc<RwLock<SemanticMemory>>,
    ) {
        let mut receiver = {
            let mut receiver_guard = self.consolidation_receiver.write().await;
            receiver_guard.take()
        };

        if let Some(mut rx) = receiver {
            loop {
                // Process pending consolidation requests
                while let Ok(request) = rx.try_recv() {
                    if let Err(e) = self.process_consolidation(
                        request,
                        &working_memory,
                        &episodic_memory,
                        &semantic_memory,
                    ).await {
                        warn!("Consolidation failed: {}", e);
                    }
                }

                // Perform periodic maintenance
                if let Err(e) = self.perform_maintenance(
                    &working_memory,
                    &episodic_memory,
                    &semantic_memory,
                ).await {
                    warn!("Maintenance failed: {}", e);
                }

                // Sleep for the consolidation interval
                sleep(TokioDuration::from_secs(self.consolidation_interval)).await;
            }
        }
    }

    /// Process a single consolidation request
    async fn process_consolidation(
        &self,
        request: ConsolidationRequest,
        working_memory: &Arc<RwLock<WorkingMemory>>,
        episodic_memory: &Arc<RwLock<EpisodicMemory>>,
        semantic_memory: &Arc<RwLock<SemanticMemory>>,
    ) -> Result<(), MemoryError> {
        debug!("Processing consolidation for memory {}", request.memory_id.0);

        // Determine consolidation target based on importance and content type
        if request.importance > 0.8 {
            // High importance -> semantic memory
            let mut semantic = semantic_memory.write().await;
            semantic.store_pattern(&request.content).await?;
            info!("Consolidated high-importance memory {} to semantic layer", request.memory_id.0);
        } else if request.importance > 0.5 {
            // Medium importance -> episodic memory
            let mut episodic = episodic_memory.write().await;
            episodic.store_episode(&request.content).await?;
            info!("Consolidated medium-importance memory {} to episodic layer", request.memory_id.0);
        }

        // Remove from working memory after consolidation
        let mut working = working_memory.write().await;
        working.remove(&request.memory_id).await?;

        Ok(())
    }

    /// Perform periodic maintenance on all memory layers
    async fn perform_maintenance(
        &self,
        working_memory: &Arc<RwLock<WorkingMemory>>,
        episodic_memory: &Arc<RwLock<EpisodicMemory>>,
        semantic_memory: &Arc<RwLock<SemanticMemory>>,
    ) -> Result<(), MemoryError> {
        debug!("Performing memory maintenance");

        // Decay working memory activations
        {
            let mut working = working_memory.write().await;
            working.decay_activations().await?;
        }

        // Clean up old episodic memories
        {
            let mut episodic = episodic_memory.write().await;
            episodic.cleanup_old_episodes().await?;
        }

        // Update semantic patterns
        {
            let mut semantic = semantic_memory.write().await;
            semantic.update_pattern_strengths().await?;
        }

        debug!("Completed memory maintenance");
        Ok(())
    }
}

/// Memory consolidation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationStats {
    pub pending_requests: usize,
    pub total_consolidated: u64,
    pub working_to_episodic: u64,
    pub working_to_semantic: u64,
    pub episodic_to_semantic: u64,
    pub last_consolidation: Option<DateTime<Utc>>,
}

impl Default for ConsolidationStats {
    fn default() -> Self {
        Self {
            pending_requests: 0,
            total_consolidated: 0,
            working_to_episodic: 0,
            working_to_semantic: 0,
            episodic_to_semantic: 0,
            last_consolidation: None,
        }
    }
}