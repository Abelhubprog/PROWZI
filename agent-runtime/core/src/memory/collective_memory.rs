//! Collective Memory Implementation
//!
//! Swarm intelligence coordination with shared knowledge pools, consensus mechanisms,
//! and distributed decision-making for autonomous agent collaboration.

use super::{
    MemoryId, MemoryContent, MemoryContext, MemoryError, RetrievedMemory, MemorySource,
    MemoryFeedback, CollectiveMemoryStats, SwarmConsensus, SharedKnowledge, AgentContribution
};
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info, warn, error};

/// Configuration for collective memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveConfig {
    pub max_agents: usize,
    pub consensus_threshold: f32,
    pub contribution_decay_days: i64,
    pub knowledge_sharing_radius: usize,
    pub min_agreement_ratio: f32,
    pub reputation_weight: f32,
    pub novelty_bonus: f32,
    pub coordination_timeout_ms: u64,
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self {
            max_agents: 1000,
            consensus_threshold: 0.75,
            contribution_decay_days: 30,
            knowledge_sharing_radius: 5,
            min_agreement_ratio: 0.6,
            reputation_weight: 0.3,
            novelty_bonus: 0.2,
            coordination_timeout_ms: 5000,
        }
    }
}

/// Agent reputation and contribution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub agent_id: String,
    pub reputation_score: f32,
    pub total_contributions: u64,
    pub successful_predictions: u64,
    pub failed_predictions: u64,
    pub specialization_domains: Vec<String>,
    pub collaboration_score: f32,
    pub last_active: DateTime<Utc>,
    pub contribution_history: VecDeque<AgentContribution>,
}

/// Consensus mechanism for collective decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRound {
    pub id: String,
    pub topic: String,
    pub proposer_id: String,
    pub proposed_content: MemoryContent,
    pub votes: HashMap<String, ConsensusVote>,
    pub status: ConsensusStatus,
    pub created_at: DateTime<Utc>,
    pub deadline: DateTime<Utc>,
    pub final_result: Option<SwarmConsensus>,
}

/// Individual vote in consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    pub agent_id: String,
    pub vote_type: VoteType,
    pub confidence: f32,
    pub reasoning: String,
    pub evidence: Vec<MemoryId>,
    pub timestamp: DateTime<Utc>,
}

/// Vote types for consensus
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VoteType {
    Agree,
    Disagree,
    Abstain,
    Counter(MemoryContent),
}

/// Status of consensus round
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusStatus {
    Active,
    Completed,
    Failed,
    Timeout,
}

/// Knowledge propagation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePropagation {
    pub knowledge_id: MemoryId,
    pub source_agent: String,
    pub propagation_path: Vec<String>,
    pub propagation_speed: Duration,
    pub adoption_rate: f32,
    pub resistance_points: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Collective memory system for swarm intelligence
pub struct CollectiveMemory {
    config: CollectiveConfig,
    agent_profiles: DashMap<String, AgentProfile>,
    shared_knowledge: DashMap<MemoryId, SharedKnowledge>,
    consensus_rounds: DashMap<String, ConsensusRound>,
    knowledge_propagation: DashMap<MemoryId, KnowledgePropagation>,
    collaboration_graph: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    event_broadcaster: broadcast::Sender<CollectiveEvent>,
    stats: CollectiveMemoryStats,
    next_consensus_id: AtomicU64,
}

/// Events broadcast to agents in the collective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectiveEvent {
    NewKnowledge { knowledge_id: MemoryId, source_agent: String },
    ConsensusStarted { consensus_id: String, topic: String },
    ConsensusCompleted { consensus_id: String, result: SwarmConsensus },
    AgentJoined { agent_id: String },
    AgentLeft { agent_id: String },
    CriticalAlert { message: String, severity: AlertSeverity },
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl CollectiveMemory {
    /// Create new collective memory instance
    pub fn new(config: CollectiveConfig) -> Self {
        let (event_broadcaster, _) = broadcast::channel(10000);
        
        Self {
            config,
            agent_profiles: DashMap::new(),
            shared_knowledge: DashMap::new(),
            consensus_rounds: DashMap::new(),
            knowledge_propagation: DashMap::new(),
            collaboration_graph: Arc::new(RwLock::new(HashMap::new())),
            event_broadcaster,
            stats: CollectiveMemoryStats::default(),
            next_consensus_id: AtomicU64::new(1),
        }
    }

    /// Register a new agent in the collective
    pub async fn register_agent(&self, agent_id: String, specializations: Vec<String>) -> Result<(), MemoryError> {
        let profile = AgentProfile {
            agent_id: agent_id.clone(),
            reputation_score: 0.5, // Start with neutral reputation
            total_contributions: 0,
            successful_predictions: 0,
            failed_predictions: 0,
            specialization_domains: specializations,
            collaboration_score: 0.5,
            last_active: Utc::now(),
            contribution_history: VecDeque::new(),
        };

        self.agent_profiles.insert(agent_id.clone(), profile);

        // Initialize collaboration graph
        let mut graph = self.collaboration_graph.write().await;
        graph.insert(agent_id.clone(), HashSet::new());
        drop(graph);

        // Broadcast agent join event
        let _ = self.event_broadcaster.send(CollectiveEvent::AgentJoined { agent_id: agent_id.clone() });

        self.stats.active_agents.fetch_add(1, Ordering::SeqCst);
        info!("Registered agent {} in collective memory", agent_id);
        Ok(())
    }

    /// Contribute knowledge to the collective
    pub async fn contribute_knowledge(
        &self,
        agent_id: &str,
        content: MemoryContent,
        context: MemoryContext,
        confidence: f32,
    ) -> Result<MemoryId, MemoryError> {
        // Validate agent exists
        let mut profile = self.agent_profiles.get_mut(agent_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Agent: {}", agent_id)))?;

        let knowledge_id = MemoryId(format!("collective_{}", Utc::now().timestamp_nanos()));

        // Create shared knowledge entry
        let shared_knowledge = SharedKnowledge {
            id: knowledge_id.clone(),
            content: content.clone(),
            context: context.clone(),
            contributing_agent: agent_id.to_string(),
            confidence_score: confidence,
            validation_votes: HashMap::new(),
            adoption_count: 0,
            challenge_count: 0,
            created_at: Utc::now(),
            last_validated: Utc::now(),
            consensus_status: None,
        };

        self.shared_knowledge.insert(knowledge_id.clone(), shared_knowledge);

        // Update agent profile
        profile.total_contributions += 1;
        profile.last_active = Utc::now();
        
        let contribution = AgentContribution {
            knowledge_id: knowledge_id.clone(),
            contribution_type: super::ContributionType::Knowledge,
            value: confidence,
            timestamp: Utc::now(),
            validation_score: None,
        };
        
        profile.contribution_history.push_back(contribution);
        if profile.contribution_history.len() > 100 {
            profile.contribution_history.pop_front();
        }

        // Initiate knowledge propagation
        self.start_knowledge_propagation(&knowledge_id, agent_id).await?;

        // Broadcast new knowledge event
        let _ = self.event_broadcaster.send(CollectiveEvent::NewKnowledge {
            knowledge_id: knowledge_id.clone(),
            source_agent: agent_id.to_string(),
        });

        self.stats.total_contributions.fetch_add(1, Ordering::SeqCst);
        debug!("Agent {} contributed knowledge {}", agent_id, knowledge_id.0);
        Ok(knowledge_id)
    }

    /// Start consensus round for important decisions
    pub async fn start_consensus(
        &self,
        proposer_id: &str,
        topic: String,
        proposed_content: MemoryContent,
        timeout_duration: Duration,
    ) -> Result<String, MemoryError> {
        // Validate proposer exists
        self.agent_profiles.get(proposer_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Agent: {}", proposer_id)))?;

        let consensus_id = format!("consensus_{}", self.next_consensus_id.fetch_add(1, Ordering::SeqCst));
        let deadline = Utc::now() + timeout_duration;

        let consensus_round = ConsensusRound {
            id: consensus_id.clone(),
            topic: topic.clone(),
            proposer_id: proposer_id.to_string(),
            proposed_content,
            votes: HashMap::new(),
            status: ConsensusStatus::Active,
            created_at: Utc::now(),
            deadline,
            final_result: None,
        };

        self.consensus_rounds.insert(consensus_id.clone(), consensus_round);

        // Broadcast consensus start event
        let _ = self.event_broadcaster.send(CollectiveEvent::ConsensusStarted {
            consensus_id: consensus_id.clone(),
            topic,
        });

        self.stats.consensus_rounds.fetch_add(1, Ordering::SeqCst);
        info!("Started consensus round {} by agent {}", consensus_id, proposer_id);
        Ok(consensus_id)
    }

    /// Submit vote for consensus round
    pub async fn vote_consensus(
        &self,
        consensus_id: &str,
        agent_id: &str,
        vote_type: VoteType,
        confidence: f32,
        reasoning: String,
        evidence: Vec<MemoryId>,
    ) -> Result<(), MemoryError> {
        // Validate agent exists
        self.agent_profiles.get(agent_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Agent: {}", agent_id)))?;

        // Get consensus round
        let mut consensus = self.consensus_rounds.get_mut(consensus_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Consensus: {}", consensus_id)))?;

        // Check if still active and before deadline
        if consensus.status != ConsensusStatus::Active {
            return Err(MemoryError::InvalidOperation("Consensus round not active".to_string()));
        }

        if Utc::now() > consensus.deadline {
            return Err(MemoryError::InvalidOperation("Consensus round expired".to_string()));
        }

        // Submit vote
        let vote = ConsensusVote {
            agent_id: agent_id.to_string(),
            vote_type,
            confidence,
            reasoning,
            evidence,
            timestamp: Utc::now(),
        };

        consensus.votes.insert(agent_id.to_string(), vote);

        // Check if consensus reached
        if self.check_consensus_completion(&mut consensus).await? {
            self.finalize_consensus(&mut consensus).await?;
        }

        debug!("Agent {} voted on consensus {}", agent_id, consensus_id);
        Ok(())
    }

    /// Retrieve knowledge from collective memory
    pub async fn retrieve_collective_knowledge(
        &self,
        query_context: &MemoryContext,
        requester_id: &str,
        max_results: usize,
    ) -> Result<Vec<RetrievedMemory>, MemoryError> {
        // Validate requester
        self.agent_profiles.get(requester_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Agent: {}", requester_id)))?;

        let mut results = Vec::new();

        // Find relevant knowledge based on context and agent specializations
        for entry in self.shared_knowledge.iter() {
            let knowledge = entry.value();
            
            // Calculate relevance score
            let relevance = self.calculate_knowledge_relevance(knowledge, query_context, requester_id).await?;
            
            if relevance > 0.3 { // Minimum relevance threshold
                let retrieved = RetrievedMemory {
                    id: knowledge.id.clone(),
                    content: knowledge.content.clone(),
                    context: knowledge.context.clone(),
                    relevance_score: relevance,
                    retrieval_time: Utc::now(),
                    access_count: knowledge.adoption_count as u64,
                    last_access: knowledge.last_validated,
                    source: MemorySource::Collective,
                };
                results.push(retrieved);
            }
        }

        // Sort by relevance and limit results
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);

        // Update collaboration graph
        self.update_collaboration_connections(requester_id, &results).await?;

        self.stats.knowledge_retrievals.fetch_add(1, Ordering::SeqCst);
        debug!("Retrieved {} collective knowledge items for agent {}", results.len(), requester_id);
        Ok(results)
    }

    /// Validate shared knowledge through peer review
    pub async fn validate_knowledge(
        &self,
        knowledge_id: &MemoryId,
        validator_id: &str,
        is_valid: bool,
        feedback: String,
    ) -> Result<(), MemoryError> {
        // Validate validator exists
        let validator_profile = self.agent_profiles.get(validator_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Agent: {}", validator_id)))?;

        // Get shared knowledge
        let mut knowledge = self.shared_knowledge.get_mut(knowledge_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Knowledge: {}", knowledge_id.0)))?;

        // Add validation vote (weighted by reputation)
        let validation_weight = validator_profile.reputation_score;
        knowledge.validation_votes.insert(validator_id.to_string(), (is_valid, validation_weight, feedback));

        // Update adoption or challenge count
        if is_valid {
            knowledge.adoption_count += 1;
        } else {
            knowledge.challenge_count += 1;
        }

        knowledge.last_validated = Utc::now();

        // Update contributor's reputation based on validation
        if let Some(mut contributor) = self.agent_profiles.get_mut(&knowledge.contributing_agent) {
            if is_valid {
                contributor.successful_predictions += 1;
                contributor.reputation_score = (contributor.reputation_score * 0.95 + 0.05).min(1.0);
            } else {
                contributor.failed_predictions += 1;
                contributor.reputation_score = (contributor.reputation_score * 0.95).max(0.0);
            }
        }

        debug!("Agent {} validated knowledge {} as {}", validator_id, knowledge_id.0, is_valid);
        Ok(())
    }

    /// Get agent's collaboration network
    pub async fn get_collaboration_network(&self, agent_id: &str) -> Result<Vec<String>, MemoryError> {
        let graph = self.collaboration_graph.read().await;
        Ok(graph.get(agent_id)
            .map(|connections| connections.iter().cloned().collect())
            .unwrap_or_default())
    }

    /// Update agent feedback and reputation
    pub async fn update_agent_feedback(
        &self,
        agent_id: &str,
        feedback: &MemoryFeedback,
    ) -> Result<(), MemoryError> {
        let mut profile = self.agent_profiles.get_mut(agent_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Agent: {}", agent_id)))?;

        match feedback {
            MemoryFeedback::Helpful { .. } => {
                profile.reputation_score = (profile.reputation_score * 0.9 + 0.1).min(1.0);
                profile.collaboration_score = (profile.collaboration_score * 0.9 + 0.1).min(1.0);
            }
            MemoryFeedback::Irrelevant { .. } => {
                profile.reputation_score = (profile.reputation_score * 0.95).max(0.0);
            }
            MemoryFeedback::Outdated { .. } => {
                // No immediate reputation impact for outdated information
            }
        }

        profile.last_active = Utc::now();
        debug!("Updated feedback for agent {}", agent_id);
        Ok(())
    }

    /// Perform collective memory maintenance
    pub async fn maintenance(&self) -> Result<(), MemoryError> {
        // Expire old consensus rounds
        self.expire_consensus_rounds().await?;

        // Decay old contributions
        self.decay_contributions().await?;

        // Update reputation scores
        self.update_reputation_scores().await?;

        // Prune inactive agents
        self.prune_inactive_agents().await?;

        // Update statistics
        self.update_statistics().await?;

        info!("Completed collective memory maintenance");
        Ok(())
    }

    /// Get collective memory statistics
    pub fn get_stats(&self) -> CollectiveMemoryStats {
        let mut stats = self.stats.clone();
        stats.active_agents = AtomicUsize::new(self.agent_profiles.len());
        stats.shared_knowledge_count = self.shared_knowledge.len();
        stats.active_consensus_rounds = self.consensus_rounds.iter()
            .filter(|round| round.status == ConsensusStatus::Active)
            .count();
        stats
    }

    /// Subscribe to collective events
    pub fn subscribe_events(&self) -> broadcast::Receiver<CollectiveEvent> {
        self.event_broadcaster.subscribe()
    }

    // Private helper methods

    async fn start_knowledge_propagation(&self, knowledge_id: &MemoryId, source_agent: &str) -> Result<(), MemoryError> {
        let propagation = KnowledgePropagation {
            knowledge_id: knowledge_id.clone(),
            source_agent: source_agent.to_string(),
            propagation_path: vec![source_agent.to_string()],
            propagation_speed: Duration::zero(),
            adoption_rate: 0.0,
            resistance_points: Vec::new(),
            created_at: Utc::now(),
        };

        self.knowledge_propagation.insert(knowledge_id.clone(), propagation);
        debug!("Started knowledge propagation for {}", knowledge_id.0);
        Ok(())
    }

    async fn check_consensus_completion(&self, consensus: &mut ConsensusRound) -> Result<bool, MemoryError> {
        let total_votes = consensus.votes.len();
        let total_agents = self.agent_profiles.len();

        // Check if enough agents have voted or deadline passed
        let participation_threshold = (total_agents as f32 * self.config.consensus_threshold) as usize;
        let deadline_passed = Utc::now() > consensus.deadline;

        if total_votes >= participation_threshold || deadline_passed {
            return Ok(true);
        }

        Ok(false)
    }

    async fn finalize_consensus(&self, consensus: &mut ConsensusRound) -> Result<(), MemoryError> {
        let mut agree_weight = 0.0;
        let mut disagree_weight = 0.0;
        let mut total_weight = 0.0;

        // Calculate weighted votes
        for vote in consensus.votes.values() {
            if let Some(profile) = self.agent_profiles.get(&vote.agent_id) {
                let weight = profile.reputation_score * vote.confidence;
                total_weight += weight;

                match vote.vote_type {
                    VoteType::Agree => agree_weight += weight,
                    VoteType::Disagree => disagree_weight += weight,
                    VoteType::Abstain => {},
                    VoteType::Counter(_) => disagree_weight += weight,
                }
            }
        }

        // Determine consensus result
        let agreement_ratio = if total_weight > 0.0 { agree_weight / total_weight } else { 0.0 };
        
        let result = if agreement_ratio >= self.config.min_agreement_ratio {
            SwarmConsensus {
                decision: super::ConsensusDecision::Accept,
                confidence: agreement_ratio,
                supporting_agents: consensus.votes.iter()
                    .filter_map(|(id, vote)| if matches!(vote.vote_type, VoteType::Agree) { Some(id.clone()) } else { None })
                    .collect(),
                dissenting_agents: consensus.votes.iter()
                    .filter_map(|(id, vote)| if matches!(vote.vote_type, VoteType::Disagree) { Some(id.clone()) } else { None })
                    .collect(),
                final_content: Some(consensus.proposed_content.clone()),
                timestamp: Utc::now(),
            }
        } else {
            SwarmConsensus {
                decision: super::ConsensusDecision::Reject,
                confidence: 1.0 - agreement_ratio,
                supporting_agents: Vec::new(),
                dissenting_agents: consensus.votes.keys().cloned().collect(),
                final_content: None,
                timestamp: Utc::now(),
            }
        };

        consensus.final_result = Some(result.clone());
        consensus.status = ConsensusStatus::Completed;

        // Broadcast completion event
        let _ = self.event_broadcaster.send(CollectiveEvent::ConsensusCompleted {
            consensus_id: consensus.id.clone(),
            result,
        });

        info!("Finalized consensus {} with agreement ratio {:.2}", consensus.id, agreement_ratio);
        Ok(())
    }

    async fn calculate_knowledge_relevance(
        &self,
        knowledge: &SharedKnowledge,
        query_context: &MemoryContext,
        requester_id: &str,
    ) -> Result<f32, MemoryError> {
        let mut relevance = 0.0;

        // Base relevance from confidence score
        relevance += knowledge.confidence_score * 0.3;

        // Context similarity (simplified)
        if knowledge.context.mission_context == query_context.mission_context {
            relevance += 0.3;
        }

        // Temporal relevance
        let age_hours = Utc::now().signed_duration_since(knowledge.created_at).num_hours();
        let temporal_factor = (1.0 / (1.0 + age_hours as f32 / 24.0)).min(1.0);
        relevance += temporal_factor * 0.2;

        // Reputation of contributing agent
        if let Some(contributor) = self.agent_profiles.get(&knowledge.contributing_agent) {
            relevance += contributor.reputation_score * self.config.reputation_weight;
        }

        // Validation consensus
        let validation_ratio = if !knowledge.validation_votes.is_empty() {
            let positive_votes = knowledge.validation_votes.values()
                .filter(|(is_valid, _, _)| *is_valid)
                .count() as f32;
            positive_votes / knowledge.validation_votes.len() as f32
        } else {
            0.5 // Neutral if no validation
        };
        relevance += validation_ratio * 0.2;

        Ok(relevance.min(1.0))
    }

    async fn update_collaboration_connections(&self, requester_id: &str, retrieved: &[RetrievedMemory]) -> Result<(), MemoryError> {
        let mut graph = self.collaboration_graph.write().await;
        
        if let Some(connections) = graph.get_mut(requester_id) {
            for memory in retrieved {
                // Find the contributing agent for this knowledge
                if let Some(knowledge) = self.shared_knowledge.get(&memory.id) {
                    connections.insert(knowledge.contributing_agent.clone());
                }
            }
        }

        Ok(())
    }

    async fn expire_consensus_rounds(&self) -> Result<(), MemoryError> {
        let now = Utc::now();
        let mut expired_rounds = Vec::new();

        for entry in self.consensus_rounds.iter() {
            if entry.status == ConsensusStatus::Active && now > entry.deadline {
                expired_rounds.push(entry.key().clone());
            }
        }

        for round_id in expired_rounds {
            if let Some(mut round) = self.consensus_rounds.get_mut(&round_id) {
                round.status = ConsensusStatus::Timeout;
            }
        }

        Ok(())
    }

    async fn decay_contributions(&self) -> Result<(), MemoryError> {
        let cutoff = Utc::now() - Duration::days(self.config.contribution_decay_days);

        for mut profile in self.agent_profiles.iter_mut() {
            profile.contribution_history.retain(|contrib| contrib.timestamp > cutoff);
        }

        Ok(())
    }

    async fn update_reputation_scores(&self) -> Result<(), MemoryError> {
        for mut profile in self.agent_profiles.iter_mut() {
            // Decay reputation over time if no recent activity
            let days_inactive = Utc::now().signed_duration_since(profile.last_active).num_days();
            if days_inactive > 7 {
                let decay_factor = 0.99_f32.powi(days_inactive as i32 - 7);
                profile.reputation_score *= decay_factor;
            }

            // Boost reputation based on recent contributions
            let recent_contributions = profile.contribution_history.iter()
                .filter(|contrib| {
                    Utc::now().signed_duration_since(contrib.timestamp).num_days() < 7
                })
                .count();

            if recent_contributions > 0 {
                let activity_bonus = (recent_contributions as f32 * 0.01).min(0.1);
                profile.reputation_score = (profile.reputation_score + activity_bonus).min(1.0);
            }
        }

        Ok(())
    }

    async fn prune_inactive_agents(&self) -> Result<(), MemoryError> {
        let cutoff = Utc::now() - Duration::days(90); // 3 months inactive
        let mut to_remove = Vec::new();

        for entry in self.agent_profiles.iter() {
            if entry.last_active < cutoff && entry.reputation_score < 0.1 {
                to_remove.push(entry.key().clone());
            }
        }

        for agent_id in to_remove {
            self.agent_profiles.remove(&agent_id);
            
            let mut graph = self.collaboration_graph.write().await;
            graph.remove(&agent_id);
            
            // Remove from other agents' connections
            for connections in graph.values_mut() {
                connections.remove(&agent_id);
            }
        }

        Ok(())
    }

    async fn update_statistics(&self) -> Result<(), MemoryError> {
        self.stats.active_agents.store(self.agent_profiles.len(), Ordering::SeqCst);
        self.stats.shared_knowledge_count = self.shared_knowledge.len();
        self.stats.active_consensus_rounds = self.consensus_rounds.iter()
            .filter(|round| round.status == ConsensusStatus::Active)
            .count();

        Ok(())
    }
}