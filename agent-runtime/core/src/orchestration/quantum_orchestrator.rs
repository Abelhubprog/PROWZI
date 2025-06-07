use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, watch};
use tokio::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use tracing::{info, warn, error, debug};
use chrono;

use crate::actor::{Actor, ActorMessage};
use crate::messages::{Message, Priority};
use crate::performance::PerformanceMetrics;

pub use self::agent_types::{AgentType, AgentRole, AgentCapabilities};
pub use self::coordination_messages::{CoordinationMessage, CoordinationResponse};
pub use self::performance_monitor::{PerformanceMonitor, OrchestrationMetrics};

mod agent_types;
mod coordination_messages;
mod performance_monitor;

#[derive(Debug, Clone)]
pub struct AgentHandle {
    pub id: Uuid,
    pub agent_type: AgentType,
    pub role: AgentRole,
    pub capabilities: AgentCapabilities,
    pub performance_score: f64,
    pub health_status: HealthStatus,
    pub last_heartbeat: Instant,
    pub coordination_tx: mpsc::UnboundedSender<CoordinationMessage>,
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Critical { reason: String },
    Unresponsive,
}

#[derive(Debug, Clone)]
pub struct AgentRegistry {
    agents: HashMap<Uuid, AgentHandle>,
    type_index: HashMap<AgentType, Vec<Uuid>>,
    role_index: HashMap<AgentRole, Vec<Uuid>>,
    performance_rankings: Vec<(Uuid, f64)>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            type_index: HashMap::new(),
            role_index: HashMap::new(),
            performance_rankings: Vec::new(),
        }
    }

    pub fn register_agent(&mut self, handle: AgentHandle) -> Result<()> {
        let agent_id = handle.id;
        let agent_type = handle.agent_type.clone();
        let agent_role = handle.role.clone();
        let performance_score = handle.performance_score;

        // Update type index
        self.type_index.entry(agent_type)
            .or_insert_with(Vec::new)
            .push(agent_id);

        // Update role index
        self.role_index.entry(agent_role)
            .or_insert_with(Vec::new)
            .push(agent_id);

        // Insert into performance rankings
        self.performance_rankings.push((agent_id, performance_score));
        self.performance_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Register agent
        self.agents.insert(agent_id, handle);

        debug!("Registered agent {} with performance score {}", agent_id, performance_score);
        Ok(())
    }

    pub fn get_best_agent_for_role(&self, role: &AgentRole) -> Option<&AgentHandle> {
        self.role_index.get(role)?
            .iter()
            .filter_map(|id| self.agents.get(id))
            .filter(|agent| matches!(agent.health_status, HealthStatus::Healthy))
            .max_by(|a, b| a.performance_score.partial_cmp(&b.performance_score).unwrap())
    }

    pub fn get_agents_by_type(&self, agent_type: &AgentType) -> Vec<&AgentHandle> {
        self.type_index.get(agent_type)
            .map(|ids| ids.iter().filter_map(|id| self.agents.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn update_health_status(&mut self, agent_id: Uuid, status: HealthStatus) -> Result<()> {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.health_status = status;
            agent.last_heartbeat = Instant::now();
            Ok(())
        } else {
            Err(anyhow!("Agent {} not found", agent_id))
        }
    }

    pub fn update_performance_score(&mut self, agent_id: Uuid, score: f64) -> Result<()> {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.performance_score = score;
            
            // Update performance rankings
            if let Some(pos) = self.performance_rankings.iter().position(|(id, _)| *id == agent_id) {
                self.performance_rankings[pos].1 = score;
                self.performance_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
            Ok(())
        } else {
            Err(anyhow!("Agent {} not found", agent_id))
        }
    }
}

pub struct CoordinationEngine {
    message_router: Arc<MessageRouter>,
    load_balancer: Arc<LoadBalancer>,
    failure_detector: Arc<FailureDetector>,
    ai_optimizer: Arc<AIOptimizer>,
}

impl CoordinationEngine {
    pub fn new() -> Self {
        Self {
            message_router: Arc::new(MessageRouter::new()),
            load_balancer: Arc::new(LoadBalancer::new()),
            failure_detector: Arc::new(FailureDetector::new()),
            ai_optimizer: Arc::new(AIOptimizer::new()),
        }
    }

    pub async fn coordinate_agents(
        &self,
        registry: &Arc<RwLock<AgentRegistry>>,
        task: CoordinationMessage,
    ) -> Result<CoordinationResponse> {
        let start_time = Instant::now();
        
        // AI-driven agent selection
        let selected_agents = self.ai_optimizer.select_optimal_agents(registry, &task).await?;
        
        // Route messages with priority handling
        let responses = self.message_router.route_to_agents(selected_agents, task).await?;
        
        // Aggregate and optimize response
        let final_response = self.ai_optimizer.aggregate_responses(responses).await?;
        
        let coordination_time = start_time.elapsed();
        debug!("Coordination completed in {:?}", coordination_time);
        
        Ok(final_response)
    }

    pub async fn handle_agent_failure(&self, registry: &Arc<RwLock<AgentRegistry>>, agent_id: Uuid) -> Result<()> {
        warn!("Handling agent failure for {}", agent_id);
        
        // Update agent status
        {
            let mut registry = registry.write().await;
            registry.update_health_status(agent_id, HealthStatus::Critical { 
                reason: "Failure detected".to_string() 
            })?;
        }
        
        // Trigger self-healing
        self.failure_detector.initiate_recovery(registry, agent_id).await?;
        
        Ok(())
    }
}

struct MessageRouter {
    priority_queues: HashMap<Priority, mpsc::UnboundedSender<(CoordinationMessage, mpsc::UnboundedSender<CoordinationResponse>)>>,
}

impl MessageRouter {
    fn new() -> Self {
        Self {
            priority_queues: HashMap::new(),
        }
    }

    async fn route_to_agents(
        &self,
        agents: Vec<Uuid>,
        message: CoordinationMessage,
    ) -> Result<Vec<CoordinationResponse>> {
        let mut responses = Vec::new();
        let message_priority = message.priority();
        
        // Parallel message dispatch with priority handling
        let handles: Vec<_> = agents.into_iter().map(|agent_id| {
            let msg = message.clone();
            tokio::spawn(async move {
                // Simulated ultra-low-latency message sending
                tokio::time::sleep(Duration::from_micros(100)).await;
                CoordinationResponse::Success {
                    agent_id,
                    data: format!("Processed {}", msg.message_type()),
                    processing_time: Duration::from_micros(500),
                }
            })
        }).collect();
        
        // Collect responses with timeout
        for handle in handles {
            match tokio::time::timeout(Duration::from_millis(10), handle).await {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => error!("Agent task failed: {}", e),
                Err(_) => warn!("Agent response timeout"),
            }
        }
        
        Ok(responses)
    }
}

struct LoadBalancer {
    current_loads: HashMap<Uuid, f64>,
}

impl LoadBalancer {
    fn new() -> Self {
        Self {
            current_loads: HashMap::new(),
        }
    }

    fn select_least_loaded_agents(&self, candidates: Vec<Uuid>, count: usize) -> Vec<Uuid> {
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| {
            let load_a = self.current_loads.get(a).unwrap_or(&0.0);
            let load_b = self.current_loads.get(b).unwrap_or(&0.0);
            load_a.partial_cmp(load_b).unwrap()
        });
        
        sorted_candidates.into_iter().take(count).collect()
    }
}

struct FailureDetector {
    failure_patterns: HashMap<Uuid, Vec<Instant>>,
    recovery_strategies: HashMap<AgentType, RecoveryStrategy>,
}

#[derive(Debug, Clone)]
enum RecoveryStrategy {
    Restart,
    Migrate,
    Replicate,
    GracefulDegradation,
}

impl FailureDetector {
    fn new() -> Self {
        let mut recovery_strategies = HashMap::new();
        recovery_strategies.insert(AgentType::Scout, RecoveryStrategy::Replicate);
        recovery_strategies.insert(AgentType::Planner, RecoveryStrategy::Migrate);
        recovery_strategies.insert(AgentType::Trader, RecoveryStrategy::Restart);
        recovery_strategies.insert(AgentType::RiskSentinel, RecoveryStrategy::Replicate);
        recovery_strategies.insert(AgentType::Guardian, RecoveryStrategy::GracefulDegradation);
        
        Self {
            failure_patterns: HashMap::new(),
            recovery_strategies,
        }
    }

    async fn initiate_recovery(&self, registry: &Arc<RwLock<AgentRegistry>>, agent_id: Uuid) -> Result<()> {
        let agent_type = {
            let registry = registry.read().await;
            registry.agents.get(&agent_id)
                .map(|agent| agent.agent_type.clone())
                .ok_or_else(|| anyhow!("Agent {} not found", agent_id))?
        };

        let strategy = self.recovery_strategies.get(&agent_type)
            .unwrap_or(&RecoveryStrategy::Restart);

        match strategy {
            RecoveryStrategy::Restart => {
                info!("Restarting agent {}", agent_id);
                // Implement agent restart logic
            },
            RecoveryStrategy::Migrate => {
                info!("Migrating agent {} to new instance", agent_id);
                // Implement agent migration logic
            },
            RecoveryStrategy::Replicate => {
                info!("Creating replica for agent {}", agent_id);
                // Implement agent replication logic
            },
            RecoveryStrategy::GracefulDegradation => {
                info!("Initiating graceful degradation for agent {}", agent_id);
                // Implement graceful degradation logic
            },
        }

        Ok(())
    }
}

struct AIOptimizer {
    ml_models: HashMap<String, MLModel>,
    performance_history: Vec<PerformanceSnapshot>,
}

#[derive(Debug, Clone)]
struct MLModel {
    model_type: String,
    accuracy: f64,
    last_trained: Instant,
}

#[derive(Debug, Clone)]
struct PerformanceSnapshot {
    timestamp: Instant,
    agent_performances: HashMap<Uuid, f64>,
    coordination_metrics: OrchestrationMetrics,
}

// Enhanced coordination support structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    pub proposal_id: String,
    pub proposal_type: String,
    pub data: serde_json::Value,
    pub deadline: chrono::DateTime<chrono::Utc>,
    pub minimum_approval_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub consensus_id: Uuid,
    pub decision: ConsensusDecision,
    pub participating_agents: Vec<Uuid>,
    pub vote_summary: VoteSummary,
    pub finalized_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusDecision {
    Approved,
    Rejected,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteSummary {
    pub approve: usize,
    pub reject: usize,
    pub abstain: usize,
    pub total_participants: usize,
}

impl Default for VoteSummary {
    fn default() -> Self {
        Self {
            approve: 0,
            reject: 0,
            abstain: 0,
            total_participants: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpawnRequest {
    pub agent_type: AgentType,
    pub agent_role: AgentRole,
    pub capabilities: AgentCapabilities,
    pub resource_requirements: ResourceRequirements,
    pub security_level: SecurityLevel,
    pub lifecycle_policy: LifecyclePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub gpu_memory_mb: u64,
    pub network_bandwidth_mbps: f64,
    pub estimated_processing_time_ms: u64,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_mb: 512,
            gpu_memory_mb: 0,
            network_bandwidth_mbps: 10.0,
            estimated_processing_time_ms: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecyclePolicy {
    Persistent,
    TaskBound,
    TimeBound(Duration),
    ResourceBound(ResourceRequirements),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpec {
    pub pipeline_id: String,
    pub stages: Vec<PipelineStage>,
    pub initial_data: serde_json::Value,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub stage_id: String,
    pub required_capabilities: Vec<String>,
    pub priority: Priority,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub pipeline_id: String,
    pub stages: Vec<PipelineStageResult>,
    pub total_processing_time: Duration,
    pub status: PipelineStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageResult {
    pub stage_id: String,
    pub agent_id: Option<Uuid>,
    pub processing_time: Duration,
    pub output_data: serde_json::Value,
    pub status: StageStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRule {
    pub rule_id: String,
    pub condition: EventCondition,
    pub action: EventAction,
    pub check_interval_ms: u64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCondition {
    pub condition_type: String,
    pub parameters: serde_json::Value,
}

impl EventCondition {
    pub async fn evaluate(&self) -> bool {
        // Simplified condition evaluation
        // In production, this would evaluate complex conditions
        match self.condition_type.as_str() {
            "market_volatility" => true, // Simulate condition met
            "system_load" => false,
            "agent_failure" => false,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventAction {
    SpawnAgent(AgentSpawnRequest),
    InitiateConsensus {
        agents: Vec<Uuid>,
        proposal: ConsensusProposal,
    },
    ExecutePipeline(PipelineSpec),
    SendAlert {
        message: String,
        priority: String,
    },
}

impl AIOptimizer {
    fn new() -> Self {
        let mut ml_models = HashMap::new();
        ml_models.insert("agent_selection".to_string(), MLModel {
            model_type: "gradient_boosting".to_string(),
            accuracy: 0.95,
            last_trained: Instant::now(),
        });
        ml_models.insert("load_prediction".to_string(), MLModel {
            model_type: "lstm".to_string(),
            accuracy: 0.92,
            last_trained: Instant::now(),
        });

        Self {
            ml_models,
            performance_history: Vec::new(),
        }
    }

    async fn select_optimal_agents(
        &self,
        registry: &Arc<RwLock<AgentRegistry>>,
        task: &CoordinationMessage,
    ) -> Result<Vec<Uuid>> {
        let registry = registry.read().await;
        
        // AI-driven agent selection based on task requirements and historical performance
        let required_role = task.required_role();
        let candidates = registry.role_index.get(&required_role)
            .cloned()
            .unwrap_or_default();

        // Filter healthy agents
        let healthy_candidates: Vec<Uuid> = candidates.into_iter()
            .filter(|id| {
                registry.agents.get(id)
                    .map(|agent| matches!(agent.health_status, HealthStatus::Healthy))
                    .unwrap_or(false)
            })
            .collect();

        // ML-based selection (simplified simulation)
        let optimal_count = std::cmp::min(3, healthy_candidates.len());
        let selected = healthy_candidates.into_iter().take(optimal_count).collect();
        
        debug!("Selected {} agents for coordination task", selected.len());
        Ok(selected)
    }

    async fn aggregate_responses(&self, responses: Vec<CoordinationResponse>) -> Result<CoordinationResponse> {
        if responses.is_empty() {
            return Ok(CoordinationResponse::Error {
                message: "No responses received".to_string(),
            });
        }

        // AI-powered response aggregation with confidence scoring
        let mut success_responses = Vec::new();
        let mut total_processing_time = Duration::new(0, 0);

        for response in responses {
            if let CoordinationResponse::Success { agent_id, data, processing_time } = response {
                success_responses.push((agent_id, data));
                total_processing_time += processing_time;
            }
        }

        if success_responses.is_empty() {
            return Ok(CoordinationResponse::Error {
                message: "All agent responses failed".to_string(),
            });
        }

        // Aggregate successful responses
        let aggregated_data = success_responses.into_iter()
            .map(|(id, data)| format!("Agent {}: {}", id, data))
            .collect::<Vec<_>>()
            .join("; ");

        Ok(CoordinationResponse::Success {
            agent_id: Uuid::new_v4(), // Orchestrator ID
            data: format!("Aggregated: {}", aggregated_data),
            processing_time: total_processing_time,
        })
    }
}

pub struct QuantumOrchestrator {
    agent_registry: Arc<RwLock<AgentRegistry>>,
    coordination_engine: Arc<CoordinationEngine>,
    performance_monitor: Arc<PerformanceMonitor>,
    message_bus: Arc<MessageBus>,
    shutdown_signal: watch::Receiver<bool>,
}

impl QuantumOrchestrator {
    pub fn new() -> (Self, watch::Sender<bool>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        
        let orchestrator = Self {
            agent_registry: Arc::new(RwLock::new(AgentRegistry::new())),
            coordination_engine: Arc::new(CoordinationEngine::new()),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
            message_bus: Arc::new(MessageBus::new()),
            shutdown_signal: shutdown_rx,
        };

        (orchestrator, shutdown_tx)
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting Quantum Orchestrator");

        // Start performance monitoring
        let monitor_task = {
            let monitor = Arc::clone(&self.performance_monitor);
            let registry = Arc::clone(&self.agent_registry);
            let mut shutdown = self.shutdown_signal.clone();
            
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = shutdown.changed() => {
                            if *shutdown.borrow() {
                                break;
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_secs(1)) => {
                            if let Err(e) = monitor.collect_metrics(&registry).await {
                                error!("Failed to collect metrics: {}", e);
                            }
                        }
                    }
                }
            })
        };

        // Start coordination loop
        let coordination_task = {
            let engine = Arc::clone(&self.coordination_engine);
            let registry = Arc::clone(&self.agent_registry);
            let message_bus = Arc::clone(&self.message_bus);
            let mut shutdown = self.shutdown_signal.clone();
            
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = shutdown.changed() => {
                            if *shutdown.borrow() {
                                break;
                            }
                        }
                        Some(coordination_request) = message_bus.receive_coordination_request() => {
                            if let Err(e) = engine.coordinate_agents(&registry, coordination_request).await {
                                error!("Coordination failed: {}", e);
                            }
                        }
                    }
                }
            })
        };

        // Wait for shutdown or error
        tokio::select! {
            _ = self.shutdown_signal.changed() => {
                info!("Shutdown signal received");
            }
            result = tokio::try_join!(monitor_task, coordination_task) => {
                if let Err(e) = result {
                    error!("Orchestrator task failed: {}", e);
                    return Err(anyhow!("Orchestrator task failed: {}", e));
                }
            }
        }

        info!("Quantum Orchestrator stopped");
        Ok(())
    }

    pub async fn register_agent(&self, agent_handle: AgentHandle) -> Result<()> {
        let mut registry = self.agent_registry.write().await;
        registry.register_agent(agent_handle)?;
        Ok(())
    }

    pub async fn coordinate_task(&self, task: CoordinationMessage) -> Result<CoordinationResponse> {
        self.coordination_engine.coordinate_agents(&self.agent_registry, task).await
    }

    pub async fn get_orchestration_metrics(&self) -> Result<OrchestrationMetrics> {
        self.performance_monitor.get_current_metrics().await
    }

    /// Enable direct agent-to-agent communication
    pub async fn send_direct_message(&self, target_agent_id: Uuid, message: CoordinationMessage) -> Result<()> {
        let registry = self.agent_registry.read().await;
        if let Some(target_agent) = registry.agents.get(&target_agent_id) {
            // Send message directly to agent's coordination channel
            target_agent.coordination_tx.send(message)
                .map_err(|_| anyhow!("Failed to send message to agent {}", target_agent_id))?;
            debug!("Sent direct message to agent {}", target_agent_id);
        } else {
            return Err(anyhow!("Agent {} not found", target_agent_id));
        }
        Ok(())
    }

    /// Support for consensus-based decision making among agents
    pub async fn initiate_consensus(&self, agents: Vec<Uuid>, proposal: ConsensusProposal) -> Result<ConsensusResult> {
        let consensus_id = Uuid::new_v4();
        let mut vote_responses = Vec::new();
        
        // Send proposal to all participating agents
        for agent_id in &agents {
            let consensus_message = CoordinationMessage::SystemCoordination {
                data: serde_json::json!({
                    "consensus_id": consensus_id,
                    "proposal": proposal,
                    "participants": agents,
                    "action": "vote_request"
                }),
                priority: Priority::High,
            };
            
            self.send_direct_message(*agent_id, consensus_message).await?;
        }
        
        // Wait for votes with timeout
        let timeout = tokio::time::timeout(
            Duration::from_millis(30000),
            self.collect_consensus_votes(consensus_id, agents.len())
        );
        
        match timeout.await {
            Ok(result) => result,
            Err(_) => Ok(ConsensusResult {
                consensus_id,
                decision: ConsensusDecision::Timeout,
                participating_agents: agents,
                vote_summary: VoteSummary::default(),
                finalized_at: chrono::Utc::now(),
            })
        }
    }
    
    async fn collect_consensus_votes(&self, consensus_id: Uuid, expected_votes: usize) -> Result<ConsensusResult> {
        // This would collect votes from agents through the message system
        // For now, simulate consensus logic
        let vote_summary = VoteSummary {
            approve: expected_votes * 7 / 10, // 70% approval
            reject: expected_votes * 2 / 10,  // 20% rejection  
            abstain: expected_votes * 1 / 10, // 10% abstain
            total_participants: expected_votes,
        };
        
        let decision = if vote_summary.approve as f64 / expected_votes as f64 > 0.6 {
            ConsensusDecision::Approved
        } else {
            ConsensusDecision::Rejected
        };
        
        Ok(ConsensusResult {
            consensus_id,
            decision,
            participating_agents: Vec::new(), // Would be populated from actual votes
            vote_summary,
            finalized_at: chrono::Utc::now(),
        })
    }

    /// Dynamic agent spawning based on workload demands
    pub async fn spawn_agent(&self, agent_spec: AgentSpawnRequest) -> Result<Uuid> {
        let agent_id = Uuid::new_v4();
        
        // Validate resource availability
        if !self.can_allocate_resources(&agent_spec.resource_requirements).await? {
            return Err(anyhow!("Insufficient resources to spawn agent"));
        }
        
        // Create agent communication channel
        let (coordination_tx, mut coordination_rx) = mpsc::unbounded_channel();
        
        // Create agent handle
        let agent_handle = AgentHandle {
            id: agent_id,
            agent_type: agent_spec.agent_type.clone(),
            role: agent_spec.agent_role,
            capabilities: agent_spec.capabilities,
            performance_score: 0.5, // Initial score
            health_status: HealthStatus::Healthy,
            last_heartbeat: Instant::now(),
            coordination_tx,
        };
        
        // Register with orchestrator
        self.register_agent(agent_handle).await?;
        
        // Start agent communication handler
        let agent_id_clone = agent_id;
        tokio::spawn(async move {
            while let Some(message) = coordination_rx.recv().await {
                debug!("Agent {} received message: {:?}", agent_id_clone, message);
                // Process coordination messages
            }
        });
        
        info!("Successfully spawned agent: {}", agent_id);
        Ok(agent_id)
    }
    
    async fn can_allocate_resources(&self, requirements: &ResourceRequirements) -> Result<bool> {
        let current_usage = self.get_total_resource_usage().await?;
        let total_capacity = self.get_total_capacity().await?;
        
        let cpu_available = total_capacity.cpu_cores - current_usage.cpu_cores;
        let memory_available = total_capacity.memory_mb - current_usage.memory_mb;
        let gpu_available = total_capacity.gpu_memory_mb - current_usage.gpu_memory_mb;
        
        Ok(requirements.cpu_cores <= cpu_available &&
           requirements.memory_mb <= memory_available &&
           requirements.gpu_memory_mb <= gpu_available)
    }
    
    async fn get_total_resource_usage(&self) -> Result<ResourceRequirements> {
        let registry = self.agent_registry.read().await;
        let mut total = ResourceRequirements::default();
        
        // Sum up resource usage from all active agents
        for agent in registry.agents.values() {
            if matches!(agent.health_status, HealthStatus::Healthy) {
                total.cpu_cores += 1.0; // Simplified: each agent uses 1 CPU core
                total.memory_mb += 512;  // Simplified: each agent uses 512MB
                total.gpu_memory_mb += 256; // Simplified: each agent uses 256MB GPU
            }
        }
        
        Ok(total)
    }
    
    async fn get_total_capacity(&self) -> Result<ResourceRequirements> {
        // In production, this would query actual system resources
        Ok(ResourceRequirements {
            cpu_cores: 32.0,      // 32 CPU cores
            memory_mb: 128000,    // 128GB RAM
            gpu_memory_mb: 24000, // 24GB GPU memory
            network_bandwidth_mbps: 1000.0,
            estimated_processing_time_ms: 0,
        })
    }

    /// Hierarchical agent management - create parent-child relationships
    pub async fn create_agent_hierarchy(&self, parent_id: Uuid, child_specs: Vec<AgentSpawnRequest>) -> Result<Vec<Uuid>> {
        let mut child_agents = Vec::new();
        
        for child_spec in child_specs {
            let child_id = self.spawn_agent(child_spec).await?;
            child_agents.push(child_id);
        }
        
        // Update parent agent metadata to include child references
        info!("Created agent hierarchy - Parent: {}, Children: {:?}", parent_id, child_agents);
        
        Ok(child_agents)
    }

    /// Pipeline coordination - sequential task processing through agent chains
    pub async fn execute_pipeline(&self, pipeline_spec: PipelineSpec) -> Result<PipelineResult> {
        let mut pipeline_result = PipelineResult {
            pipeline_id: pipeline_spec.pipeline_id.clone(),
            stages: Vec::new(),
            total_processing_time: Duration::new(0, 0),
            status: PipelineStatus::Running,
        };
        
        let start_time = Instant::now();
        let mut current_data = pipeline_spec.initial_data;
        
        for (stage_index, stage) in pipeline_spec.stages.iter().enumerate() {
            let stage_start = Instant::now();
            
            // Execute stage with assigned agent
            let coordination_message = CoordinationMessage::MarketIntelligence {
                data: current_data.clone(),
                priority: stage.priority,
                required_capabilities: stage.required_capabilities.clone(),
            };
            
            let stage_result = self.coordinate_task(coordination_message).await?;
            
            let stage_duration = stage_start.elapsed();
            
            // Update pipeline state
            let pipeline_stage_result = PipelineStageResult {
                stage_id: stage.stage_id.clone(),
                agent_id: None, // Would be filled with actual agent ID
                processing_time: stage_duration,
                output_data: match stage_result {
                    CoordinationResponse::Success { data, .. } => serde_json::json!(data),
                    CoordinationResponse::Error { message } => {
                        pipeline_result.status = PipelineStatus::Failed;
                        return Err(anyhow!("Pipeline stage {} failed: {}", stage.stage_id, message));
                    }
                },
                status: StageStatus::Completed,
            };
            
            pipeline_result.stages.push(pipeline_stage_result);
            
            // Pass output as input to next stage
            if let CoordinationResponse::Success { data, .. } = stage_result {
                current_data = serde_json::json!(data);
            }
        }
        
        pipeline_result.total_processing_time = start_time.elapsed();
        pipeline_result.status = PipelineStatus::Completed;
        
        Ok(pipeline_result)
    }

    /// Event-driven coordination based on market events or system triggers
    pub async fn setup_event_coordination(&self, event_rules: Vec<EventRule>) -> Result<()> {
        for rule in event_rules {
            let rule_clone = rule.clone();
            let orchestrator = self.clone();
            
            tokio::spawn(async move {
                // Event monitoring loop
                loop {
                    // Check if event condition is met
                    if rule_clone.condition.evaluate().await {
                        // Trigger coordination action
                        if let Err(e) = orchestrator.handle_event_trigger(&rule_clone).await {
                            error!("Failed to handle event trigger for rule {}: {}", rule_clone.rule_id, e);
                        }
                    }
                    
                    // Wait before next check
                    tokio::time::sleep(Duration::from_millis(rule_clone.check_interval_ms)).await;
                }
            });
        }
        
        Ok(())
    }
    
    async fn handle_event_trigger(&self, rule: &EventRule) -> Result<()> {
        info!("Event rule {} triggered", rule.rule_id);
        
        // Execute rule action
        match &rule.action {
            EventAction::SpawnAgent(spec) => {
                self.spawn_agent(spec.clone()).await?;
            },
            EventAction::InitiateConsensus { agents, proposal } => {
                self.initiate_consensus(agents.clone(), proposal.clone()).await?;
            },
            EventAction::ExecutePipeline(spec) => {
                self.execute_pipeline(spec.clone()).await?;
            },
            EventAction::SendAlert { message, priority } => {
                // Send alert through message bus
                info!("Event alert ({}): {}", priority, message);
            },
        }
        
        Ok(())
    }
}

// Clone implementation for QuantumOrchestrator
impl Clone for QuantumOrchestrator {
    fn clone(&self) -> Self {
        Self {
            agent_registry: Arc::clone(&self.agent_registry),
            coordination_engine: Arc::clone(&self.coordination_engine),
            performance_monitor: Arc::clone(&self.performance_monitor),
            message_bus: Arc::clone(&self.message_bus),
            shutdown_signal: self.shutdown_signal.clone(),
        }
    }
}

struct MessageBus {
    coordination_requests: mpsc::UnboundedReceiver<CoordinationMessage>,
    _coordination_tx: mpsc::UnboundedSender<CoordinationMessage>,
}

impl MessageBus {
    fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            coordination_requests: rx,
            _coordination_tx: tx,
        }
    }

    async fn receive_coordination_request(&mut self) -> Option<CoordinationMessage> {
        self.coordination_requests.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_registry() {
        let mut registry = AgentRegistry::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        
        let agent = AgentHandle {
            id: Uuid::new_v4(),
            agent_type: AgentType::Scout,
            role: AgentRole::MarketIntelligence,
            capabilities: AgentCapabilities::default(),
            performance_score: 0.85,
            health_status: HealthStatus::Healthy,
            last_heartbeat: Instant::now(),
            coordination_tx: tx,
        };

        assert!(registry.register_agent(agent).is_ok());
        assert_eq!(registry.agents.len(), 1);
    }

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let (orchestrator, _shutdown) = QuantumOrchestrator::new();
        assert!(orchestrator.agent_registry.read().await.agents.is_empty());
    }
}