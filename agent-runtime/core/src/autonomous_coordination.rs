use crate::{Actor, Budget, Message};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapability {
    pub name: String,
    pub description: String,
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
    pub resource_requirements: ResourceRequirements,
    pub performance_metrics: PerformanceMetrics,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub gpu_memory_mb: u64,
    pub network_bandwidth_mbps: f64,
    pub storage_gb: f64,
    pub estimated_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub success_rate: f64,
    pub average_latency_ms: u64,
    pub throughput_per_second: f64,
    pub accuracy_score: f64,
    pub reliability_score: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub agent_id: String,
    pub agent_type: AgentType,
    pub capabilities: Vec<AgentCapability>,
    pub current_status: AgentStatus,
    pub load_factor: f64,
    pub priority_level: u8,
    pub trust_score: f64,
    pub specialization: Vec<String>,
    pub collaboration_history: Vec<CollaborationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    Scout,
    Planner,
    Trader,
    RiskSentinel,
    Guardian,
    Analyzer,
    Coordinator,
    Specialist(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Working(TaskContext),
    Collaborating(Vec<String>),
    Learning,
    Maintenance,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub task_id: String,
    pub task_type: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub estimated_completion: chrono::DateTime<chrono::Utc>,
    pub progress_percentage: f64,
    pub dependencies: Vec<String>,
    pub collaborators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationRecord {
    pub partner_agent_id: String,
    pub task_type: String,
    pub outcome: CollaborationOutcome,
    pub efficiency_score: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationOutcome {
    Success,
    PartialSuccess,
    Failed,
    Timeout,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task_id: String,
    pub task_type: String,
    pub priority: TaskPriority,
    pub requirements: TaskRequirements,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub dependencies: Vec<String>,
    pub callback_endpoint: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Emergency,
    High,
    Medium,
    Low,
    Background,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    pub required_capabilities: Vec<String>,
    pub minimum_trust_score: f64,
    pub resource_constraints: ResourceRequirements,
    pub collaboration_preferences: CollaborationPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationPreferences {
    pub prefer_specialized_agents: bool,
    pub maximum_collaborators: u8,
    pub require_consensus: bool,
    pub parallel_execution: bool,
    pub cross_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAssignment {
    pub task_id: String,
    pub assigned_agents: Vec<String>,
    pub lead_agent: String,
    pub assignment_strategy: AssignmentStrategy,
    pub expected_completion: chrono::DateTime<chrono::Utc>,
    pub monitoring_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignmentStrategy {
    SingleAgent,
    PrimaryWithBackup,
    ParallelExecution,
    Pipeline,
    Consensus,
    Hierarchical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub results: Vec<AgentResult>,
    pub performance_summary: PerformanceSummary,
    pub lessons_learned: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Assigned,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    RequiresIntervention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub agent_id: String,
    pub result_data: serde_json::Value,
    pub confidence_score: f64,
    pub execution_time_ms: u64,
    pub resource_usage: ResourceUsage,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_time_ms: u64,
    pub memory_peak_mb: u64,
    pub gpu_time_ms: u64,
    pub network_bytes: u64,
    pub storage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_execution_time_ms: u64,
    pub efficiency_score: f64,
    pub accuracy_score: f64,
    pub resource_utilization: f64,
    pub collaboration_effectiveness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningUpdate {
    pub agent_id: String,
    pub learned_patterns: Vec<Pattern>,
    pub performance_improvements: Vec<Improvement>,
    pub new_capabilities: Vec<AgentCapability>,
    pub updated_trust_scores: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_type: String,
    pub description: String,
    pub confidence: f64,
    pub applicability: Vec<String>,
    pub discovered_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    pub metric_name: String,
    pub previous_value: f64,
    pub new_value: f64,
    pub improvement_percentage: f64,
    pub context: String,
}

pub struct AutonomousCoordinator {
    agent_registry: Arc<RwLock<HashMap<String, AgentProfile>>>,
    task_queue: Arc<RwLock<Vec<TaskRequest>>>,
    active_assignments: Arc<RwLock<HashMap<String, TaskAssignment>>>,
    completed_tasks: Arc<RwLock<Vec<CoordinationResult>>>,
    learning_engine: Arc<RwLock<LearningEngine>>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    coordination_strategies: Arc<RwLock<Vec<CoordinationStrategy>>>,
}

pub struct LearningEngine {
    pattern_database: HashMap<String, Vec<Pattern>>,
    performance_history: HashMap<String, Vec<PerformanceMetrics>>,
    collaboration_graph: HashMap<String, HashMap<String, f64>>,
    optimization_rules: Vec<OptimizationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRule {
    pub rule_id: String,
    pub condition: String,
    pub action: String,
    pub confidence: f64,
    pub success_rate: f64,
}

pub struct PerformanceTracker {
    real_time_metrics: HashMap<String, Vec<MetricPoint>>,
    aggregated_statistics: HashMap<String, AggregatedStats>,
    anomaly_detection: AnomalyDetector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub value: f64,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedStats {
    pub mean: f64,
    pub median: f64,
    pub std_deviation: f64,
    pub percentile_95: f64,
    pub trend: TrendDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Unknown,
}

pub struct AnomalyDetector {
    models: HashMap<String, AnomalyModel>,
    thresholds: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyModel {
    pub baseline_mean: f64,
    pub baseline_std: f64,
    pub sensitivity: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStrategy {
    pub name: String,
    pub applicable_scenarios: Vec<String>,
    pub selection_criteria: SelectionCriteria,
    pub execution_algorithm: ExecutionAlgorithm,
    pub success_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    pub task_complexity: f64,
    pub time_constraints: bool,
    pub resource_availability: f64,
    pub collaboration_requirements: bool,
    pub risk_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionAlgorithm {
    GreedySelection,
    OptimalAllocation,
    MachineLearningBased,
    HybridApproach,
    ConsensusBuilding,
}

impl AutonomousCoordinator {
    pub fn new() -> Self {
        let learning_engine = LearningEngine {
            pattern_database: HashMap::new(),
            performance_history: HashMap::new(),
            collaboration_graph: HashMap::new(),
            optimization_rules: Vec::new(),
        };

        let performance_tracker = PerformanceTracker {
            real_time_metrics: HashMap::new(),
            aggregated_statistics: HashMap::new(),
            anomaly_detection: AnomalyDetector {
                models: HashMap::new(),
                thresholds: HashMap::new(),
            },
        };

        let mut strategies = Vec::new();
        strategies.push(CoordinationStrategy {
            name: "Emergency Response".to_string(),
            applicable_scenarios: vec!["emergency".to_string(), "critical_failure".to_string()],
            selection_criteria: SelectionCriteria {
                task_complexity: 0.8,
                time_constraints: true,
                resource_availability: 0.5,
                collaboration_requirements: false,
                risk_tolerance: 0.2,
            },
            execution_algorithm: ExecutionAlgorithm::GreedySelection,
            success_metrics: vec!["response_time".to_string(), "problem_resolution".to_string()],
        });

        strategies.push(CoordinationStrategy {
            name: "Collaborative Analysis".to_string(),
            applicable_scenarios: vec!["market_analysis".to_string(), "strategy_optimization".to_string()],
            selection_criteria: SelectionCriteria {
                task_complexity: 0.9,
                time_constraints: false,
                resource_availability: 0.8,
                collaboration_requirements: true,
                risk_tolerance: 0.7,
            },
            execution_algorithm: ExecutionAlgorithm::ConsensusBuilding,
            success_metrics: vec!["accuracy".to_string(), "consensus_quality".to_string()],
        });

        Self {
            agent_registry: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(RwLock::new(Vec::new())),
            active_assignments: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(Vec::new())),
            learning_engine: Arc::new(RwLock::new(learning_engine)),
            performance_tracker: Arc::new(RwLock::new(performance_tracker)),
            coordination_strategies: Arc::new(RwLock::new(strategies)),
        }
    }

    pub async fn register_agent(&self, profile: AgentProfile) -> Result<()> {
        let mut registry = self.agent_registry.write().await;
        registry.insert(profile.agent_id.clone(), profile);
        Ok(())
    }

    pub async fn submit_task(&self, task: TaskRequest) -> Result<String> {
        let task_id = task.task_id.clone();
        
        // Add to task queue
        let mut queue = self.task_queue.write().await;
        queue.push(task);
        
        // Sort by priority
        queue.sort_by(|a, b| {
            let priority_order = |p: &TaskPriority| match p {
                TaskPriority::Emergency => 0,
                TaskPriority::High => 1,
                TaskPriority::Medium => 2,
                TaskPriority::Low => 3,
                TaskPriority::Background => 4,
            };
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
        });

        // Trigger assignment process
        tokio::spawn({
            let coordinator = self.clone();
            async move {
                if let Err(e) = coordinator.process_task_queue().await {
                    log::error!("Error processing task queue: {}", e);
                }
            }
        });

        Ok(task_id)
    }

    async fn process_task_queue(&self) -> Result<()> {
        let mut queue = self.task_queue.write().await;
        let mut tasks_to_process = Vec::new();

        // Take tasks that can be processed immediately
        while let Some(task) = queue.first() {
            if self.can_assign_task(task).await? {
                tasks_to_process.push(queue.remove(0));
            } else {
                break; // Wait for resources to become available
            }
        }
        drop(queue);

        for task in tasks_to_process {
            self.assign_task(task).await?;
        }

        Ok(())
    }

    async fn can_assign_task(&self, task: &TaskRequest) -> Result<bool> {
        let registry = self.agent_registry.read().await;
        
        // Check if we have agents with required capabilities
        let suitable_agents = self.find_suitable_agents(task, &registry).await?;
        
        Ok(!suitable_agents.is_empty())
    }

    async fn find_suitable_agents(&self, task: &TaskRequest, registry: &HashMap<String, AgentProfile>) -> Result<Vec<String>> {
        let mut suitable_agents = Vec::new();

        for (agent_id, profile) in registry {
            // Check if agent has required capabilities
            let has_capabilities = task.requirements.required_capabilities.iter()
                .all(|req_cap| profile.capabilities.iter()
                    .any(|cap| &cap.name == req_cap));

            // Check trust score
            let meets_trust = profile.trust_score >= task.requirements.minimum_trust_score;

            // Check availability
            let is_available = matches!(profile.current_status, AgentStatus::Idle) 
                || profile.load_factor < 0.8;

            if has_capabilities && meets_trust && is_available {
                suitable_agents.push(agent_id.clone());
            }
        }

        Ok(suitable_agents)
    }

    async fn assign_task(&self, task: TaskRequest) -> Result<()> {
        let registry = self.agent_registry.read().await;
        let suitable_agents = self.find_suitable_agents(&task, &registry).await?;
        
        if suitable_agents.is_empty() {
            return Err(anyhow!("No suitable agents available for task: {}", task.task_id));
        }

        // Select best assignment strategy
        let strategy = self.select_coordination_strategy(&task).await?;
        
        // Create assignment based on strategy
        let assignment = self.create_assignment(&task, &suitable_agents, &strategy).await?;
        
        // Store assignment
        let mut assignments = self.active_assignments.write().await;
        assignments.insert(task.task_id.clone(), assignment.clone());
        drop(assignments);

        // Notify assigned agents
        self.notify_agents(&assignment).await?;

        // Start monitoring
        self.start_task_monitoring(&assignment).await?;

        Ok(())
    }

    async fn select_coordination_strategy(&self, task: &TaskRequest) -> Result<CoordinationStrategy> {
        let strategies = self.coordination_strategies.read().await;
        
        // Find applicable strategies
        let mut applicable = Vec::new();
        for strategy in strategies.iter() {
            if strategy.applicable_scenarios.iter().any(|scenario| {
                task.task_type.contains(scenario) || 
                task.metadata.contains_key(scenario)
            }) {
                applicable.push(strategy.clone());
            }
        }

        if applicable.is_empty() {
            // Default strategy
            return Ok(CoordinationStrategy {
                name: "Default".to_string(),
                applicable_scenarios: vec!["default".to_string()],
                selection_criteria: SelectionCriteria {
                    task_complexity: 0.5,
                    time_constraints: false,
                    resource_availability: 0.7,
                    collaboration_requirements: false,
                    risk_tolerance: 0.5,
                },
                execution_algorithm: ExecutionAlgorithm::GreedySelection,
                success_metrics: vec!["completion_time".to_string()],
            });
        }

        // Score strategies based on task requirements
        let mut best_strategy = applicable[0].clone();
        let mut best_score = 0.0;

        for strategy in applicable {
            let score = self.score_strategy(&strategy, task).await?;
            if score > best_score {
                best_score = score;
                best_strategy = strategy;
            }
        }

        Ok(best_strategy)
    }

    async fn score_strategy(&self, strategy: &CoordinationStrategy, task: &TaskRequest) -> Result<f64> {
        let mut score = 0.0;

        // Factor in time constraints
        if let Some(deadline) = task.deadline {
            let time_available = deadline.timestamp() - chrono::Utc::now().timestamp();
            if time_available < 300 && strategy.selection_criteria.time_constraints {
                score += 0.3; // Boost for time-sensitive strategies
            }
        }

        // Factor in collaboration requirements
        if task.requirements.collaboration_preferences.require_consensus {
            if matches!(strategy.execution_algorithm, ExecutionAlgorithm::ConsensusBuilding) {
                score += 0.4;
            }
        }

        // Factor in task complexity (based on required capabilities)
        let complexity = task.requirements.required_capabilities.len() as f64 / 10.0;
        if complexity > strategy.selection_criteria.task_complexity {
            score += 0.3;
        }

        Ok(score.min(1.0))
    }

    async fn create_assignment(&self, task: &TaskRequest, suitable_agents: &[String], strategy: &CoordinationStrategy) -> Result<TaskAssignment> {
        let assignment_strategy = match strategy.execution_algorithm {
            ExecutionAlgorithm::GreedySelection => AssignmentStrategy::SingleAgent,
            ExecutionAlgorithm::ConsensusBuilding => AssignmentStrategy::Consensus,
            ExecutionAlgorithm::OptimalAllocation => AssignmentStrategy::ParallelExecution,
            _ => AssignmentStrategy::SingleAgent,
        };

        let assigned_agents = match assignment_strategy {
            AssignmentStrategy::SingleAgent => {
                vec![suitable_agents[0].clone()]
            },
            AssignmentStrategy::Consensus => {
                let max_agents = task.requirements.collaboration_preferences.maximum_collaborators.min(suitable_agents.len() as u8);
                suitable_agents.iter().take(max_agents as usize).cloned().collect()
            },
            AssignmentStrategy::ParallelExecution => {
                suitable_agents.to_vec()
            },
            _ => vec![suitable_agents[0].clone()],
        };

        let lead_agent = assigned_agents[0].clone();
        
        let expected_completion = if let Some(deadline) = task.deadline {
            deadline
        } else {
            chrono::Utc::now() + chrono::Duration::hours(1) // Default 1 hour
        };

        Ok(TaskAssignment {
            task_id: task.task_id.clone(),
            assigned_agents,
            lead_agent,
            assignment_strategy,
            expected_completion,
            monitoring_interval_ms: 5000, // Monitor every 5 seconds
        })
    }

    async fn notify_agents(&self, assignment: &TaskAssignment) -> Result<()> {
        // In a real implementation, this would send messages to the assigned agents
        for agent_id in &assignment.assigned_agents {
            log::info!("Notifying agent {} of task assignment: {}", agent_id, assignment.task_id);
        }
        Ok(())
    }

    async fn start_task_monitoring(&self, assignment: &TaskAssignment) -> Result<()> {
        let assignment_clone = assignment.clone();
        let coordinator = self.clone();
        
        tokio::spawn(async move {
            coordinator.monitor_task_execution(assignment_clone).await;
        });

        Ok(())
    }

    async fn monitor_task_execution(&self, assignment: TaskAssignment) {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_millis(assignment.monitoring_interval_ms)
        );

        loop {
            interval.tick().await;

            // Check if task is still active
            let assignments = self.active_assignments.read().await;
            if !assignments.contains_key(&assignment.task_id) {
                break; // Task completed or cancelled
            }
            drop(assignments);

            // Monitor agent health and progress
            if let Err(e) = self.check_task_progress(&assignment).await {
                log::error!("Error monitoring task {}: {}", assignment.task_id, e);
            }

            // Check for timeouts
            if chrono::Utc::now() > assignment.expected_completion {
                log::warn!("Task {} exceeded expected completion time", assignment.task_id);
                
                // Trigger intervention if needed
                if let Err(e) = self.handle_task_timeout(&assignment).await {
                    log::error!("Error handling timeout for task {}: {}", assignment.task_id, e);
                }
            }
        }
    }

    async fn check_task_progress(&self, assignment: &TaskAssignment) -> Result<()> {
        let registry = self.agent_registry.read().await;
        
        for agent_id in &assignment.assigned_agents {
            if let Some(profile) = registry.get(agent_id) {
                match &profile.current_status {
                    AgentStatus::Error(error) => {
                        log::error!("Agent {} encountered error in task {}: {}", 
                                  agent_id, assignment.task_id, error);
                        // Handle agent error
                        self.handle_agent_error(assignment, agent_id, error).await?;
                    },
                    AgentStatus::Working(context) => {
                        if context.task_id == assignment.task_id {
                            log::debug!("Agent {} progress on task {}: {:.1}%", 
                                      agent_id, assignment.task_id, context.progress_percentage);
                        }
                    },
                    _ => {}
                }
            }
        }

        Ok(())
    }

    async fn handle_task_timeout(&self, assignment: &TaskAssignment) -> Result<()> {
        log::warn!("Handling timeout for task: {}", assignment.task_id);
        
        // Try to reassign to backup agents or extend deadline
        // For now, just log the timeout
        
        Ok(())
    }

    async fn handle_agent_error(&self, assignment: &TaskAssignment, agent_id: &str, error: &str) -> Result<()> {
        log::error!("Handling agent error for {} in task {}: {}", agent_id, assignment.task_id, error);
        
        // Try to reassign task to another agent
        let registry = self.agent_registry.read().await;
        let other_agents: Vec<String> = assignment.assigned_agents.iter()
            .filter(|id| *id != agent_id)
            .cloned()
            .collect();
        
        if !other_agents.is_empty() {
            log::info!("Reassigning task {} from failed agent {} to remaining agents", 
                      assignment.task_id, agent_id);
        } else {
            log::error!("No backup agents available for task {}", assignment.task_id);
        }
        
        Ok(())
    }

    pub async fn complete_task(&self, task_id: &str, results: Vec<AgentResult>) -> Result<()> {
        // Remove from active assignments
        let mut assignments = self.active_assignments.write().await;
        let assignment = assignments.remove(task_id)
            .ok_or_else(|| anyhow!("Task not found: {}", task_id))?;
        drop(assignments);

        // Calculate performance summary
        let performance_summary = self.calculate_performance_summary(&results).await?;

        // Generate lessons learned
        let lessons_learned = self.extract_lessons_learned(&assignment, &results).await?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&assignment, &results).await?;

        // Create completion result
        let result = CoordinationResult {
            task_id: task_id.to_string(),
            status: TaskStatus::Completed,
            results,
            performance_summary,
            lessons_learned,
            recommendations,
        };

        // Store completed task
        let mut completed = self.completed_tasks.write().await;
        completed.push(result.clone());
        drop(completed);

        // Update learning engine
        self.update_learning_from_completion(&result).await?;

        log::info!("Task {} completed successfully", task_id);
        Ok(())
    }

    async fn calculate_performance_summary(&self, results: &[AgentResult]) -> Result<PerformanceSummary> {
        let total_execution_time = results.iter()
            .map(|r| r.execution_time_ms)
            .max()
            .unwrap_or(0);

        let avg_confidence = results.iter()
            .map(|r| r.confidence_score)
            .sum::<f64>() / results.len() as f64;

        let resource_efficiency = results.iter()
            .map(|r| {
                let total_resources = r.resource_usage.cpu_time_ms + 
                                    r.resource_usage.memory_peak_mb + 
                                    r.resource_usage.gpu_time_ms;
                if total_resources > 0 { 1.0 } else { 0.0 }
            })
            .sum::<f64>() / results.len() as f64;

        Ok(PerformanceSummary {
            total_execution_time_ms: total_execution_time,
            efficiency_score: resource_efficiency,
            accuracy_score: avg_confidence,
            resource_utilization: resource_efficiency,
            collaboration_effectiveness: if results.len() > 1 { 0.8 } else { 1.0 },
        })
    }

    async fn extract_lessons_learned(&self, assignment: &TaskAssignment, results: &[AgentResult]) -> Result<Vec<String>> {
        let mut lessons = Vec::new();

        // Analyze execution patterns
        if results.iter().any(|r| r.execution_time_ms > 30000) {
            lessons.push("Consider breaking down long-running tasks into smaller chunks".to_string());
        }

        if results.iter().any(|r| !r.errors.is_empty()) {
            lessons.push("Implement better error handling and recovery mechanisms".to_string());
        }

        if assignment.assigned_agents.len() > 1 {
            let avg_confidence = results.iter().map(|r| r.confidence_score).sum::<f64>() / results.len() as f64;
            if avg_confidence > 0.9 {
                lessons.push("Multi-agent collaboration improved result quality".to_string());
            }
        }

        Ok(lessons)
    }

    async fn generate_recommendations(&self, assignment: &TaskAssignment, results: &[AgentResult]) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        let avg_execution_time = results.iter().map(|r| r.execution_time_ms).sum::<u64>() / results.len() as u64;
        if avg_execution_time > 10000 {
            recommendations.push("Consider GPU acceleration for compute-intensive tasks".to_string());
        }

        // Resource optimization
        let max_memory_usage = results.iter().map(|r| r.resource_usage.memory_peak_mb).max().unwrap_or(0);
        if max_memory_usage > 1000 {
            recommendations.push("Implement memory-efficient algorithms for large datasets".to_string());
        }

        // Collaboration improvements
        if assignment.assigned_agents.len() == 1 && results[0].confidence_score < 0.8 {
            recommendations.push("Consider multi-agent validation for complex tasks".to_string());
        }

        Ok(recommendations)
    }

    async fn update_learning_from_completion(&self, result: &CoordinationResult) -> Result<()> {
        let mut learning_engine = self.learning_engine.write().await;
        
        // Update performance history
        for agent_result in &result.results {
            let performance = PerformanceMetrics {
                success_rate: if matches!(result.status, TaskStatus::Completed) { 1.0 } else { 0.0 },
                average_latency_ms: agent_result.execution_time_ms,
                throughput_per_second: 1000.0 / agent_result.execution_time_ms as f64,
                accuracy_score: agent_result.confidence_score,
                reliability_score: if agent_result.errors.is_empty() { 1.0 } else { 0.5 },
                last_updated: chrono::Utc::now(),
            };

            learning_engine.performance_history
                .entry(agent_result.agent_id.clone())
                .or_insert_with(Vec::new)
                .push(performance);
        }

        // Extract new patterns
        if result.performance_summary.efficiency_score > 0.9 {
            let pattern = Pattern {
                pattern_type: "high_efficiency_collaboration".to_string(),
                description: format!("Task {} achieved high efficiency with {} agents", 
                                   result.task_id, result.results.len()),
                confidence: result.performance_summary.efficiency_score,
                applicability: vec!["collaboration".to_string(), "efficiency".to_string()],
                discovered_at: chrono::Utc::now(),
            };

            learning_engine.pattern_database
                .entry("efficiency_patterns".to_string())
                .or_insert_with(Vec::new)
                .push(pattern);
        }

        Ok(())
    }
}

impl Clone for AutonomousCoordinator {
    fn clone(&self) -> Self {
        Self {
            agent_registry: Arc::clone(&self.agent_registry),
            task_queue: Arc::clone(&self.task_queue),
            active_assignments: Arc::clone(&self.active_assignments),
            completed_tasks: Arc::clone(&self.completed_tasks),
            learning_engine: Arc::clone(&self.learning_engine),
            performance_tracker: Arc::clone(&self.performance_tracker),
            coordination_strategies: Arc::clone(&self.coordination_strategies),
        }
    }
}

#[async_trait]
impl Actor for AutonomousCoordinator {
    async fn init(&mut self, _budget: Budget) -> Result<()> {
        log::info!("Autonomous Coordinator initialized");
        Ok(())
    }

    async fn handle(&mut self, message: Message, _budget: Budget) -> Result<()> {
        match message.message_type.as_str() {
            "register_agent" => {
                if let Ok(profile) = serde_json::from_value::<AgentProfile>(message.payload) {
                    self.register_agent(profile).await?;
                }
            },
            "submit_task" => {
                if let Ok(task) = serde_json::from_value::<TaskRequest>(message.payload) {
                    self.submit_task(task).await?;
                }
            },
            "task_completed" => {
                if let Ok(completion_data) = serde_json::from_value::<serde_json::Value>(message.payload) {
                    if let (Some(task_id), Some(results)) = (
                        completion_data.get("task_id").and_then(|v| v.as_str()),
                        completion_data.get("results").and_then(|v| serde_json::from_value::<Vec<AgentResult>>(v.clone()).ok())
                    ) {
                        self.complete_task(task_id, results).await?;
                    }
                }
            },
            _ => {}
        }
        Ok(())
    }

    async fn tick(&mut self, _budget: Budget) -> Result<()> {
        // Process pending tasks
        self.process_task_queue().await?;
        
        // Update performance metrics
        // This would normally update real-time performance tracking
        
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        log::info!("Autonomous Coordinator shutting down");
        Ok(())
    }
}