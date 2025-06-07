
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::messages::{Message, TaskSpec, Priority};

#[derive(Debug, Error)]
pub enum CollaborationError {
    #[error("Agent not found: {0}")]
    AgentNotFound(String),
    #[error("Circular dependency detected in task: {0}")]
    CircularDependency(String),
    #[error("Task assignment failed: {0}")]
    AssignmentFailed(String),
    #[error("Communication channel closed")]
    ChannelClosed,
    #[error("Collaboration timeout")]
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapability {
    pub name: String,
    pub version: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub cost_per_use: f32,
    pub avg_duration_ms: u64,
    pub success_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub agent_id: String,
    pub agent_type: String,
    pub capabilities: Vec<AgentCapability>,
    pub current_load: f32,
    pub max_concurrent_tasks: u32,
    pub reliability_score: f32,
    pub last_seen: DateTime<Utc>,
    pub status: AgentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Available,
    Busy,
    Overloaded,
    Maintenance,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationRequest {
    pub request_id: String,
    pub from_agent: String,
    pub to_agent: Option<String>, // None for broadcast
    pub task_spec: TaskSpec,
    pub priority: Priority,
    pub deadline: Option<DateTime<Utc>>,
    pub context: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationResponse {
    pub request_id: String,
    pub agent_id: String,
    pub accepted: bool,
    pub estimated_duration: Option<u64>,
    pub cost_estimate: Option<f32>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAssignment {
    pub assignment_id: String,
    pub task_spec: TaskSpec,
    pub assigned_to: String,
    pub assigned_by: String,
    pub assigned_at: DateTime<Utc>,
    pub deadline: Option<DateTime<Utc>>,
    pub status: TaskStatus,
    pub dependencies: Vec<String>,
    pub dependents: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Assigned,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

pub struct CollaborationCoordinator {
    agents: Arc<RwLock<HashMap<String, AgentProfile>>>,
    active_tasks: Arc<RwLock<HashMap<String, TaskAssignment>>>,
    pending_requests: Arc<RwLock<HashMap<String, CollaborationRequest>>>,
    task_dependencies: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    message_channels: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<Message>>>>,
    collaboration_history: Arc<RwLock<VecDeque<CollaborationEvent>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationEvent {
    pub event_id: String,
    pub event_type: CollaborationEventType,
    pub participants: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationEventType {
    TaskRequest,
    TaskAssignment,
    TaskCompletion,
    AgentJoined,
    AgentLeft,
    CollaborationStarted,
    CollaborationEnded,
}

impl CollaborationCoordinator {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            task_dependencies: Arc::new(RwLock::new(HashMap::new())),
            message_channels: Arc::new(RwLock::new(HashMap::new())),
            collaboration_history: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    pub async fn register_agent(
        &self,
        profile: AgentProfile,
        channel: mpsc::UnboundedSender<Message>,
    ) -> Result<(), CollaborationError> {
        let mut agents = self.agents.write().await;
        let mut channels = self.message_channels.write().await;

        agents.insert(profile.agent_id.clone(), profile.clone());
        channels.insert(profile.agent_id.clone(), channel);

        // Record event
        self.record_event(CollaborationEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CollaborationEventType::AgentJoined,
            participants: vec![profile.agent_id],
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }).await;

        Ok(())
    }

    pub async fn unregister_agent(&self, agent_id: &str) -> Result<(), CollaborationError> {
        let mut agents = self.agents.write().await;
        let mut channels = self.message_channels.write().await;

        agents.remove(agent_id);
        channels.remove(agent_id);

        // Cancel any tasks assigned to this agent
        self.cancel_agent_tasks(agent_id).await?;

        // Record event
        self.record_event(CollaborationEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CollaborationEventType::AgentLeft,
            participants: vec![agent_id.to_string()],
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }).await;

        Ok(())
    }

    pub async fn request_collaboration(
        &self,
        request: CollaborationRequest,
    ) -> Result<Vec<CollaborationResponse>, CollaborationError> {
        let agents = self.agents.read().await;
        let mut pending = self.pending_requests.write().await;

        pending.insert(request.request_id.clone(), request.clone());

        let mut responses = Vec::new();

        match &request.to_agent {
            Some(target_agent) => {
                // Direct request to specific agent
                if let Some(agent) = agents.get(target_agent) {
                    let response = self.evaluate_request(&request, agent).await;
                    responses.push(response);
                } else {
                    return Err(CollaborationError::AgentNotFound(target_agent.clone()));
                }
            }
            None => {
                // Broadcast request to all capable agents
                for (agent_id, agent) in agents.iter() {
                    if agent_id != &request.from_agent && self.can_handle_task(agent, &request.task_spec).await {
                        let response = self.evaluate_request(&request, agent).await;
                        responses.push(response);
                    }
                }
            }
        }

        // Record event
        self.record_event(CollaborationEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CollaborationEventType::TaskRequest,
            participants: vec![request.from_agent.clone()],
            timestamp: Utc::now(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("task_type".to_string(), serde_json::Value::String(request.task_spec.task_type.clone()));
                meta
            },
        }).await;

        Ok(responses)
    }

    async fn evaluate_request(&self, request: &CollaborationRequest, agent: &AgentProfile) -> CollaborationResponse {
        let can_handle = self.can_handle_task(agent, &request.task_spec).await;
        let has_capacity = agent.current_load < 0.8; // 80% capacity threshold
        let is_available = matches!(agent.status, AgentStatus::Available);

        let accepted = can_handle && has_capacity && is_available;

        let estimated_duration = if accepted {
            agent.capabilities
                .iter()
                .find(|cap| cap.name == request.task_spec.task_type)
                .map(|cap| cap.avg_duration_ms)
        } else {
            None
        };

        let cost_estimate = if accepted {
            agent.capabilities
                .iter()
                .find(|cap| cap.name == request.task_spec.task_type)
                .map(|cap| cap.cost_per_use)
        } else {
            None
        };

        let reason = if !accepted {
            if !can_handle {
                Some("Cannot handle this task type".to_string())
            } else if !has_capacity {
                Some("At capacity".to_string())
            } else {
                Some("Not available".to_string())
            }
        } else {
            None
        };

        CollaborationResponse {
            request_id: request.request_id.clone(),
            agent_id: agent.agent_id.clone(),
            accepted,
            estimated_duration,
            cost_estimate,
            reason,
        }
    }

    pub async fn assign_task(
        &self,
        request_id: &str,
        selected_agent: &str,
    ) -> Result<TaskAssignment, CollaborationError> {
        let mut pending = self.pending_requests.write().await;
        let mut active_tasks = self.active_tasks.write().await;
        let channels = self.message_channels.read().await;

        let request = pending.remove(request_id)
            .ok_or_else(|| CollaborationError::AssignmentFailed("Request not found".to_string()))?;

        let channel = channels.get(selected_agent)
            .ok_or_else(|| CollaborationError::AgentNotFound(selected_agent.to_string()))?;

        // Check for circular dependencies
        self.check_circular_dependencies(&request.task_spec).await?;

        let assignment = TaskAssignment {
            assignment_id: Uuid::new_v4().to_string(),
            task_spec: request.task_spec.clone(),
            assigned_to: selected_agent.to_string(),
            assigned_by: request.from_agent.clone(),
            assigned_at: Utc::now(),
            deadline: request.deadline,
            status: TaskStatus::Assigned,
            dependencies: request.task_spec.dependencies.clone(),
            dependents: Vec::new(),
        };

        // Send task assignment message
        let task_message = Message::TaskAssignment {
            task_id: assignment.assignment_id.clone(),
            spec: request.task_spec,
            priority: request.priority,
        };

        channel.send(task_message)
            .map_err(|_| CollaborationError::ChannelClosed)?;

        active_tasks.insert(assignment.assignment_id.clone(), assignment.clone());

        // Update dependencies
        self.update_task_dependencies(&assignment).await;

        // Record event
        self.record_event(CollaborationEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CollaborationEventType::TaskAssignment,
            participants: vec![assignment.assigned_by.clone(), assignment.assigned_to.clone()],
            timestamp: Utc::now(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("task_id".to_string(), serde_json::Value::String(assignment.assignment_id.clone()));
                meta
            },
        }).await;

        Ok(assignment)
    }

    async fn can_handle_task(&self, agent: &AgentProfile, task_spec: &TaskSpec) -> bool {
        agent.capabilities
            .iter()
            .any(|cap| cap.name == task_spec.task_type)
    }

    async fn check_circular_dependencies(&self, task_spec: &TaskSpec) -> Result<(), CollaborationError> {
        let dependencies = self.task_dependencies.read().await;
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        fn has_cycle(
            task_id: &str,
            dependencies: &HashMap<String, HashSet<String>>,
            visited: &mut HashSet<String>,
            rec_stack: &mut HashSet<String>,
        ) -> bool {
            visited.insert(task_id.to_string());
            rec_stack.insert(task_id.to_string());

            if let Some(deps) = dependencies.get(task_id) {
                for dep in deps {
                    if !visited.contains(dep) {
                        if has_cycle(dep, dependencies, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack.contains(dep) {
                        return true;
                    }
                }
            }

            rec_stack.remove(task_id);
            false
        }

        for dep in &task_spec.dependencies {
            if has_cycle(dep, &dependencies, &mut visited, &mut rec_stack) {
                return Err(CollaborationError::CircularDependency(task_spec.id.clone()));
            }
        }

        Ok(())
    }

    async fn update_task_dependencies(&self, assignment: &TaskAssignment) {
        let mut dependencies = self.task_dependencies.write().await;
        
        if !assignment.dependencies.is_empty() {
            dependencies.insert(
                assignment.assignment_id.clone(),
                assignment.dependencies.iter().cloned().collect(),
            );
        }

        // Update dependents for dependency tasks
        for dep in &assignment.dependencies {
            if let Some(dep_dependents) = dependencies.get_mut(dep) {
                dep_dependents.insert(assignment.assignment_id.clone());
            }
        }
    }

    pub async fn complete_task(&self, task_id: &str, success: bool) -> Result<(), CollaborationError> {
        let mut active_tasks = self.active_tasks.write().await;
        
        if let Some(task) = active_tasks.get_mut(task_id) {
            task.status = if success { TaskStatus::Completed } else { TaskStatus::Failed };

            // Record event
            self.record_event(CollaborationEvent {
                event_id: Uuid::new_v4().to_string(),
                event_type: CollaborationEventType::TaskCompletion,
                participants: vec![task.assigned_to.clone(), task.assigned_by.clone()],
                timestamp: Utc::now(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("task_id".to_string(), serde_json::Value::String(task_id.to_string()));
                    meta.insert("success".to_string(), serde_json::Value::Bool(success));
                    meta
                },
            }).await;

            // Check if dependent tasks can now be started
            self.check_ready_dependents(task_id).await;
        }

        Ok(())
    }

    async fn check_ready_dependents(&self, completed_task_id: &str) {
        let dependencies = self.task_dependencies.read().await;
        let active_tasks = self.active_tasks.read().await;

        // Find tasks that depend on the completed task
        for (task_id, deps) in dependencies.iter() {
            if deps.contains(completed_task_id) {
                // Check if all dependencies are completed
                let all_completed = deps.iter().all(|dep_id| {
                    active_tasks.get(dep_id)
                        .map(|task| matches!(task.status, TaskStatus::Completed))
                        .unwrap_or(false)
                });

                if all_completed {
                    // Task is ready to be executed
                    // In a real implementation, this would trigger the task execution
                    println!("Task {} is ready to execute", task_id);
                }
            }
        }
    }

    async fn cancel_agent_tasks(&self, agent_id: &str) -> Result<(), CollaborationError> {
        let mut active_tasks = self.active_tasks.write().await;
        
        for task in active_tasks.values_mut() {
            if task.assigned_to == agent_id && !matches!(task.status, TaskStatus::Completed | TaskStatus::Failed) {
                task.status = TaskStatus::Cancelled;
            }
        }

        Ok(())
    }

    pub async fn get_agent_load(&self, agent_id: &str) -> Option<f32> {
        let agents = self.agents.read().await;
        agents.get(agent_id).map(|agent| agent.current_load)
    }

    pub async fn update_agent_status(&self, agent_id: &str, status: AgentStatus) {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.get_mut(agent_id) {
            agent.status = status;
            agent.last_seen = Utc::now();
        }
    }

    pub async fn get_collaboration_metrics(&self) -> CollaborationMetrics {
        let agents = self.agents.read().await;
        let active_tasks = self.active_tasks.read().await;
        let history = self.collaboration_history.read().await;

        let total_agents = agents.len();
        let available_agents = agents.values()
            .filter(|agent| matches!(agent.status, AgentStatus::Available))
            .count();

        let total_tasks = active_tasks.len();
        let completed_tasks = active_tasks.values()
            .filter(|task| matches!(task.status, TaskStatus::Completed))
            .count();

        let avg_task_duration = if completed_tasks > 0 {
            active_tasks.values()
                .filter(|task| matches!(task.status, TaskStatus::Completed))
                .map(|task| Utc::now().signed_duration_since(task.assigned_at).num_milliseconds() as f64)
                .sum::<f64>() / completed_tasks as f64
        } else {
            0.0
        };

        CollaborationMetrics {
            total_agents,
            available_agents,
            total_tasks,
            completed_tasks,
            avg_task_duration_ms: avg_task_duration as u64,
            collaboration_events: history.len(),
        }
    }

    async fn record_event(&self, event: CollaborationEvent) {
        let mut history = self.collaboration_history.write().await;
        history.push_back(event);

        // Keep only last 1000 events
        if history.len() > 1000 {
            history.pop_front();
        }
    }

    pub async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        let active_tasks = self.active_tasks.read().await;
        active_tasks.get(task_id).map(|task| task.status.clone())
    }

    pub async fn get_agent_tasks(&self, agent_id: &str) -> Vec<TaskAssignment> {
        let active_tasks = self.active_tasks.read().await;
        active_tasks.values()
            .filter(|task| task.assigned_to == agent_id)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CollaborationMetrics {
    pub total_agents: usize,
    pub available_agents: usize,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub avg_task_duration_ms: u64,
    pub collaboration_events: usize,
}

impl Default for CollaborationCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
