
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use crate::{SpawnRequest, Mission, MissionStatus};

#[derive(Debug, Clone, Serialize)]
pub struct SpawnResult {
    pub agent_ids: Vec<String>,
    pub status: String,
    pub spawned_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub id: String,
    pub agent_type: String,
    pub mission_id: String,
    pub status: AgentStatus,
    pub spawned_at: DateTime<Utc>,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub resource_usage: AgentResourceUsage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Starting,
    Running,
    Paused,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, Default)]
pub struct AgentResourceUsage {
    pub cpu_percent: f32,
    pub memory_mb: u64,
    pub tokens_used: u64,
}

pub struct MissionScheduler {
    agents: Arc<RwLock<HashMap<String, AgentInfo>>>,
    task_queue: Arc<RwLock<Vec<TaskSpec>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub id: String,
    pub mission_id: String,
    pub agent_type: String,
    pub priority: u8,
    pub resource_requirements: crate::ResourceRequirements,
}

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("Agent spawn failed: {0}")]
    SpawnFailed(String),
    
    #[error("Mission not found: {0}")]
    MissionNotFound(String),
    
    #[error("Resource allocation failed")]
    ResourceAllocation,
}

impl MissionScheduler {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn schedule_spawn(
        &self,
        mission_id: &str,
        request: SpawnRequest,
    ) -> Result<SpawnResult, SchedulerError> {
        let mut agent_ids = Vec::new();
        
        for i in 0..request.count {
            let agent_id = format!("{}-{}-{}", request.agent_type, mission_id, i);
            
            let agent_info = AgentInfo {
                id: agent_id.clone(),
                agent_type: request.agent_type.clone(),
                mission_id: mission_id.to_string(),
                status: AgentStatus::Starting,
                spawned_at: Utc::now(),
                last_heartbeat: None,
                resource_usage: AgentResourceUsage::default(),
            };
            
            // Store agent info
            {
                let mut agents = self.agents.write().await;
                agents.insert(agent_id.clone(), agent_info);
            }
            
            // Simulate agent spawn (in real implementation, this would spawn actual processes)
            tokio::spawn(Self::simulate_agent_lifecycle(
                agent_id.clone(),
                self.agents.clone(),
            ));
            
            agent_ids.push(agent_id);
        }
        
        Ok(SpawnResult {
            agent_ids,
            status: "spawning".to_string(),
            spawned_at: Utc::now(),
        })
    }
    
    pub async fn throttle_mission(
        &self,
        mission_id: &str,
        throttle_factor: f32,
    ) -> Result<(), SchedulerError> {
        let mut agents = self.agents.write().await;
        
        for (_, agent) in agents.iter_mut() {
            if agent.mission_id == mission_id && agent.status == AgentStatus::Running {
                // Apply throttling by pausing some agents
                if throttle_factor < 0.5 {
                    agent.status = AgentStatus::Paused;
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn check_agent_health(&self) {
        let mut agents = self.agents.write().await;
        let now = Utc::now();
        
        for (agent_id, agent) in agents.iter_mut() {
            if let Some(last_heartbeat) = agent.last_heartbeat {
                let elapsed = now.signed_duration_since(last_heartbeat);
                
                // Mark as failed if no heartbeat for 2 minutes
                if elapsed.num_seconds() > 120 && agent.status == AgentStatus::Running {
                    tracing::warn!("Agent {} missed heartbeat, marking as failed", agent_id);
                    agent.status = AgentStatus::Failed;
                }
            }
        }
    }
    
    pub async fn get_mission_agents(&self, mission_id: &str) -> Vec<AgentInfo> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|agent| agent.mission_id == mission_id)
            .cloned()
            .collect()
    }
    
    async fn simulate_agent_lifecycle(agent_id: String, agents: Arc<RwLock<HashMap<String, AgentInfo>>>) {
        // Simulate startup delay
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        // Mark as running
        {
            let mut agents_lock = agents.write().await;
            if let Some(agent) = agents_lock.get_mut(&agent_id) {
                agent.status = AgentStatus::Running;
                agent.last_heartbeat = Some(Utc::now());
            }
        }
        
        // Simulate periodic heartbeats
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            
            let mut agents_lock = agents.write().await;
            if let Some(agent) = agents_lock.get_mut(&agent_id) {
                if agent.status == AgentStatus::Running {
                    agent.last_heartbeat = Some(Utc::now());
                    // Simulate some resource usage
                    agent.resource_usage.cpu_percent = 25.0 + (rand::random::<f32>() * 50.0);
                    agent.resource_usage.memory_mb = 256 + (rand::random::<u64>() % 512);
                    agent.resource_usage.tokens_used += 10;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
}
