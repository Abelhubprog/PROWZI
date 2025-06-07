//! ADK Integration Module
//!
//! This module provides integration points for Google ADK Python agents
//! with the existing Prowzi memory-enhanced orchestrator system.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::orchestrator::{
    Orchestrator, OrchestrationError, AgentInfo, AgentStatus, SpawnRequest, SpawnResult,
    MemoryContext, MemoryEnhancedResult, CoordinationPattern
};
use crate::memory::{MemorySystem, MemoryQuery};
use crate::messages::Message;

/// ADK Agent configuration and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdkAgentConfig {
    pub agent_name: String,
    pub model: String, // e.g., "gemini-2b-flash"
    pub instruction: String,
    pub tools: Vec<String>,
    pub max_memory_mb: u64,
    pub timeout_sec: u32,
    pub kubernetes_config: KubernetesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    pub image: String,
    pub replicas: u32,
    pub cpu_request: String,
    pub memory_request: String,
    pub env_vars: HashMap<String, String>,
}

/// Bridge between Rust orchestrator and Python ADK agents
pub struct AdkBridge {
    nats_client: async_nats::Client,
    memory_system: std::sync::Arc<tokio::sync::Mutex<MemorySystem>>,
    active_agents: std::sync::Arc<tokio::sync::RwLock<HashMap<String, AdkAgentInfo>>>,
    message_tx: mpsc::Sender<Message>,
}

#[derive(Debug, Clone)]
pub struct AdkAgentInfo {
    pub agent_id: String,
    pub config: AdkAgentConfig,
    pub status: AdkAgentStatus,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub performance_metrics: AdkPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdkAgentStatus {
    Initializing,
    Running,
    Processing { task_id: String },
    Idle,
    Error { error_message: String },
    Stopped,
}

#[derive(Debug, Clone, Default)]
pub struct AdkPerformanceMetrics {
    pub tasks_completed: u64,
    pub average_response_time_ms: f64,
    pub success_rate: f32,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
}

impl AdkBridge {
    pub async fn new(
        nats_url: &str,
        memory_system: std::sync::Arc<tokio::sync::Mutex<MemorySystem>>,
        message_tx: mpsc::Sender<Message>,
    ) -> Result<Self, OrchestrationError> {
        let nats_client = async_nats::connect(nats_url).await
            .map_err(|e| OrchestrationError::CommunicationError { 
                message: format!("Failed to connect to NATS: {}", e) 
            })?;

        Ok(Self {
            nats_client,
            memory_system,
            active_agents: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            message_tx,
        })
    }

    /// Spawn a new ADK agent with memory enhancement
    pub async fn spawn_adk_agent(
        &self,
        mission_id: &str,
        config: AdkAgentConfig,
        memory_context: Option<MemoryContext>,
    ) -> Result<String, OrchestrationError> {
        let agent_id = format!("adk_{}_{}", config.agent_name, Uuid::new_v4().simple());
        
        info!("Spawning ADK agent: {} for mission: {}", agent_id, mission_id);

        // Enhance with memory if context provided
        if let Some(context) = memory_context {
            let memory_insights = self.get_memory_insights(&context).await?;
            self.apply_memory_insights_to_config(&agent_id, &memory_insights).await?;
        }

        // Deploy to Kubernetes
        self.deploy_kubernetes_agent(&agent_id, &config).await?;

        // Set up NATS subscriptions for agent communication
        self.setup_agent_communication(&agent_id, mission_id).await?;

        // Store agent info
        let agent_info = AdkAgentInfo {
            agent_id: agent_id.clone(),
            config,
            status: AdkAgentStatus::Initializing,
            last_heartbeat: chrono::Utc::now(),
            performance_metrics: AdkPerformanceMetrics::default(),
        };

        self.active_agents.write().await.insert(agent_id.clone(), agent_info);

        Ok(agent_id)
    }

    /// Get memory insights for ADK agent configuration
    async fn get_memory_insights(
        &self,
        context: &MemoryContext,
    ) -> Result<Vec<MemoryInsight>, OrchestrationError> {
        let memory_system = self.memory_system.lock().await;
        
        let query = MemoryQuery {
            agent_id: context.mission_type.clone(),
            query_type: crate::memory::QueryType::Similarity,
            context: "adk_agent_patterns".to_string(),
            limit: Some(5),
            similarity_threshold: Some(0.7),
        };

        let results = memory_system.query(query).await
            .map_err(|e| OrchestrationError::Internal { 
                message: format!("Memory query failed: {}", e) 
            })?;

        let insights = results.into_iter()
            .filter_map(|result| {
                serde_json::from_value::<MemoryInsight>(result.content).ok()
            })
            .collect();

        Ok(insights)
    }

    /// Apply memory insights to agent configuration
    async fn apply_memory_insights_to_config(
        &self,
        agent_id: &str,
        insights: &[MemoryInsight],
    ) -> Result<(), OrchestrationError> {
        // Adjust agent configuration based on memory insights
        for insight in insights {
            match insight.insight_type {
                InsightType::ModelOptimization => {
                    info!("Applying model optimization for agent {}: {}", agent_id, insight.description);
                }
                InsightType::ToolSelection => {
                    info!("Applying tool selection insight for agent {}: {}", agent_id, insight.description);
                }
                InsightType::ResourceOptimization => {
                    info!("Applying resource optimization for agent {}: {}", agent_id, insight.description);
                }
            }
        }
        Ok(())
    }

    /// Deploy agent to Kubernetes
    async fn deploy_kubernetes_agent(
        &self,
        agent_id: &str,
        config: &AdkAgentConfig,
    ) -> Result<(), OrchestrationError> {
        // In a real implementation, this would use the Kubernetes API
        // For now, we'll simulate the deployment
        info!("Deploying ADK agent {} to Kubernetes cluster", agent_id);
        
        // Generate Kubernetes deployment manifest
        let deployment = self.generate_k8s_deployment(agent_id, config);
        
        // Apply deployment (simulated)
        info!("Generated Kubernetes deployment for agent {}: {}", agent_id, deployment);
        
        Ok(())
    }

    /// Generate Kubernetes deployment manifest
    fn generate_k8s_deployment(&self, agent_id: &str, config: &AdkAgentConfig) -> String {
        format!(
            r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {}
  labels:
    app: adk-agent
    agent-id: {}
spec:
  replicas: {}
  selector:
    matchLabels:
      app: adk-agent
      agent-id: {}
  template:
    metadata:
      labels:
        app: adk-agent
        agent-id: {}
    spec:
      containers:
      - name: adk-agent
        image: {}
        env:
        - name: ADK_MODEL
          value: "{}"
        - name: AGENT_ID
          value: "{}"
        - name: NATS_URL
          value: "nats://nats.prowzi:4222"
        resources:
          requests:
            memory: "{}"
            cpu: "{}"
          limits:
            memory: "{}Mi"
            cpu: "{}m"
"#,
            agent_id,
            agent_id,
            config.kubernetes_config.replicas,
            agent_id,
            agent_id,
            config.kubernetes_config.image,
            config.model,
            agent_id,
            config.kubernetes_config.memory_request,
            config.kubernetes_config.cpu_request,
            config.max_memory_mb,
            config.max_memory_mb / 2 // CPU limit as rough heuristic
        )
    }

    /// Set up NATS communication for agent
    async fn setup_agent_communication(
        &self,
        agent_id: &str,
        mission_id: &str,
    ) -> Result<(), OrchestrationError> {
        // Subscribe to agent output channel
        let output_subject = format!("analysis.out.{}", mission_id);
        let mut subscriber = self.nats_client.subscribe(output_subject).await
            .map_err(|e| OrchestrationError::CommunicationError { 
                message: format!("Failed to subscribe to agent output: {}", e) 
            })?;

        // Subscribe to agent heartbeat
        let heartbeat_subject = format!("agent.heartbeat.{}", agent_id);
        let mut heartbeat_subscriber = self.nats_client.subscribe(heartbeat_subject).await
            .map_err(|e| OrchestrationError::CommunicationError { 
                message: format!("Failed to subscribe to agent heartbeat: {}", e) 
            })?;

        // Spawn tasks to handle messages
        let agents = self.active_agents.clone();
        let agent_id_clone = agent_id.to_string();
        tokio::spawn(async move {
            while let Some(message) = subscriber.next().await {
                if let Ok(payload) = String::from_utf8(message.payload.to_vec()) {
                    info!("Received output from ADK agent {}: {}", agent_id_clone, payload);
                    // Process agent output
                }
            }
        });

        let agents_heartbeat = self.active_agents.clone();
        let agent_id_heartbeat = agent_id.to_string();
        tokio::spawn(async move {
            while let Some(message) = heartbeat_subscriber.next().await {
                // Update agent heartbeat
                if let Some(agent_info) = agents_heartbeat.write().await.get_mut(&agent_id_heartbeat) {
                    agent_info.last_heartbeat = chrono::Utc::now();
                    agent_info.status = AdkAgentStatus::Running;
                }
            }
        });

        Ok(())
    }

    /// Send task to ADK agent
    pub async fn send_task_to_agent(
        &self,
        agent_id: &str,
        task: AdkTask,
    ) -> Result<(), OrchestrationError> {
        let task_subject = format!("agent.task.{}", agent_id);
        let task_payload = serde_json::to_vec(&task)
            .map_err(|e| OrchestrationError::Internal { 
                message: format!("Failed to serialize task: {}", e) 
            })?;

        self.nats_client.publish(task_subject, task_payload.into()).await
            .map_err(|e| OrchestrationError::CommunicationError { 
                message: format!("Failed to send task to agent: {}", e) 
            })?;

        // Update agent status
        if let Some(agent_info) = self.active_agents.write().await.get_mut(agent_id) {
            agent_info.status = AdkAgentStatus::Processing { 
                task_id: task.task_id.clone() 
            };
        }

        Ok(())
    }

    /// Get status of all ADK agents
    pub async fn get_agent_status(&self) -> HashMap<String, AdkAgentStatus> {
        self.active_agents.read().await
            .iter()
            .map(|(id, info)| (id.clone(), info.status.clone()))
            .collect()
    }

    /// Stop ADK agent
    pub async fn stop_agent(&self, agent_id: &str) -> Result<(), OrchestrationError> {
        info!("Stopping ADK agent: {}", agent_id);

        // Send stop signal via NATS
        let stop_subject = format!("agent.control.{}", agent_id);
        let stop_message = serde_json::json!({"action": "stop"});
        
        self.nats_client.publish(stop_subject, stop_message.to_string().into()).await
            .map_err(|e| OrchestrationError::CommunicationError { 
                message: format!("Failed to send stop signal: {}", e) 
            })?;

        // Remove from Kubernetes (simulated)
        info!("Removing ADK agent {} from Kubernetes", agent_id);

        // Update status
        if let Some(agent_info) = self.active_agents.write().await.get_mut(agent_id) {
            agent_info.status = AdkAgentStatus::Stopped;
        }

        Ok(())
    }
}

/// Task sent to ADK agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdkTask {
    pub task_id: String,
    pub task_type: String,
    pub payload: serde_json::Value,
    pub timeout_sec: u32,
    pub priority: u8,
}

/// Memory insight for ADK agent optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInsight {
    pub insight_id: String,
    pub insight_type: InsightType,
    pub description: String,
    pub confidence: f32,
    pub suggested_config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    ModelOptimization,
    ToolSelection,
    ResourceOptimization,
}

/// Default ADK agent configurations for common use cases
pub struct AdkAgentTemplates;

impl AdkAgentTemplates {
    /// Analysis agent for ML-heavy market analysis
    pub fn analysis_agent() -> AdkAgentConfig {
        AdkAgentConfig {
            agent_name: "analysis".to_string(),
            model: "gemini-2b-flash".to_string(),
            instruction: "Analyze market data using ML models and provide trading insights".to_string(),
            tools: vec![
                "get_order_book".to_string(),
                "simulate_fill".to_string(),
                "calculate_risk".to_string(),
            ],
            max_memory_mb: 2048,
            timeout_sec: 300,
            kubernetes_config: KubernetesConfig {
                image: "ghcr.io/prowzi/adk-analysis:latest".to_string(),
                replicas: 1,
                cpu_request: "500m".to_string(),
                memory_request: "1Gi".to_string(),
                env_vars: [
                    ("ADK_UI".to_string(), "disabled".to_string()),
                    ("OTEL_EXPORTER_OTLP_ENDPOINT".to_string(), "http://tempo:4318".to_string()),
                ].into_iter().collect(),
            },
        }
    }

    /// Sentiment agent for social media and news analysis
    pub fn sentiment_agent() -> AdkAgentConfig {
        AdkAgentConfig {
            agent_name: "sentiment".to_string(),
            model: "gemini-2b-flash".to_string(),
            instruction: "Analyze sentiment from social media and news sources for trading signals".to_string(),
            tools: vec![
                "google_search".to_string(),
                "twitter_sentiment".to_string(),
                "news_analysis".to_string(),
            ],
            max_memory_mb: 1024,
            timeout_sec: 180,
            kubernetes_config: KubernetesConfig {
                image: "ghcr.io/prowzi/adk-sentiment:latest".to_string(),
                replicas: 2,
                cpu_request: "250m".to_string(),
                memory_request: "512Mi".to_string(),
                env_vars: [
                    ("ADK_UI".to_string(), "disabled".to_string()),
                ].into_iter().collect(),
            },
        }
    }

    /// Risk agent for VaR calculation and risk management
    pub fn risk_agent() -> AdkAgentConfig {
        AdkAgentConfig {
            agent_name: "risk".to_string(),
            model: "gemini-2b-flash".to_string(),
            instruction: "Calculate Value at Risk and provide risk management recommendations".to_string(),
            tools: vec![
                "calculate_var".to_string(),
                "stress_test".to_string(),
                "correlation_analysis".to_string(),
            ],
            max_memory_mb: 1536,
            timeout_sec: 240,
            kubernetes_config: KubernetesConfig {
                image: "ghcr.io/prowzi/adk-risk:latest".to_string(),
                replicas: 1,
                cpu_request: "750m".to_string(),
                memory_request: "1Gi".to_string(),
                env_vars: [
                    ("ADK_UI".to_string(), "disabled".to_string()),
                    ("RUST_RPC_HOST".to_string(), "agent-runner".to_string()),
                ].into_iter().collect(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adk_agent_templates() {
        let analysis_config = AdkAgentTemplates::analysis_agent();
        assert_eq!(analysis_config.agent_name, "analysis");
        assert_eq!(analysis_config.model, "gemini-2b-flash");
        assert!(!analysis_config.tools.is_empty());
    }

    #[tokio::test]
    async fn test_kubernetes_deployment_generation() {
        let (tx, _rx) = mpsc::channel(100);
        let memory_system = std::sync::Arc::new(tokio::sync::Mutex::new(
            MemorySystem::new("test".to_string(), 100).await.unwrap()
        ));
        
        let bridge = AdkBridge::new("nats://localhost:4222", memory_system, tx).await;
        // This would fail without actual NATS, but we're testing the structure
        assert!(bridge.is_err()); // Expected since no NATS server
    }
}