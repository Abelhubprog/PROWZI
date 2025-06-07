use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::v1::metrics::PodMetrics;
use kube::{Api, Client};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct QuotaEnforcer {
    k8s_client: Client,
    mission_budgets: Arc<RwLock<HashMap<String, ResourceBudget>>>,
    violation_tracker: Arc<RwLock<HashMap<String, ViolationRecord>>>,
}

#[derive(Debug, Clone)]
struct ResourceBudget {
    cpu_millis: i64,
    memory_mb: i64,
    gpu_minutes: Option<i64>,
}

#[derive(Debug)]
struct ViolationRecord {
    agent_id: String,
    first_violation: Instant,
    violation_count: u32,
    resource_type: String,
}

impl QuotaEnforcer {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let client = Client::try_default().await?;

        Ok(Self {
            k8s_client: client,
            mission_budgets: Arc::new(RwLock::new(HashMap::new())),
            violation_tracker: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn enforce_quotas(&self) -> Result<(), Box<dyn std::error::Error>> {
        let pods_api: Api<Pod> = Api::namespaced(self.k8s_client.clone(), "prowzi");
        let metrics_api: Api<PodMetrics> = Api::namespaced(self.k8s_client.clone(), "prowzi");

        let pods = pods_api.list(&Default::default()).await?;

        for pod in pods.items {
            let pod_name = pod.metadata.name.clone().unwrap_or_default();

            // Get pod metrics
            if let Ok(metrics) = metrics_api.get(&pod_name).await {
                self.check_pod_quota(&pod, &metrics).await?;
            }
        }

        Ok(())
    }

    async fn check_pod_quota(
        &self,
        pod: &Pod,
        metrics: &PodMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let labels = &pod.metadata.labels;
        let agent_id = labels.as_ref()
            .and_then(|l| l.get("agent-id"))
            .cloned()
            .unwrap_or_default();

        let mission_id = labels.as_ref()
            .and_then(|l| l.get("mission-id"))
            .cloned()
            .unwrap_or_default();

        let budgets = self.mission_budgets.read().await;
        if let Some(budget) = budgets.get(&mission_id) {
            for container in &metrics.containers {
                let cpu_usage = parse_cpu(&container.usage.cpu);
                let memory_usage = parse_memory(&container.usage.memory);

                // Check CPU (with 2x grace)
                if cpu_usage > budget.cpu_millis * 2 {
                    self.handle_violation(
                        &agent_id,
                        "cpu",
                        cpu_usage as f64 / budget.cpu_millis as f64,
                    ).await?;
                }

                // Check memory (strict)
                if memory_usage > budget.memory_mb {
                    self.handle_violation(
                        &agent_id,
                        "memory",
                        memory_usage as f64 / budget.memory_mb as f64,
                    ).await?;
                }

                // Check GPU if applicable
                if let Some(gpu_budget) = budget.gpu_minutes {
                    if let Some(gpu_usage) = self.get_gpu_usage(&pod.metadata.name.unwrap()).await? {
                        if gpu_usage > gpu_budget {
                            self.handle_violation(
                                &agent_id,
                                "gpu",
                                gpu_usage as f64 / gpu_budget as f64,
                            ).await?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_violation(
        &self,
        agent_id: &str,
        resource_type: &str,
        ratio: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tracker = self.violation_tracker.write().await;

        let record = tracker.entry(agent_id.to_string()).or_insert_with(|| {
            ViolationRecord {
                agent_id: agent_id.to_string(),
                first_violation: Instant::now(),
                violation_count: 0,
                resource_type: resource_type.to_string(),
            }
        });

        record.violation_count += 1;

        // Metrics
        metrics::QUOTA_VIOLATIONS_TOTAL
            .with_label_values(&[agent_id, resource_type])
            .inc();

        // Decide action based on severity
        if resource_type == "gpu" || ratio > 3.0 || record.violation_count > 3 {
            // Terminate
            self.terminate_agent(agent_id).await?;
            tracker.remove(agent_id);

            tracing::warn!(
                "Terminated agent {} for {} violation (ratio: {:.2})",
                agent_id, resource_type, ratio
            );
        } else if record.violation_count > 1 {
            // Throttle
            self.throttle_agent(agent_id).await?;

            tracing::warn!(
                "Throttled agent {} for {} violation (ratio: {:.2})",
                agent_id, resource_type, ratio
            );
        }

        Ok(())
    }

    async fn terminate_agent(&self, agent_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let pods_api: Api<Pod> = Api::namespaced(self.k8s_client.clone(), "prowzi");

        // Find and delete pod
        let pods = pods_api.list(&Default::default()).await?;
        for pod in pods.items {
            if pod.metadata.labels.as_ref()
                .and_then(|l| l.get("agent-id"))
                .map(|id| id == agent_id)
                .unwrap_or(false) 
            {
                pods_api.delete(&pod.metadata.name.unwrap(), &Default::default()).await?;
            }
        }

        Ok(())
    }

    async fn throttle_agent(&self, agent_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Send throttle command via NATS
        let nats = async_nats::connect("nats://nats:4222").await?;
        nats.publish(
            format!("commands.throttle.{}", agent_id),
            "reduce:50".into(),
        ).await?;

        Ok(())
    }
}

fn parse_cpu(cpu_str: &str) -> i64 {
    // Parse formats like "250m" or "1.5"
    if cpu_str.ends_with('m') {
        cpu_str.trim_end_matches('m').parse::<i64>().unwrap_or(0)
    } else {
        (cpu_str.parse::<f64>().unwrap_or(0.0) * 1000.0) as i64
    }
}

fn parse_memory(mem_str: &str) -> i64 {
    // Parse formats like "512Mi" or "1Gi"
    if mem_str.ends_with("Mi") {
        mem_str.trim_end_matches("Mi").parse::<i64>().unwrap_or(0)
    } else if mem_str.ends_with("Gi") {
        mem_str.trim_end_matches("Gi").parse::<i64>().unwrap_or(0) * 1024
    } else {
        mem_str.parse::<i64>().unwrap_or(0) / (1024 * 1024)
    }
}

// Prometheus metrics
lazy_static! {
    pub static ref QUOTA_VIOLATIONS_TOTAL: prometheus::CounterVec = 
        prometheus::register_counter_vec!(
            "prowzi_quota_violations_total",
            "Total quota violations by agent and resource type",
            &["agent_id", "resource_type"]
        ).unwrap();
}
