
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("Budget exhausted for resource: {resource}")]
    Exhausted { resource: String },
    #[error("Invalid budget allocation: {0}")]
    InvalidAllocation(String),
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    #[error("Concurrent modification detected")]
    ConcurrentModification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub tokens: u64,
    pub api_calls: u32,
    pub compute_hours: f32,
    pub memory_mb: u32,
    pub storage_gb: f32,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            tokens: 10_000,
            api_calls: 1_000,
            compute_hours: 1.0,
            memory_mb: 512,
            storage_gb: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub tokens_used: u64,
    pub api_calls_made: u32,
    pub compute_hours_used: f32,
    pub memory_mb_used: u32,
    pub storage_gb_used: f32,
    pub last_updated: DateTime<Utc>,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            tokens_used: 0,
            api_calls_made: 0,
            compute_hours_used: 0.0,
            memory_mb_used: 0,
            storage_gb_used: 0.0,
            last_updated: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAllocation {
    pub agent_id: String,
    pub mission_id: Option<String>,
    pub allocated: ResourceBudget,
    pub used: ResourceUsage,
    pub reserved: ResourceBudget,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

pub struct BudgetManager {
    allocations: Arc<RwLock<HashMap<String, BudgetAllocation>>>,
    global_limits: Arc<RwLock<ResourceBudget>>,
    usage_history: Arc<RwLock<Vec<(String, ResourceUsage, DateTime<Utc>)>>>,
    alert_thresholds: Arc<RwLock<HashMap<String, f32>>>, // Resource -> threshold (0.0-1.0)
}

impl BudgetManager {
    pub fn new(global_limits: ResourceBudget) -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("tokens".to_string(), 0.8);
        thresholds.insert("api_calls".to_string(), 0.9);
        thresholds.insert("compute_hours".to_string(), 0.75);
        thresholds.insert("memory_mb".to_string(), 0.85);
        thresholds.insert("storage_gb".to_string(), 0.9);

        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            global_limits: Arc::new(RwLock::new(global_limits)),
            usage_history: Arc::new(RwLock::new(Vec::new())),
            alert_thresholds: Arc::new(RwLock::new(thresholds)),
        }
    }

    pub async fn allocate_budget(
        &self,
        agent_id: String,
        mission_id: Option<String>,
        requested: ResourceBudget,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<BudgetAllocation, BudgetError> {
        let mut allocations = self.allocations.write().await;
        let global_limits = self.global_limits.read().await;

        // Check if we have enough global resources
        let total_allocated = self.calculate_total_allocated(&allocations).await;
        
        if total_allocated.tokens + requested.tokens > global_limits.tokens {
            return Err(BudgetError::Exhausted {
                resource: "tokens".to_string(),
            });
        }

        if total_allocated.api_calls + requested.api_calls > global_limits.api_calls {
            return Err(BudgetError::Exhausted {
                resource: "api_calls".to_string(),
            });
        }

        if total_allocated.compute_hours + requested.compute_hours > global_limits.compute_hours {
            return Err(BudgetError::Exhausted {
                resource: "compute_hours".to_string(),
            });
        }

        let allocation = BudgetAllocation {
            agent_id: agent_id.clone(),
            mission_id,
            allocated: requested,
            used: ResourceUsage::default(),
            reserved: ResourceBudget::default(),
            created_at: Utc::now(),
            expires_at,
        };

        allocations.insert(agent_id, allocation.clone());
        Ok(allocation)
    }

    pub async fn record_usage(
        &self,
        agent_id: &str,
        usage: ResourceUsage,
    ) -> Result<ResourceBudget, BudgetError> {
        let mut allocations = self.allocations.write().await;
        let mut history = self.usage_history.write().await;

        let allocation = allocations
            .get_mut(agent_id)
            .ok_or_else(|| BudgetError::ResourceNotFound(agent_id.to_string()))?;

        // Check if usage exceeds allocation
        if allocation.used.tokens_used + usage.tokens_used > allocation.allocated.tokens {
            return Err(BudgetError::Exhausted {
                resource: "tokens".to_string(),
            });
        }

        if allocation.used.api_calls_made + usage.api_calls_made > allocation.allocated.api_calls {
            return Err(BudgetError::Exhausted {
                resource: "api_calls".to_string(),
            });
        }

        if allocation.used.compute_hours_used + usage.compute_hours_used > allocation.allocated.compute_hours {
            return Err(BudgetError::Exhausted {
                resource: "compute_hours".to_string(),
            });
        }

        // Update usage
        allocation.used.tokens_used += usage.tokens_used;
        allocation.used.api_calls_made += usage.api_calls_made;
        allocation.used.compute_hours_used += usage.compute_hours_used;
        allocation.used.memory_mb_used = allocation.used.memory_mb_used.max(usage.memory_mb_used);
        allocation.used.storage_gb_used += usage.storage_gb_used;
        allocation.used.last_updated = Utc::now();

        // Record in history
        history.push((agent_id.to_string(), usage, Utc::now()));

        // Keep only last 1000 entries
        if history.len() > 1000 {
            history.drain(0..100);
        }

        // Calculate remaining budget
        let remaining = ResourceBudget {
            tokens: allocation.allocated.tokens - allocation.used.tokens_used,
            api_calls: allocation.allocated.api_calls - allocation.used.api_calls_made,
            compute_hours: allocation.allocated.compute_hours - allocation.used.compute_hours_used,
            memory_mb: allocation.allocated.memory_mb - allocation.used.memory_mb_used,
            storage_gb: allocation.allocated.storage_gb - allocation.used.storage_gb_used,
        };

        Ok(remaining)
    }

    pub async fn check_budget(&self, agent_id: &str) -> Result<ResourceBudget, BudgetError> {
        let allocations = self.allocations.read().await;
        let allocation = allocations
            .get(agent_id)
            .ok_or_else(|| BudgetError::ResourceNotFound(agent_id.to_string()))?;

        let remaining = ResourceBudget {
            tokens: allocation.allocated.tokens - allocation.used.tokens_used,
            api_calls: allocation.allocated.api_calls - allocation.used.api_calls_made,
            compute_hours: allocation.allocated.compute_hours - allocation.used.compute_hours_used,
            memory_mb: allocation.allocated.memory_mb - allocation.used.memory_mb_used,
            storage_gb: allocation.allocated.storage_gb - allocation.used.storage_gb_used,
        };

        Ok(remaining)
    }

    pub async fn get_usage_percentage(&self, agent_id: &str) -> Result<HashMap<String, f32>, BudgetError> {
        let allocations = self.allocations.read().await;
        let allocation = allocations
            .get(agent_id)
            .ok_or_else(|| BudgetError::ResourceNotFound(agent_id.to_string()))?;

        let mut percentages = HashMap::new();
        
        percentages.insert(
            "tokens".to_string(),
            allocation.used.tokens_used as f32 / allocation.allocated.tokens as f32,
        );
        
        percentages.insert(
            "api_calls".to_string(),
            allocation.used.api_calls_made as f32 / allocation.allocated.api_calls as f32,
        );
        
        percentages.insert(
            "compute_hours".to_string(),
            allocation.used.compute_hours_used / allocation.allocated.compute_hours,
        );
        
        percentages.insert(
            "memory_mb".to_string(),
            allocation.used.memory_mb_used as f32 / allocation.allocated.memory_mb as f32,
        );
        
        percentages.insert(
            "storage_gb".to_string(),
            allocation.used.storage_gb_used / allocation.allocated.storage_gb,
        );

        Ok(percentages)
    }

    pub async fn check_alerts(&self, agent_id: &str) -> Result<Vec<String>, BudgetError> {
        let percentages = self.get_usage_percentage(agent_id).await?;
        let thresholds = self.alert_thresholds.read().await;
        let mut alerts = Vec::new();

        for (resource, percentage) in percentages {
            if let Some(threshold) = thresholds.get(&resource) {
                if percentage >= *threshold {
                    alerts.push(format!(
                        "Resource {} is at {:.1}% usage (threshold: {:.1}%)",
                        resource,
                        percentage * 100.0,
                        threshold * 100.0
                    ));
                }
            }
        }

        Ok(alerts)
    }

    pub async fn extend_budget(
        &self,
        agent_id: &str,
        additional: ResourceBudget,
    ) -> Result<(), BudgetError> {
        let mut allocations = self.allocations.write().await;
        let global_limits = self.global_limits.read().await;

        let allocation = allocations
            .get_mut(agent_id)
            .ok_or_else(|| BudgetError::ResourceNotFound(agent_id.to_string()))?;

        // Check global limits
        let total_allocated = self.calculate_total_allocated(&allocations).await;
        
        if total_allocated.tokens + additional.tokens > global_limits.tokens {
            return Err(BudgetError::Exhausted {
                resource: "tokens".to_string(),
            });
        }

        // Extend allocation
        allocation.allocated.tokens += additional.tokens;
        allocation.allocated.api_calls += additional.api_calls;
        allocation.allocated.compute_hours += additional.compute_hours;
        allocation.allocated.memory_mb += additional.memory_mb;
        allocation.allocated.storage_gb += additional.storage_gb;

        Ok(())
    }

    pub async fn release_budget(&self, agent_id: &str) -> Result<(), BudgetError> {
        let mut allocations = self.allocations.write().await;
        allocations.remove(agent_id);
        Ok(())
    }

    pub async fn get_all_allocations(&self) -> HashMap<String, BudgetAllocation> {
        let allocations = self.allocations.read().await;
        allocations.clone()
    }

    pub async fn cleanup_expired(&self) -> usize {
        let mut allocations = self.allocations.write().await;
        let now = Utc::now();
        let initial_count = allocations.len();

        allocations.retain(|_, allocation| {
            match allocation.expires_at {
                Some(expires) => expires > now,
                None => true, // No expiration
            }
        });

        initial_count - allocations.len()
    }

    async fn calculate_total_allocated(&self, allocations: &HashMap<String, BudgetAllocation>) -> ResourceBudget {
        let mut total = ResourceBudget::default();
        total.tokens = 0;
        total.api_calls = 0;
        total.compute_hours = 0.0;
        total.memory_mb = 0;
        total.storage_gb = 0.0;

        for allocation in allocations.values() {
            total.tokens += allocation.allocated.tokens;
            total.api_calls += allocation.allocated.api_calls;
            total.compute_hours += allocation.allocated.compute_hours;
            total.memory_mb += allocation.allocated.memory_mb;
            total.storage_gb += allocation.allocated.storage_gb;
        }

        total
    }

    pub async fn get_usage_history(&self, agent_id: Option<&str>) -> Vec<(String, ResourceUsage, DateTime<Utc>)> {
        let history = self.usage_history.read().await;
        
        match agent_id {
            Some(id) => history
                .iter()
                .filter(|(agent, _, _)| agent == id)
                .cloned()
                .collect(),
            None => history.clone(),
        }
    }

    pub async fn update_global_limits(&self, new_limits: ResourceBudget) -> Result<(), BudgetError> {
        let mut global_limits = self.global_limits.write().await;
        let allocations = self.allocations.read().await;

        // Check if current allocations exceed new limits
        let total_allocated = self.calculate_total_allocated(&allocations).await;
        
        if total_allocated.tokens > new_limits.tokens {
            return Err(BudgetError::InvalidAllocation(
                format!("New token limit {} is less than currently allocated {}", 
                    new_limits.tokens, total_allocated.tokens)
            ));
        }

        *global_limits = new_limits;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_budget_allocation() {
        let global_limits = ResourceBudget {
            tokens: 100_000,
            api_calls: 10_000,
            compute_hours: 100.0,
            memory_mb: 8192,
            storage_gb: 100.0,
        };

        let manager = BudgetManager::new(global_limits);
        
        let requested = ResourceBudget {
            tokens: 1000,
            api_calls: 100,
            compute_hours: 1.0,
            memory_mb: 512,
            storage_gb: 1.0,
        };

        let allocation = manager
            .allocate_budget("test_agent".to_string(), None, requested, None)
            .await
            .unwrap();

        assert_eq!(allocation.agent_id, "test_agent");
        assert_eq!(allocation.allocated.tokens, 1000);
    }

    #[tokio::test]
    async fn test_usage_recording() {
        let global_limits = ResourceBudget::default();
        let manager = BudgetManager::new(global_limits);
        
        let requested = ResourceBudget {
            tokens: 1000,
            api_calls: 100,
            compute_hours: 1.0,
            memory_mb: 512,
            storage_gb: 1.0,
        };

        manager
            .allocate_budget("test_agent".to_string(), None, requested, None)
            .await
            .unwrap();

        let usage = ResourceUsage {
            tokens_used: 100,
            api_calls_made: 10,
            compute_hours_used: 0.1,
            memory_mb_used: 256,
            storage_gb_used: 0.1,
            last_updated: Utc::now(),
        };

        let remaining = manager.record_usage("test_agent", usage).await.unwrap();
        assert_eq!(remaining.tokens, 900);
        assert_eq!(remaining.api_calls, 90);
    }
}
