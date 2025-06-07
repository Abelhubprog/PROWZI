
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

use crate::{ResourceRequirements, ResourceUsage};

#[derive(Debug, Clone)]
pub struct TokenBucket {
    capacity: u64,
    tokens: u64,
    refill_rate: u64,
    last_refill: Instant,
}

impl TokenBucket {
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }
    
    pub fn try_consume(&mut self, amount: u64) -> bool {
        self.refill();
        
        if self.tokens >= amount {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }
    
    pub fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        
        let tokens_to_add = (elapsed.as_secs_f64() * self.refill_rate as f64) as u64;
        self.tokens = (self.tokens + tokens_to_add).min(self.capacity);
        self.last_refill = now;
    }
    
    pub fn available(&self) -> u64 {
        self.tokens
    }
}

pub struct BudgetManager {
    budgets: Arc<RwLock<HashMap<String, MissionBudget>>>,
}

#[derive(Debug)]
pub struct MissionBudget {
    pub cpu_ms: TokenBucket,
    pub gpu_minutes: TokenBucket,
    pub tokens: TokenBucket,
    pub bandwidth_mb: TokenBucket,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetLimits {
    pub cpu_ms_capacity: u64,
    pub cpu_ms_refill_rate: u64,
    pub gpu_minutes_capacity: u64,
    pub gpu_minutes_refill_rate: u64,
    pub tokens_capacity: u64,
    pub tokens_refill_rate: u64,
    pub bandwidth_mb_capacity: u64,
    pub bandwidth_mb_refill_rate: u64,
}

impl Default for BudgetLimits {
    fn default() -> Self {
        Self {
            cpu_ms_capacity: 3600000,     // 1 hour
            cpu_ms_refill_rate: 1000,     // 1 second per second
            gpu_minutes_capacity: 60,     // 1 hour
            gpu_minutes_refill_rate: 1,   // 1 minute per minute
            tokens_capacity: 100000,      // 100k tokens
            tokens_refill_rate: 100,      // 100 tokens per second
            bandwidth_mb_capacity: 1000,  // 1GB
            bandwidth_mb_refill_rate: 10, // 10MB per second
        }
    }
}

#[derive(Debug, Serialize)]
pub struct BudgetStatus {
    pub cpu_ms_available: u64,
    pub gpu_minutes_available: u64,
    pub tokens_available: u64,
    pub bandwidth_mb_available: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    #[error("Mission not found")]
    MissionNotFound,
    
    #[error("Insufficient CPU")]
    InsufficientCPU,
    
    #[error("Insufficient GPU")]
    InsufficientGPU,
    
    #[error("Insufficient tokens")]
    InsufficientTokens,
    
    #[error("Insufficient bandwidth")]
    InsufficientBandwidth,
}

impl MissionBudget {
    pub fn new(limits: BudgetLimits) -> Self {
        Self {
            cpu_ms: TokenBucket::new(limits.cpu_ms_capacity, limits.cpu_ms_refill_rate),
            gpu_minutes: TokenBucket::new(limits.gpu_minutes_capacity, limits.gpu_minutes_refill_rate),
            tokens: TokenBucket::new(limits.tokens_capacity, limits.tokens_refill_rate),
            bandwidth_mb: TokenBucket::new(limits.bandwidth_mb_capacity, limits.bandwidth_mb_refill_rate),
        }
    }
}

impl BudgetManager {
    pub fn new() -> Self {
        Self {
            budgets: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn create_mission_budget(&self, mission_id: &str, limits: BudgetLimits) {
        let mut budgets = self.budgets.write().await;
        budgets.insert(mission_id.to_string(), MissionBudget::new(limits));
    }
    
    pub async fn check_budget(
        &self,
        mission_id: &str,
        requirements: &ResourceRequirements,
    ) -> Result<bool, BudgetError> {
        let mut budgets = self.budgets.write().await;
        
        // Create default budget if not exists
        if !budgets.contains_key(mission_id) {
            budgets.insert(mission_id.to_string(), MissionBudget::new(BudgetLimits::default()));
        }
        
        let budget = budgets
            .get_mut(mission_id)
            .ok_or(BudgetError::MissionNotFound)?;
            
        // Check all resources
        let cpu_ok = budget.cpu_ms.available() >= requirements.cpu_ms;
        let gpu_ok = requirements.gpu_minutes.map_or(true, |gpu| {
            budget.gpu_minutes.available() >= gpu
        });
        let tokens_ok = budget.tokens.available() >= requirements.tokens;
        let bandwidth_ok = budget.bandwidth_mb.available() >= requirements.bandwidth_mb;
        
        Ok(cpu_ok && gpu_ok && tokens_ok && bandwidth_ok)
    }
    
    pub async fn reserve_budget(
        &self,
        mission_id: &str,
        requirements: &ResourceRequirements,
    ) -> Result<(), BudgetError> {
        let mut budgets = self.budgets.write().await;
        
        let budget = budgets
            .get_mut(mission_id)
            .ok_or(BudgetError::MissionNotFound)?;
            
        // Consume from buckets
        if !budget.cpu_ms.try_consume(requirements.cpu_ms) {
            return Err(BudgetError::InsufficientCPU);
        }
        
        if let Some(gpu) = requirements.gpu_minutes {
            if !budget.gpu_minutes.try_consume(gpu) {
                // Rollback CPU
                budget.cpu_ms.tokens += requirements.cpu_ms;
                return Err(BudgetError::InsufficientGPU);
            }
        }
        
        if !budget.tokens.try_consume(requirements.tokens) {
            // Rollback
            budget.cpu_ms.tokens += requirements.cpu_ms;
            if let Some(gpu) = requirements.gpu_minutes {
                budget.gpu_minutes.tokens += gpu;
            }
            return Err(BudgetError::InsufficientTokens);
        }
        
        if !budget.bandwidth_mb.try_consume(requirements.bandwidth_mb) {
            // Rollback all
            budget.cpu_ms.tokens += requirements.cpu_ms;
            if let Some(gpu) = requirements.gpu_minutes {
                budget.gpu_minutes.tokens += gpu;
            }
            budget.tokens.tokens += requirements.tokens;
            return Err(BudgetError::InsufficientBandwidth);
        }
        
        Ok(())
    }
    
    pub async fn get_budget_status(
        &self,
        mission_id: &str,
    ) -> Result<BudgetStatus, BudgetError> {
        let budgets = self.budgets.read().await;
        
        let budget = budgets
            .get(mission_id)
            .ok_or(BudgetError::MissionNotFound)?;
            
        Ok(BudgetStatus {
            cpu_ms_available: budget.cpu_ms.available(),
            gpu_minutes_available: budget.gpu_minutes.available(),
            tokens_available: budget.tokens.available(),
            bandwidth_mb_available: budget.bandwidth_mb.available(),
        })
    }
    
    pub async fn update_limits(&self, mission_id: &str, _new_limits: ResourceRequirements) -> Result<(), BudgetError> {
        // Implementation would update the bucket limits
        Ok(())
    }
    
    pub async fn is_budget_exhausted(&self, mission_id: &str) -> bool {
        let budgets = self.budgets.read().await;
        if let Some(budget) = budgets.get(mission_id) {
            budget.tokens.available() < 100 && budget.cpu_ms.available() < 1000
        } else {
            false
        }
    }
}
