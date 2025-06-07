//! GPU Resource Management for Prowzi Agents
//! 
//! Provides GPU resource allocation, monitoring, and inference acceleration
//! for AI-powered trading agents.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;

/// GPU resource requirements for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    pub memory_mb: u32,
    pub compute_units: f32,
    pub tensor_cores: bool,
    pub fp16_support: bool,
    pub min_vram_mb: u32,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub device_id: u32,
    pub name: String,
    pub memory_total_mb: u32,
    pub memory_free_mb: u32,
    pub compute_capability: String,
    pub power_usage_w: f32,
    pub temperature_c: u32,
    pub utilization_percent: u32,
}

/// GPU allocation for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    pub allocation_id: String,
    pub agent_id: String,
    pub device_id: u32,
    pub memory_allocated_mb: u32,
    pub compute_units_allocated: f32,
    pub allocated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// GPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub device_id: u32,
    pub memory_used_mb: u32,
    pub compute_utilization: f32,
    pub power_draw_w: f32,
    pub temperature_c: u32,
    pub inference_throughput_ops_per_sec: f32,
    pub last_updated: DateTime<Utc>,
}

/// GPU inference acceleration types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceAcceleration {
    None,
    Cuda,
    OpenCL,
    TensorRT,
    DirectML,
    Metal,
}

/// GPU resource manager
pub struct GpuResourceManager {
    devices: Arc<RwLock<HashMap<u32, GpuDevice>>>,
    allocations: Arc<RwLock<HashMap<String, GpuAllocation>>>,
    metrics: Arc<RwLock<HashMap<u32, GpuMetrics>>>,
    allocation_semaphore: Arc<Semaphore>,
}

impl GpuResourceManager {
    pub fn new(max_concurrent_allocations: usize) -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            allocation_semaphore: Arc::new(Semaphore::new(max_concurrent_allocations)),
        }
    }

    /// Initialize GPU devices
    pub async fn initialize(&self) -> Result<(), GpuError> {
        // TODO: Implement actual GPU discovery and initialization
        // For now, create a mock GPU device
        let mock_device = GpuDevice {
            device_id: 0,
            name: "Mock GPU".to_string(),
            memory_total_mb: 8192,
            memory_free_mb: 8192,
            compute_capability: "8.0".to_string(),
            power_usage_w: 150.0,
            temperature_c: 45,
            utilization_percent: 0,
        };

        let mut devices = self.devices.write().await;
        devices.insert(0, mock_device);

        Ok(())
    }

    /// Allocate GPU resources for an agent
    pub async fn allocate_gpu(
        &self,
        agent_id: String,
        requirements: GpuRequirements,
    ) -> Result<GpuAllocation, GpuError> {
        let _permit = self.allocation_semaphore.acquire().await
            .map_err(|_| GpuError::AllocationFailed("Semaphore acquisition failed".to_string()))?;

        let devices = self.devices.read().await;
        let mut allocations = self.allocations.write().await;

        // Find suitable GPU device
        let suitable_device = devices.values()
            .find(|device| {
                device.memory_free_mb >= requirements.memory_mb &&
                device.memory_total_mb >= requirements.min_vram_mb
            })
            .ok_or(GpuError::NoSuitableDevice)?;

        let allocation = GpuAllocation {
            allocation_id: Uuid::new_v4().to_string(),
            agent_id,
            device_id: suitable_device.device_id,
            memory_allocated_mb: requirements.memory_mb,
            compute_units_allocated: requirements.compute_units,
            allocated_at: Utc::now(),
            expires_at: None,
        };

        allocations.insert(allocation.allocation_id.clone(), allocation.clone());

        Ok(allocation)
    }

    /// Release GPU resources
    pub async fn release_gpu(&self, allocation_id: &str) -> Result<(), GpuError> {
        let mut allocations = self.allocations.write().await;
        allocations.remove(allocation_id)
            .ok_or(GpuError::AllocationNotFound)?;

        Ok(())
    }

    /// Get GPU metrics
    pub async fn get_metrics(&self, device_id: u32) -> Option<GpuMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(&device_id).cloned()
    }

    /// Update GPU metrics
    pub async fn update_metrics(&self, device_id: u32, metrics: GpuMetrics) {
        let mut metrics_map = self.metrics.write().await;
        metrics_map.insert(device_id, metrics);
    }

    /// Get all GPU devices
    pub async fn get_devices(&self) -> Vec<GpuDevice> {
        let devices = self.devices.read().await;
        devices.values().cloned().collect()
    }

    /// Get active allocations
    pub async fn get_allocations(&self) -> Vec<GpuAllocation> {
        let allocations = self.allocations.read().await;
        allocations.values().cloned().collect()
    }

    /// Check if GPU acceleration is available
    pub fn is_acceleration_available(&self, acceleration: &InferenceAcceleration) -> bool {
        match acceleration {
            InferenceAcceleration::None => true,
            InferenceAcceleration::Cuda => cfg!(feature = "cuda"),
            InferenceAcceleration::OpenCL => cfg!(feature = "opencl"),
            InferenceAcceleration::TensorRT => cfg!(feature = "tensorrt"),
            InferenceAcceleration::DirectML => cfg!(target_os = "windows"),
            InferenceAcceleration::Metal => cfg!(target_os = "macos"),
        }
    }
}

/// GPU-related errors
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No suitable GPU device found")]
    NoSuitableDevice,

    #[error("GPU allocation failed: {0}")]
    AllocationFailed(String),

    #[error("GPU allocation not found")]
    AllocationNotFound,

    #[error("GPU memory exhausted")]
    MemoryExhausted,

    #[error("GPU compute units exhausted")]
    ComputeExhausted,

    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    #[error("GPU driver error: {0}")]
    DriverError(String),
}

/// GPU inference context for agents
pub struct GpuInferenceContext {
    pub allocation: GpuAllocation,
    pub acceleration: InferenceAcceleration,
    pub precision: InferencePrecision,
}

/// Inference precision settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferencePrecision {
    FP32,
    FP16,
    INT8,
    INT4,
}

impl GpuInferenceContext {
    pub fn new(
        allocation: GpuAllocation,
        acceleration: InferenceAcceleration,
        precision: InferencePrecision,
    ) -> Self {
        Self {
            allocation,
            acceleration,
            precision,
        }
    }

    /// Execute inference with GPU acceleration
    pub async fn execute_inference<T>(&self, input: T) -> Result<T, GpuError> {
        // TODO: Implement actual GPU inference
        // For now, return input as placeholder
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_manager_initialization() {
        let manager = GpuResourceManager::new(4);
        assert!(manager.initialize().await.is_ok());

        let devices = manager.get_devices().await;
        assert!(!devices.is_empty());
    }

    #[tokio::test]
    async fn test_gpu_allocation() {
        let manager = GpuResourceManager::new(4);
        manager.initialize().await.unwrap();

        let requirements = GpuRequirements {
            memory_mb: 1024,
            compute_units: 0.5,
            tensor_cores: false,
            fp16_support: false,
            min_vram_mb: 2048,
        };

        let allocation = manager.allocate_gpu("test_agent".to_string(), requirements).await;
        assert!(allocation.is_ok());

        let allocation = allocation.unwrap();
        assert!(manager.release_gpu(&allocation.allocation_id).await.is_ok());
    }

    #[tokio::test]
    async fn test_acceleration_availability() {
        let manager = GpuResourceManager::new(4);
        
        // None should always be available
        assert!(manager.is_acceleration_available(&InferenceAcceleration::None));
        
        // Platform-specific accelerations
        #[cfg(target_os = "windows")]
        assert!(manager.is_acceleration_available(&InferenceAcceleration::DirectML));
        
        #[cfg(target_os = "macos")]
        assert!(manager.is_acceleration_available(&InferenceAcceleration::Metal));
    }
}
