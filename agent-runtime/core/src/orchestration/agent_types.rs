use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum AgentType {
    Scout,
    Planner,
    Trader,
    RiskSentinel,
    Guardian,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum AgentRole {
    MarketIntelligence,
    StrategyFormulation,
    OrderExecution,
    RiskManagement,
    EmergencyControl,
    PerformanceOptimization,
    DataAggregation,
    ComplianceMonitoring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub processing_power: ProcessingCapability,
    pub memory_capacity: MemoryCapability,
    pub network_bandwidth: NetworkCapability,
    pub specialized_functions: Vec<SpecializedFunction>,
    pub ai_models: Vec<AIModelCapability>,
    pub real_time_performance: RealTimeCapability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingCapability {
    pub cpu_cores: u32,
    pub gpu_acceleration: bool,
    pub quantum_processing: bool,
    pub max_operations_per_second: u64,
    pub parallel_task_limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCapability {
    pub total_memory_gb: u32,
    pub cache_size_mb: u32,
    pub persistent_storage_gb: u32,
    pub ultra_low_latency_access: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapability {
    pub bandwidth_mbps: u32,
    pub latency_microseconds: u32,
    pub concurrent_connections: u32,
    pub quantum_entanglement_simulation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecializedFunction {
    MarketDataAnalysis,
    PredictiveModeling,
    RiskAssessment,
    OrderRouting,
    PortfolioOptimization,
    SentimentAnalysis,
    TechnicalAnalysis,
    FundamentalAnalysis,
    ArbitrageDetection,
    LiquidityProvision,
    EmergencyShutdown,
    ComplianceValidation,
    PerformanceBenchmarking,
    AIModelTraining,
    QuantumOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModelCapability {
    pub model_type: AIModelType,
    pub accuracy_percentage: f64,
    pub inference_time_ms: f64,
    pub training_frequency: TrainingFrequency,
    pub specialized_domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIModelType {
    DeepLearning,
    ReinforcementLearning,
    TransformerBased,
    ConvolutionalNeural,
    RecurrentNeural,
    GradientBoosting,
    SupportVectorMachine,
    RandomForest,
    QuantumMachineLearning,
    HybridNeuroSymbolic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingFrequency {
    RealTime,
    Continuous,
    Hourly,
    Daily,
    Weekly,
    EventDriven,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeCapability {
    pub max_response_time_ms: f64,
    pub throughput_operations_per_second: u64,
    pub concurrent_task_handling: u32,
    pub predictive_caching: bool,
    pub adaptive_optimization: bool,
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            processing_power: ProcessingCapability {
                cpu_cores: 8,
                gpu_acceleration: true,
                quantum_processing: false,
                max_operations_per_second: 1_000_000,
                parallel_task_limit: 100,
            },
            memory_capacity: MemoryCapability {
                total_memory_gb: 32,
                cache_size_mb: 512,
                persistent_storage_gb: 1000,
                ultra_low_latency_access: true,
            },
            network_bandwidth: NetworkCapability {
                bandwidth_mbps: 10_000,
                latency_microseconds: 100,
                concurrent_connections: 1000,
                quantum_entanglement_simulation: false,
            },
            specialized_functions: Vec::new(),
            ai_models: Vec::new(),
            real_time_performance: RealTimeCapability {
                max_response_time_ms: 10.0,
                throughput_operations_per_second: 100_000,
                concurrent_task_handling: 50,
                predictive_caching: true,
                adaptive_optimization: true,
            },
        }
    }
}

impl AgentType {
    pub fn default_role(&self) -> AgentRole {
        match self {
            AgentType::Scout => AgentRole::MarketIntelligence,
            AgentType::Planner => AgentRole::StrategyFormulation,
            AgentType::Trader => AgentRole::OrderExecution,
            AgentType::RiskSentinel => AgentRole::RiskManagement,
            AgentType::Guardian => AgentRole::EmergencyControl,
        }
    }

    pub fn default_capabilities(&self) -> AgentCapabilities {
        let mut capabilities = AgentCapabilities::default();
        
        match self {
            AgentType::Scout => {
                capabilities.specialized_functions = vec![
                    SpecializedFunction::MarketDataAnalysis,
                    SpecializedFunction::SentimentAnalysis,
                    SpecializedFunction::TechnicalAnalysis,
                    SpecializedFunction::ArbitrageDetection,
                ];
                capabilities.ai_models = vec![
                    AIModelCapability {
                        model_type: AIModelType::TransformerBased,
                        accuracy_percentage: 92.5,
                        inference_time_ms: 5.0,
                        training_frequency: TrainingFrequency::RealTime,
                        specialized_domains: vec!["market_sentiment".to_string(), "price_prediction".to_string()],
                    },
                    AIModelCapability {
                        model_type: AIModelType::ConvolutionalNeural,
                        accuracy_percentage: 89.0,
                        inference_time_ms: 3.0,
                        training_frequency: TrainingFrequency::Continuous,
                        specialized_domains: vec!["technical_patterns".to_string(), "chart_analysis".to_string()],
                    },
                ];
                capabilities.real_time_performance.max_response_time_ms = 5.0;
                capabilities.real_time_performance.throughput_operations_per_second = 200_000;
            },
            
            AgentType::Planner => {
                capabilities.specialized_functions = vec![
                    SpecializedFunction::PredictiveModeling,
                    SpecializedFunction::PortfolioOptimization,
                    SpecializedFunction::FundamentalAnalysis,
                    SpecializedFunction::QuantumOptimization,
                ];
                capabilities.ai_models = vec![
                    AIModelCapability {
                        model_type: AIModelType::ReinforcementLearning,
                        accuracy_percentage: 94.0,
                        inference_time_ms: 15.0,
                        training_frequency: TrainingFrequency::Continuous,
                        specialized_domains: vec!["strategy_optimization".to_string(), "portfolio_allocation".to_string()],
                    },
                    AIModelCapability {
                        model_type: AIModelType::QuantumMachineLearning,
                        accuracy_percentage: 96.5,
                        inference_time_ms: 8.0,
                        training_frequency: TrainingFrequency::EventDriven,
                        specialized_domains: vec!["multi_dimensional_optimization".to_string()],
                    },
                ];
                capabilities.processing_power.quantum_processing = true;
                capabilities.real_time_performance.max_response_time_ms = 20.0;
            },
            
            AgentType::Trader => {
                capabilities.specialized_functions = vec![
                    SpecializedFunction::OrderRouting,
                    SpecializedFunction::LiquidityProvision,
                    SpecializedFunction::ArbitrageDetection,
                ];
                capabilities.ai_models = vec![
                    AIModelCapability {
                        model_type: AIModelType::DeepLearning,
                        accuracy_percentage: 91.0,
                        inference_time_ms: 2.0,
                        training_frequency: TrainingFrequency::RealTime,
                        specialized_domains: vec!["execution_optimization".to_string(), "slippage_minimization".to_string()],
                    },
                ];
                capabilities.network_bandwidth.latency_microseconds = 50;
                capabilities.real_time_performance.max_response_time_ms = 2.0;
                capabilities.real_time_performance.throughput_operations_per_second = 500_000;
            },
            
            AgentType::RiskSentinel => {
                capabilities.specialized_functions = vec![
                    SpecializedFunction::RiskAssessment,
                    SpecializedFunction::ComplianceValidation,
                    SpecializedFunction::PredictiveModeling,
                ];
                capabilities.ai_models = vec![
                    AIModelCapability {
                        model_type: AIModelType::HybridNeuroSymbolic,
                        accuracy_percentage: 97.0,
                        inference_time_ms: 1.0,
                        training_frequency: TrainingFrequency::RealTime,
                        specialized_domains: vec!["risk_prediction".to_string(), "anomaly_detection".to_string()],
                    },
                    AIModelCapability {
                        model_type: AIModelType::GradientBoosting,
                        accuracy_percentage: 93.5,
                        inference_time_ms: 0.5,
                        training_frequency: TrainingFrequency::Continuous,
                        specialized_domains: vec!["fraud_detection".to_string(), "compliance_scoring".to_string()],
                    },
                ];
                capabilities.real_time_performance.max_response_time_ms = 1.0;
                capabilities.real_time_performance.predictive_caching = true;
            },
            
            AgentType::Guardian => {
                capabilities.specialized_functions = vec![
                    SpecializedFunction::EmergencyShutdown,
                    SpecializedFunction::PerformanceBenchmarking,
                    SpecializedFunction::ComplianceValidation,
                ];
                capabilities.ai_models = vec![
                    AIModelCapability {
                        model_type: AIModelType::ReinforcementLearning,
                        accuracy_percentage: 99.0,
                        inference_time_ms: 0.5,
                        training_frequency: TrainingFrequency::EventDriven,
                        specialized_domains: vec!["emergency_response".to_string(), "system_protection".to_string()],
                    },
                ];
                capabilities.processing_power.max_operations_per_second = 10_000_000;
                capabilities.real_time_performance.max_response_time_ms = 0.5;
                capabilities.network_bandwidth.quantum_entanglement_simulation = true;
            },
        }
        
        capabilities
    }

    pub fn compatible_roles(&self) -> Vec<AgentRole> {
        match self {
            AgentType::Scout => vec![
                AgentRole::MarketIntelligence,
                AgentRole::DataAggregation,
                AgentRole::PerformanceOptimization,
            ],
            AgentType::Planner => vec![
                AgentRole::StrategyFormulation,
                AgentRole::PerformanceOptimization,
            ],
            AgentType::Trader => vec![
                AgentRole::OrderExecution,
                AgentRole::PerformanceOptimization,
            ],
            AgentType::RiskSentinel => vec![
                AgentRole::RiskManagement,
                AgentRole::ComplianceMonitoring,
                AgentRole::PerformanceOptimization,
            ],
            AgentType::Guardian => vec![
                AgentRole::EmergencyControl,
                AgentRole::ComplianceMonitoring,
                AgentRole::PerformanceOptimization,
            ],
        }
    }

    pub fn performance_weights(&self) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        
        match self {
            AgentType::Scout => {
                weights.insert("accuracy".to_string(), 0.4);
                weights.insert("speed".to_string(), 0.3);
                weights.insert("coverage".to_string(), 0.2);
                weights.insert("reliability".to_string(), 0.1);
            },
            AgentType::Planner => {
                weights.insert("accuracy".to_string(), 0.5);
                weights.insert("optimization".to_string(), 0.3);
                weights.insert("adaptability".to_string(), 0.2);
            },
            AgentType::Trader => {
                weights.insert("speed".to_string(), 0.4);
                weights.insert("accuracy".to_string(), 0.3);
                weights.insert("efficiency".to_string(), 0.2);
                weights.insert("reliability".to_string(), 0.1);
            },
            AgentType::RiskSentinel => {
                weights.insert("accuracy".to_string(), 0.5);
                weights.insert("speed".to_string(), 0.3);
                weights.insert("sensitivity".to_string(), 0.2);
            },
            AgentType::Guardian => {
                weights.insert("reliability".to_string(), 0.4);
                weights.insert("speed".to_string(), 0.3);
                weights.insert("robustness".to_string(), 0.3);
            },
        }
        
        weights
    }

    pub fn required_infrastructure(&self) -> InfrastructureRequirements {
        match self {
            AgentType::Scout => InfrastructureRequirements {
                min_cpu_cores: 4,
                min_memory_gb: 8,
                requires_gpu: true,
                requires_quantum: false,
                network_requirements: NetworkRequirements::HighBandwidth,
                storage_requirements: StorageRequirements::FastAccess,
            },
            AgentType::Planner => InfrastructureRequirements {
                min_cpu_cores: 8,
                min_memory_gb: 16,
                requires_gpu: true,
                requires_quantum: true,
                network_requirements: NetworkRequirements::Standard,
                storage_requirements: StorageRequirements::LargeCapacity,
            },
            AgentType::Trader => InfrastructureRequirements {
                min_cpu_cores: 8,
                min_memory_gb: 16,
                requires_gpu: false,
                requires_quantum: false,
                network_requirements: NetworkRequirements::UltraLowLatency,
                storage_requirements: StorageRequirements::FastAccess,
            },
            AgentType::RiskSentinel => InfrastructureRequirements {
                min_cpu_cores: 6,
                min_memory_gb: 12,
                requires_gpu: true,
                requires_quantum: false,
                network_requirements: NetworkRequirements::UltraLowLatency,
                storage_requirements: StorageRequirements::HighReliability,
            },
            AgentType::Guardian => InfrastructureRequirements {
                min_cpu_cores: 16,
                min_memory_gb: 32,
                requires_gpu: true,
                requires_quantum: true,
                network_requirements: NetworkRequirements::UltraLowLatency,
                storage_requirements: StorageRequirements::HighReliability,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct InfrastructureRequirements {
    pub min_cpu_cores: u32,
    pub min_memory_gb: u32,
    pub requires_gpu: bool,
    pub requires_quantum: bool,
    pub network_requirements: NetworkRequirements,
    pub storage_requirements: StorageRequirements,
}

#[derive(Debug, Clone)]
pub enum NetworkRequirements {
    Standard,
    HighBandwidth,
    UltraLowLatency,
    QuantumEntangled,
}

#[derive(Debug, Clone)]
pub enum StorageRequirements {
    Standard,
    FastAccess,
    LargeCapacity,
    HighReliability,
    QuantumStorage,
}

impl AgentRole {
    pub fn priority_level(&self) -> u8 {
        match self {
            AgentRole::EmergencyControl => 1,
            AgentRole::RiskManagement => 2,
            AgentRole::OrderExecution => 3,
            AgentRole::ComplianceMonitoring => 4,
            AgentRole::StrategyFormulation => 5,
            AgentRole::MarketIntelligence => 6,
            AgentRole::PerformanceOptimization => 7,
            AgentRole::DataAggregation => 8,
        }
    }

    pub fn response_time_sla(&self) -> Duration {
        match self {
            AgentRole::EmergencyControl => Duration::from_millis(1),
            AgentRole::RiskManagement => Duration::from_millis(2),
            AgentRole::OrderExecution => Duration::from_millis(5),
            AgentRole::ComplianceMonitoring => Duration::from_millis(10),
            AgentRole::StrategyFormulation => Duration::from_millis(50),
            AgentRole::MarketIntelligence => Duration::from_millis(100),
            AgentRole::PerformanceOptimization => Duration::from_millis(200),
            AgentRole::DataAggregation => Duration::from_millis(500),
        }
    }

    pub fn required_capabilities(&self) -> Vec<SpecializedFunction> {
        match self {
            AgentRole::MarketIntelligence => vec![
                SpecializedFunction::MarketDataAnalysis,
                SpecializedFunction::SentimentAnalysis,
                SpecializedFunction::TechnicalAnalysis,
            ],
            AgentRole::StrategyFormulation => vec![
                SpecializedFunction::PredictiveModeling,
                SpecializedFunction::PortfolioOptimization,
                SpecializedFunction::FundamentalAnalysis,
            ],
            AgentRole::OrderExecution => vec![
                SpecializedFunction::OrderRouting,
                SpecializedFunction::LiquidityProvision,
            ],
            AgentRole::RiskManagement => vec![
                SpecializedFunction::RiskAssessment,
                SpecializedFunction::PredictiveModeling,
            ],
            AgentRole::EmergencyControl => vec![
                SpecializedFunction::EmergencyShutdown,
                SpecializedFunction::RiskAssessment,
            ],
            AgentRole::PerformanceOptimization => vec![
                SpecializedFunction::PerformanceBenchmarking,
                SpecializedFunction::AIModelTraining,
            ],
            AgentRole::DataAggregation => vec![
                SpecializedFunction::MarketDataAnalysis,
            ],
            AgentRole::ComplianceMonitoring => vec![
                SpecializedFunction::ComplianceValidation,
                SpecializedFunction::RiskAssessment,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_type_default_capabilities() {
        let scout = AgentType::Scout;
        let capabilities = scout.default_capabilities();
        
        assert!(capabilities.specialized_functions.contains(&SpecializedFunction::MarketDataAnalysis));
        assert!(capabilities.ai_models.len() > 0);
        assert_eq!(capabilities.real_time_performance.max_response_time_ms, 5.0);
    }

    #[test]
    fn test_agent_role_priority() {
        assert_eq!(AgentRole::EmergencyControl.priority_level(), 1);
        assert_eq!(AgentRole::DataAggregation.priority_level(), 8);
    }

    #[test]
    fn test_infrastructure_requirements() {
        let trader_reqs = AgentType::Trader.required_infrastructure();
        assert_eq!(trader_reqs.min_cpu_cores, 8);
        assert_eq!(trader_reqs.min_memory_gb, 16);
        assert!(!trader_reqs.requires_quantum);
    }
}