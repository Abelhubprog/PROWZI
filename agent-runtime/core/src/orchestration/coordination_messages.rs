use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};
use uuid::Uuid;
use crate::messages::Priority;
use super::agent_types::{AgentRole, SpecializedFunction};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    MarketIntelligence {
        symbols: Vec<String>,
        analysis_depth: AnalysisDepth,
        real_time_updates: bool,
        ai_models_required: Vec<String>,
        priority: Priority,
        deadline: Option<Instant>,
    },
    StrategyFormulation {
        portfolio_size: f64,
        risk_tolerance: RiskTolerance,
        time_horizon: TimeHorizon,
        optimization_goals: Vec<OptimizationGoal>,
        constraints: Vec<StrategyConstraint>,
        priority: Priority,
    },
    OrderExecution {
        orders: Vec<TradingOrder>,
        execution_strategy: ExecutionStrategy,
        slippage_tolerance: f64,
        urgency_level: UrgencyLevel,
        risk_checks_required: bool,
        priority: Priority,
    },
    RiskAssessment {
        portfolio_positions: Vec<Position>,
        market_conditions: MarketConditions,
        assessment_type: RiskAssessmentType,
        threshold_alerts: Vec<RiskThreshold>,
        priority: Priority,
    },
    EmergencyControl {
        emergency_type: EmergencyType,
        affected_systems: Vec<String>,
        immediate_actions: Vec<EmergencyAction>,
        escalation_required: bool,
        priority: Priority,
    },
    PerformanceOptimization {
        target_metrics: Vec<PerformanceMetric>,
        optimization_scope: OptimizationScope,
        ml_training_required: bool,
        benchmark_comparison: bool,
        priority: Priority,
    },
    SystemCoordination {
        coordination_type: CoordinationType,
        participating_agents: Vec<Uuid>,
        coordination_parameters: CoordinationParameters,
        success_criteria: Vec<SuccessCriterion>,
        priority: Priority,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationResponse {
    Success {
        agent_id: Uuid,
        data: String,
        processing_time: Duration,
        confidence_score: Option<f64>,
        recommendations: Option<Vec<Recommendation>>,
    },
    PartialSuccess {
        agent_id: Uuid,
        completed_tasks: Vec<String>,
        failed_tasks: Vec<String>,
        error_details: String,
        processing_time: Duration,
    },
    Error {
        message: String,
        error_code: Option<String>,
        recovery_suggestions: Option<Vec<String>>,
    },
    Timeout {
        agent_id: Uuid,
        partial_results: Option<String>,
        timeout_duration: Duration,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Surface,
    Standard,
    Deep,
    Comprehensive,
    UltraDeep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskTolerance {
    VeryConservative,
    Conservative,
    Moderate,
    Aggressive,
    VeryAggressive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeHorizon {
    Milliseconds(u64),
    Seconds(u64),
    Minutes(u64),
    Hours(u64),
    Days(u64),
    Weeks(u64),
    Months(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationGoal {
    MaximizeReturns,
    MinimizeRisk,
    MaximizeSharpeRatio,
    MinimizeDrawdown,
    MaximizeLiquidity,
    MinimizeFees,
    MaximizeAlpha,
    MinimizeBeta,
    CustomGoal { name: String, target_value: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyConstraint {
    MaxPositionSize(f64),
    MaxDrawdown(f64),
    MinLiquidity(f64),
    SectorExposure { sector: String, max_percentage: f64 },
    GeographicExposure { region: String, max_percentage: f64 },
    AssetClassLimit { asset_class: String, max_percentage: f64 },
    CustomConstraint { name: String, parameters: serde_json::Value },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingOrder {
    pub symbol: String,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub special_instructions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop,
    Iceberg,
    TWAP,
    VWAP,
    QuantumOptimized,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    GTC, // Good Till Canceled
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTD(Instant), // Good Till Date
    UltraFast, // Microsecond execution
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    TWAP,
    VWAP,
    MinimizeImpact,
    MaximizeSpeed,
    SmartOrder,
    LiquidityProvision,
    ArbitrageCapture,
    QuantumOptimized,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub risk_metrics: PositionRiskMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRiskMetrics {
    pub var_95: f64, // Value at Risk 95%
    pub var_99: f64, // Value at Risk 99%
    pub expected_shortfall: f64,
    pub beta: f64,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub liquidity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility_regime: VolatilityRegime,
    pub trend_direction: TrendDirection,
    pub liquidity_conditions: LiquidityConditions,
    pub correlation_environment: CorrelationEnvironment,
    pub sentiment_indicators: SentimentIndicators,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityRegime {
    VeryLow,
    Low,
    Normal,
    Elevated,
    High,
    Extreme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    StrongBull,
    WeakBull,
    Sideways,
    WeakBear,
    StrongBear,
    Undefined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityConditions {
    Abundant,
    Normal,
    Tight,
    Stressed,
    Frozen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationEnvironment {
    Diversified,
    ModeratelyCorrelated,
    HighlyCorrelated,
    CrisisCorrelation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentIndicators {
    pub fear_greed_index: f64,
    pub put_call_ratio: f64,
    pub vix_level: f64,
    pub social_sentiment: f64,
    pub institutional_flows: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskAssessmentType {
    RealTime,
    Stress,
    Scenario,
    Monte,
    Historical,
    Predictive,
    Quantum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskThreshold {
    pub metric: RiskMetric,
    pub threshold_value: f64,
    pub action: ThresholdAction,
    pub notification_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskMetric {
    VaR95,
    VaR99,
    ExpectedShortfall,
    MaxDrawdown,
    VolatilitySpike,
    ConcentrationRisk,
    LiquidityRisk,
    CounterpartyRisk,
    ModelRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdAction {
    Alert,
    ReducePosition,
    HedgePosition,
    StopTrading,
    EmergencyExit,
    EscalateToHuman,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyType {
    MarketCrash,
    SystemFailure,
    SecurityBreach,
    LiquidityCrisis,
    RegulatoryAction,
    TechnicalGlitch,
    ExternalThreat,
    DataCorruption,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAction {
    HaltTrading,
    LiquidatePositions,
    ActivateBackup,
    NotifyAuthorities,
    IsolateSystem,
    EngageFailsafe,
    EscalateToHuman,
    ActivateQuantumProtocol,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Latency,
    Throughput,
    Accuracy,
    Reliability,
    ResourceUtilization,
    PredictionAccuracy,
    RiskAdjustedReturns,
    InformationRatio,
    MaxDrawdown,
    SharpeRatio,
    CalmarRatio,
    SortinoRatio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationScope {
    SingleAgent,
    AgentGroup,
    SystemWide,
    CrossPlatform,
    QuantumEnhanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    TaskDistribution,
    ResourceSharing,
    KnowledgeSync,
    PerformanceOptimization,
    EmergencyResponse,
    LoadBalancing,
    CapacityPlanning,
    QuantumEntanglement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationParameters {
    pub timeout_duration: Duration,
    pub max_participants: u32,
    pub consensus_threshold: f64,
    pub retry_attempts: u32,
    pub priority_boost: bool,
    pub ai_assistance_level: AIAssistanceLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIAssistanceLevel {
    None,
    Basic,
    Advanced,
    Expert,
    QuantumEnhanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    pub metric: String,
    pub target_value: f64,
    pub tolerance: f64,
    pub mandatory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub confidence_score: f64,
    pub expected_impact: f64,
    pub implementation_complexity: ComplexityLevel,
    pub time_to_implement: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    ParameterAdjustment,
    StrategyModification,
    RiskMitigation,
    PerformanceImprovement,
    ResourceReallocation,
    SystemUpgrade,
    ProcessOptimization,
    QuantumEnhancement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Trivial,
    Simple,
    Moderate,
    Complex,
    HighlyComplex,
    RequiresQuantumComputing,
}

impl CoordinationMessage {
    pub fn priority(&self) -> Priority {
        match self {
            CoordinationMessage::MarketIntelligence { priority, .. } => *priority,
            CoordinationMessage::StrategyFormulation { priority, .. } => *priority,
            CoordinationMessage::OrderExecution { priority, .. } => *priority,
            CoordinationMessage::RiskAssessment { priority, .. } => *priority,
            CoordinationMessage::EmergencyControl { priority, .. } => *priority,
            CoordinationMessage::PerformanceOptimization { priority, .. } => *priority,
            CoordinationMessage::SystemCoordination { priority, .. } => *priority,
        }
    }

    pub fn message_type(&self) -> &'static str {
        match self {
            CoordinationMessage::MarketIntelligence { .. } => "MarketIntelligence",
            CoordinationMessage::StrategyFormulation { .. } => "StrategyFormulation",
            CoordinationMessage::OrderExecution { .. } => "OrderExecution",
            CoordinationMessage::RiskAssessment { .. } => "RiskAssessment",
            CoordinationMessage::EmergencyControl { .. } => "EmergencyControl",
            CoordinationMessage::PerformanceOptimization { .. } => "PerformanceOptimization",
            CoordinationMessage::SystemCoordination { .. } => "SystemCoordination",
        }
    }

    pub fn required_role(&self) -> AgentRole {
        match self {
            CoordinationMessage::MarketIntelligence { .. } => AgentRole::MarketIntelligence,
            CoordinationMessage::StrategyFormulation { .. } => AgentRole::StrategyFormulation,
            CoordinationMessage::OrderExecution { .. } => AgentRole::OrderExecution,
            CoordinationMessage::RiskAssessment { .. } => AgentRole::RiskManagement,
            CoordinationMessage::EmergencyControl { .. } => AgentRole::EmergencyControl,
            CoordinationMessage::PerformanceOptimization { .. } => AgentRole::PerformanceOptimization,
            CoordinationMessage::SystemCoordination { .. } => AgentRole::PerformanceOptimization,
        }
    }

    pub fn required_capabilities(&self) -> Vec<SpecializedFunction> {
        match self {
            CoordinationMessage::MarketIntelligence { ai_models_required, .. } => {
                let mut capabilities = vec![
                    SpecializedFunction::MarketDataAnalysis,
                    SpecializedFunction::SentimentAnalysis,
                ];
                if ai_models_required.contains(&"technical_analysis".to_string()) {
                    capabilities.push(SpecializedFunction::TechnicalAnalysis);
                }
                if ai_models_required.contains(&"fundamental_analysis".to_string()) {
                    capabilities.push(SpecializedFunction::FundamentalAnalysis);
                }
                capabilities
            },
            CoordinationMessage::StrategyFormulation { optimization_goals, .. } => {
                let mut capabilities = vec![
                    SpecializedFunction::PredictiveModeling,
                    SpecializedFunction::PortfolioOptimization,
                ];
                if optimization_goals.iter().any(|goal| matches!(goal, OptimizationGoal::CustomGoal { .. })) {
                    capabilities.push(SpecializedFunction::QuantumOptimization);
                }
                capabilities
            },
            CoordinationMessage::OrderExecution { execution_strategy, .. } => {
                let mut capabilities = vec![SpecializedFunction::OrderRouting];
                match execution_strategy {
                    ExecutionStrategy::LiquidityProvision => {
                        capabilities.push(SpecializedFunction::LiquidityProvision);
                    },
                    ExecutionStrategy::ArbitrageCapture => {
                        capabilities.push(SpecializedFunction::ArbitrageDetection);
                    },
                    ExecutionStrategy::QuantumOptimized => {
                        capabilities.push(SpecializedFunction::QuantumOptimization);
                    },
                    _ => {},
                }
                capabilities
            },
            CoordinationMessage::RiskAssessment { assessment_type, .. } => {
                let mut capabilities = vec![SpecializedFunction::RiskAssessment];
                match assessment_type {
                    RiskAssessmentType::Predictive => {
                        capabilities.push(SpecializedFunction::PredictiveModeling);
                    },
                    RiskAssessmentType::Quantum => {
                        capabilities.push(SpecializedFunction::QuantumOptimization);
                    },
                    _ => {},
                }
                capabilities
            },
            CoordinationMessage::EmergencyControl { .. } => {
                vec![
                    SpecializedFunction::EmergencyShutdown,
                    SpecializedFunction::RiskAssessment,
                ]
            },
            CoordinationMessage::PerformanceOptimization { ml_training_required, .. } => {
                let mut capabilities = vec![SpecializedFunction::PerformanceBenchmarking];
                if *ml_training_required {
                    capabilities.push(SpecializedFunction::AIModelTraining);
                }
                capabilities
            },
            CoordinationMessage::SystemCoordination { coordination_type, .. } => {
                match coordination_type {
                    CoordinationType::QuantumEntanglement => {
                        vec![SpecializedFunction::QuantumOptimization]
                    },
                    _ => vec![SpecializedFunction::PerformanceBenchmarking],
                }
            },
        }
    }

    pub fn estimated_processing_time(&self) -> Duration {
        match self {
            CoordinationMessage::MarketIntelligence { analysis_depth, .. } => {
                match analysis_depth {
                    AnalysisDepth::Surface => Duration::from_millis(10),
                    AnalysisDepth::Standard => Duration::from_millis(50),
                    AnalysisDepth::Deep => Duration::from_millis(200),
                    AnalysisDepth::Comprehensive => Duration::from_millis(500),
                    AnalysisDepth::UltraDeep => Duration::from_secs(2),
                }
            },
            CoordinationMessage::StrategyFormulation { .. } => Duration::from_millis(100),
            CoordinationMessage::OrderExecution { urgency_level, .. } => {
                match urgency_level {
                    UrgencyLevel::Emergency => Duration::from_millis(1),
                    UrgencyLevel::Critical => Duration::from_millis(5),
                    UrgencyLevel::High => Duration::from_millis(10),
                    UrgencyLevel::Medium => Duration::from_millis(50),
                    UrgencyLevel::Low => Duration::from_millis(100),
                }
            },
            CoordinationMessage::RiskAssessment { assessment_type, .. } => {
                match assessment_type {
                    RiskAssessmentType::RealTime => Duration::from_millis(2),
                    RiskAssessmentType::Stress => Duration::from_millis(100),
                    RiskAssessmentType::Scenario => Duration::from_millis(200),
                    RiskAssessmentType::Monte => Duration::from_millis(500),
                    RiskAssessmentType::Historical => Duration::from_millis(300),
                    RiskAssessmentType::Predictive => Duration::from_millis(150),
                    RiskAssessmentType::Quantum => Duration::from_millis(50),
                }
            },
            CoordinationMessage::EmergencyControl { .. } => Duration::from_millis(1),
            CoordinationMessage::PerformanceOptimization { .. } => Duration::from_millis(200),
            CoordinationMessage::SystemCoordination { .. } => Duration::from_millis(100),
        }
    }

    pub fn resource_requirements(&self) -> ResourceRequirements {
        match self {
            CoordinationMessage::MarketIntelligence { analysis_depth, .. } => {
                ResourceRequirements {
                    cpu_intensity: match analysis_depth {
                        AnalysisDepth::Surface => 0.2,
                        AnalysisDepth::Standard => 0.4,
                        AnalysisDepth::Deep => 0.7,
                        AnalysisDepth::Comprehensive => 0.9,
                        AnalysisDepth::UltraDeep => 1.0,
                    },
                    memory_mb: 512,
                    network_bandwidth_mbps: 100,
                    gpu_required: true,
                    quantum_required: false,
                }
            },
            CoordinationMessage::StrategyFormulation { .. } => {
                ResourceRequirements {
                    cpu_intensity: 0.8,
                    memory_mb: 1024,
                    network_bandwidth_mbps: 50,
                    gpu_required: true,
                    quantum_required: true,
                }
            },
            CoordinationMessage::OrderExecution { .. } => {
                ResourceRequirements {
                    cpu_intensity: 0.6,
                    memory_mb: 256,
                    network_bandwidth_mbps: 1000,
                    gpu_required: false,
                    quantum_required: false,
                }
            },
            CoordinationMessage::RiskAssessment { .. } => {
                ResourceRequirements {
                    cpu_intensity: 0.7,
                    memory_mb: 512,
                    network_bandwidth_mbps: 200,
                    gpu_required: true,
                    quantum_required: false,
                }
            },
            CoordinationMessage::EmergencyControl { .. } => {
                ResourceRequirements {
                    cpu_intensity: 1.0,
                    memory_mb: 2048,
                    network_bandwidth_mbps: 2000,
                    gpu_required: true,
                    quantum_required: true,
                }
            },
            CoordinationMessage::PerformanceOptimization { .. } => {
                ResourceRequirements {
                    cpu_intensity: 0.5,
                    memory_mb: 1024,
                    network_bandwidth_mbps: 100,
                    gpu_required: true,
                    quantum_required: false,
                }
            },
            CoordinationMessage::SystemCoordination { .. } => {
                ResourceRequirements {
                    cpu_intensity: 0.4,
                    memory_mb: 512,
                    network_bandwidth_mbps: 500,
                    gpu_required: false,
                    quantum_required: false,
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_intensity: f64, // 0.0 to 1.0
    pub memory_mb: u32,
    pub network_bandwidth_mbps: u32,
    pub gpu_required: bool,
    pub quantum_required: bool,
}

impl CoordinationResponse {
    pub fn processing_time(&self) -> Option<Duration> {
        match self {
            CoordinationResponse::Success { processing_time, .. } => Some(*processing_time),
            CoordinationResponse::PartialSuccess { processing_time, .. } => Some(*processing_time),
            CoordinationResponse::Timeout { timeout_duration, .. } => Some(*timeout_duration),
            CoordinationResponse::Error { .. } => None,
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self, CoordinationResponse::Success { .. })
    }

    pub fn agent_id(&self) -> Option<Uuid> {
        match self {
            CoordinationResponse::Success { agent_id, .. } => Some(*agent_id),
            CoordinationResponse::PartialSuccess { agent_id, .. } => Some(*agent_id),
            CoordinationResponse::Timeout { agent_id, .. } => Some(*agent_id),
            CoordinationResponse::Error { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordination_message_priority() {
        let msg = CoordinationMessage::EmergencyControl {
            emergency_type: EmergencyType::MarketCrash,
            affected_systems: vec!["trading".to_string()],
            immediate_actions: vec![EmergencyAction::HaltTrading],
            escalation_required: true,
            priority: Priority::Critical,
        };
        
        assert_eq!(msg.priority(), Priority::Critical);
        assert_eq!(msg.message_type(), "EmergencyControl");
    }

    #[test]
    fn test_resource_requirements() {
        let msg = CoordinationMessage::MarketIntelligence {
            symbols: vec!["BTC/USD".to_string()],
            analysis_depth: AnalysisDepth::Deep,
            real_time_updates: true,
            ai_models_required: vec!["technical_analysis".to_string()],
            priority: Priority::High,
            deadline: None,
        };
        
        let requirements = msg.resource_requirements();
        assert_eq!(requirements.cpu_intensity, 0.7);
        assert!(requirements.gpu_required);
    }

    #[test]
    fn test_coordination_response() {
        let response = CoordinationResponse::Success {
            agent_id: Uuid::new_v4(),
            data: "Test data".to_string(),
            processing_time: Duration::from_millis(50),
            confidence_score: Some(0.95),
            recommendations: None,
        };
        
        assert!(response.is_success());
        assert_eq!(response.processing_time(), Some(Duration::from_millis(50)));
    }
}