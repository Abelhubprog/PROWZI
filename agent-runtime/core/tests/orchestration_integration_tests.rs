use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;
use prowzi_core::{
    orchestration::{
        QuantumOrchestrator, AgentHandle, AgentType, AgentRole, AgentCapabilities,
        CoordinationMessage, CoordinationResponse, AnalysisDepth, Priority,
        HealthStatus, AgentPerformanceSnapshot, CoordinationEvent,
        CoordinationEventType, EventPerformanceMetrics
    },
    messages::Priority as MessagePriority,
};

#[tokio::test]
async fn test_orchestrator_initialization() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    
    // Test that orchestrator starts with empty registry
    let metrics = orchestrator.get_orchestration_metrics().await;
    assert!(metrics.is_err()); // No metrics initially
    
    // Clean shutdown
    shutdown_tx.send(true).unwrap();
}

#[tokio::test]
async fn test_agent_registration() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    
    let agent_handle = AgentHandle {
        id: Uuid::new_v4(),
        agent_type: AgentType::Scout,
        role: AgentRole::MarketIntelligence,
        capabilities: AgentCapabilities::default(),
        performance_score: 0.85,
        health_status: HealthStatus::Healthy,
        last_heartbeat: std::time::Instant::now(),
        coordination_tx: tx,
    };
    
    // Register agent
    let result = orchestrator.register_agent(agent_handle).await;
    assert!(result.is_ok());
    
    // Clean shutdown
    shutdown_tx.send(true).unwrap();
}

#[tokio::test]
async fn test_coordination_task_execution() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    
    // Register a scout agent
    let scout_agent = AgentHandle {
        id: Uuid::new_v4(),
        agent_type: AgentType::Scout,
        role: AgentRole::MarketIntelligence,
        capabilities: AgentCapabilities::default(),
        performance_score: 0.90,
        health_status: HealthStatus::Healthy,
        last_heartbeat: std::time::Instant::now(),
        coordination_tx: tx.clone(),
    };
    
    orchestrator.register_agent(scout_agent).await.unwrap();
    
    // Create a market intelligence coordination task
    let coordination_task = CoordinationMessage::MarketIntelligence {
        symbols: vec!["BTC/USD".to_string(), "ETH/USD".to_string()],
        analysis_depth: AnalysisDepth::Standard,
        real_time_updates: true,
        ai_models_required: vec!["technical_analysis".to_string()],
        priority: Priority::High,
        deadline: None,
    };
    
    // Execute coordination task
    let response = orchestrator.coordinate_task(coordination_task).await;
    assert!(response.is_ok());
    
    let coordination_response = response.unwrap();
    match coordination_response {
        CoordinationResponse::Success { agent_id, data, processing_time, .. } => {
            assert!(!data.is_empty());
            assert!(processing_time > Duration::from_millis(0));
            println!("Coordination successful: Agent {} processed in {:?}", agent_id, processing_time);
        },
        CoordinationResponse::Error { message } => {
            println!("Coordination error: {}", message);
        },
        _ => {},
    }
    
    // Clean shutdown
    shutdown_tx.send(true).unwrap();
}

#[tokio::test]
async fn test_multi_agent_coordination() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    
    // Register multiple agents with different types
    let agent_types = vec![
        (AgentType::Scout, AgentRole::MarketIntelligence, 0.88),
        (AgentType::Planner, AgentRole::StrategyFormulation, 0.92),
        (AgentType::Trader, AgentRole::OrderExecution, 0.85),
        (AgentType::RiskSentinel, AgentRole::RiskManagement, 0.95),
    ];
    
    for (agent_type, role, performance_score) in agent_types {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let agent = AgentHandle {
            id: Uuid::new_v4(),
            agent_type,
            role,
            capabilities: agent_type.default_capabilities(),
            performance_score,
            health_status: HealthStatus::Healthy,
            last_heartbeat: std::time::Instant::now(),
            coordination_tx: tx,
        };
        
        orchestrator.register_agent(agent).await.unwrap();
    }
    
    // Test different coordination scenarios
    let coordination_tasks = vec![
        CoordinationMessage::MarketIntelligence {
            symbols: vec!["BTC/USD".to_string()],
            analysis_depth: AnalysisDepth::Deep,
            real_time_updates: true,
            ai_models_required: vec!["sentiment_analysis".to_string()],
            priority: Priority::High,
            deadline: None,
        },
        CoordinationMessage::RiskAssessment {
            portfolio_positions: vec![],
            market_conditions: prowzi_core::orchestration::MarketConditions {
                volatility_regime: prowzi_core::orchestration::VolatilityRegime::Elevated,
                trend_direction: prowzi_core::orchestration::TrendDirection::WeakBull,
                liquidity_conditions: prowzi_core::orchestration::LiquidityConditions::Normal,
                correlation_environment: prowzi_core::orchestration::CorrelationEnvironment::ModeratelyCorrelated,
                sentiment_indicators: prowzi_core::orchestration::SentimentIndicators {
                    fear_greed_index: 45.0,
                    put_call_ratio: 1.2,
                    vix_level: 22.5,
                    social_sentiment: 0.6,
                    institutional_flows: 0.3,
                },
            },
            assessment_type: prowzi_core::orchestration::RiskAssessmentType::RealTime,
            threshold_alerts: vec![],
            priority: Priority::Critical,
        },
    ];
    
    for task in coordination_tasks {
        let response = orchestrator.coordinate_task(task).await;
        assert!(response.is_ok());
        
        match response.unwrap() {
            CoordinationResponse::Success { processing_time, .. } => {
                println!("Multi-agent coordination completed in {:?}", processing_time);
                assert!(processing_time < Duration::from_millis(100)); // Should be fast
            },
            _ => println!("Coordination had non-success response"),
        }
    }
    
    // Clean shutdown
    shutdown_tx.send(true).unwrap();
}

#[tokio::test]
async fn test_agent_health_monitoring() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    
    let agent_id = Uuid::new_v4();
    let agent = AgentHandle {
        id: agent_id,
        agent_type: AgentType::Guardian,
        role: AgentRole::EmergencyControl,
        capabilities: AgentCapabilities::default(),
        performance_score: 0.98,
        health_status: HealthStatus::Healthy,
        last_heartbeat: std::time::Instant::now(),
        coordination_tx: tx,
    };
    
    orchestrator.register_agent(agent).await.unwrap();
    
    // Wait briefly to allow metrics collection
    sleep(Duration::from_millis(100)).await;
    
    // Get orchestration metrics
    let metrics_result = orchestrator.get_orchestration_metrics().await;
    if let Ok(metrics) = metrics_result {
        assert_eq!(metrics.total_agents, 1);
        assert_eq!(metrics.healthy_agents, 1);
        assert_eq!(metrics.degraded_agents, 0);
        assert_eq!(metrics.critical_agents, 0);
        assert!(metrics.agent_performance_scores.contains_key(&agent_id));
        assert_eq!(metrics.agent_performance_scores[&agent_id], 0.98);
        
        println!("Health monitoring metrics: {:?}", metrics);
    }
    
    // Clean shutdown
    shutdown_tx.send(true).unwrap();
}

#[tokio::test]
async fn test_performance_monitoring() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    
    // Create and record a performance snapshot
    let agent_id = Uuid::new_v4();
    let snapshot = AgentPerformanceSnapshot {
        agent_id,
        timestamp: std::time::SystemTime::now(),
        response_time_ms: 15.5,
        success_rate: 0.96,
        throughput: 1500.0,
        resource_efficiency: 0.85,
        prediction_accuracy: 0.91,
        health_score: 0.93,
        task_completion_rate: 0.98,
        error_count: 2,
        quantum_operations: 25,
    };
    
    // Record the snapshot (this would normally be done by the performance monitor)
    // For testing, we'll access the performance monitor directly
    let performance_monitor = &orchestrator.performance_monitor;
    let record_result = performance_monitor.record_agent_snapshot(snapshot).await;
    assert!(record_result.is_ok());
    
    // Create and record a coordination event
    let event = CoordinationEvent {
        event_id: Uuid::new_v4(),
        timestamp: std::time::SystemTime::now(),
        event_type: CoordinationEventType::TaskDistribution,
        participating_agents: vec![agent_id],
        duration: Duration::from_millis(25),
        success: true,
        performance_metrics: EventPerformanceMetrics {
            latency_ms: 25.0,
            throughput: 2000.0,
            resource_consumption: 0.4,
            ai_assistance_effectiveness: 0.88,
            quantum_speedup_factor: 1.3,
        },
    };
    
    let event_record_result = performance_monitor.record_coordination_event(event).await;
    assert!(event_record_result.is_ok());
    
    // Test getting agent performance trend
    let trend_result = performance_monitor.get_agent_performance_trend(
        agent_id, 
        Duration::from_secs(3600)
    ).await;
    assert!(trend_result.is_ok());
    
    let trend = trend_result.unwrap();
    assert_eq!(trend.len(), 1); // Should have one snapshot
    assert_eq!(trend[0].agent_id, agent_id);
    assert_eq!(trend[0].response_time_ms, 15.5);
    
    println!("Performance trend recorded successfully: {} snapshots", trend.len());
    
    // Clean shutdown
    shutdown_tx.send(true).unwrap();
}

#[tokio::test]
async fn test_emergency_coordination() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    
    // Register a guardian agent for emergency control
    let guardian_agent = AgentHandle {
        id: Uuid::new_v4(),
        agent_type: AgentType::Guardian,
        role: AgentRole::EmergencyControl,
        capabilities: AgentType::Guardian.default_capabilities(),
        performance_score: 0.99,
        health_status: HealthStatus::Healthy,
        last_heartbeat: std::time::Instant::now(),
        coordination_tx: tx,
    };
    
    orchestrator.register_agent(guardian_agent).await.unwrap();
    
    // Create an emergency coordination message
    let emergency_task = CoordinationMessage::EmergencyControl {
        emergency_type: prowzi_core::orchestration::EmergencyType::MarketCrash,
        affected_systems: vec!["trading_engine".to_string(), "risk_monitor".to_string()],
        immediate_actions: vec![
            prowzi_core::orchestration::EmergencyAction::HaltTrading,
            prowzi_core::orchestration::EmergencyAction::ActivateBackup,
        ],
        escalation_required: true,
        priority: Priority::Critical,
    };
    
    // Execute emergency coordination
    let response = orchestrator.coordinate_task(emergency_task).await;
    assert!(response.is_ok());
    
    match response.unwrap() {
        CoordinationResponse::Success { processing_time, .. } => {
            println!("Emergency coordination completed in {:?}", processing_time);
            // Emergency coordination should be very fast
            assert!(processing_time < Duration::from_millis(10));
        },
        CoordinationResponse::Error { message } => {
            panic!("Emergency coordination failed: {}", message);
        },
        _ => {},
    }
    
    // Clean shutdown
    shutdown_tx.send(true).unwrap();
}

#[tokio::test]
async fn test_agent_type_capabilities() {
    // Test that each agent type has appropriate capabilities
    let scout_caps = AgentType::Scout.default_capabilities();
    assert!(scout_caps.specialized_functions.contains(&prowzi_core::orchestration::SpecializedFunction::MarketDataAnalysis));
    assert!(!scout_caps.ai_models.is_empty());
    assert_eq!(scout_caps.real_time_performance.max_response_time_ms, 5.0);
    
    let planner_caps = AgentType::Planner.default_capabilities();
    assert!(planner_caps.specialized_functions.contains(&prowzi_core::orchestration::SpecializedFunction::PortfolioOptimization));
    assert!(planner_caps.processing_power.quantum_processing);
    assert_eq!(planner_caps.real_time_performance.max_response_time_ms, 20.0);
    
    let trader_caps = AgentType::Trader.default_capabilities();
    assert!(trader_caps.specialized_functions.contains(&prowzi_core::orchestration::SpecializedFunction::OrderRouting));
    assert_eq!(trader_caps.network_bandwidth.latency_microseconds, 50);
    assert_eq!(trader_caps.real_time_performance.max_response_time_ms, 2.0);
    
    let risk_sentinel_caps = AgentType::RiskSentinel.default_capabilities();
    assert!(risk_sentinel_caps.specialized_functions.contains(&prowzi_core::orchestration::SpecializedFunction::RiskAssessment));
    assert_eq!(risk_sentinel_caps.real_time_performance.max_response_time_ms, 1.0);
    
    let guardian_caps = AgentType::Guardian.default_capabilities();
    assert!(guardian_caps.specialized_functions.contains(&prowzi_core::orchestration::SpecializedFunction::EmergencyShutdown));
    assert_eq!(guardian_caps.real_time_performance.max_response_time_ms, 0.5);
    assert!(guardian_caps.network_bandwidth.quantum_entanglement_simulation);
    
    println!("All agent type capabilities validated successfully");
}

#[tokio::test]
async fn test_coordination_message_routing() {
    // Test that coordination messages are routed to appropriate agent types
    use prowzi_core::orchestration::{CoordinationMessage, AnalysisDepth, Priority};
    
    let market_intel_msg = CoordinationMessage::MarketIntelligence {
        symbols: vec!["BTC/USD".to_string()],
        analysis_depth: AnalysisDepth::Standard,
        real_time_updates: true,
        ai_models_required: vec!["technical_analysis".to_string()],
        priority: Priority::High,
        deadline: None,
    };
    
    assert_eq!(market_intel_msg.required_role(), AgentRole::MarketIntelligence);
    assert_eq!(market_intel_msg.message_type(), "MarketIntelligence");
    assert_eq!(market_intel_msg.priority(), Priority::High);
    
    let required_caps = market_intel_msg.required_capabilities();
    assert!(required_caps.contains(&prowzi_core::orchestration::SpecializedFunction::MarketDataAnalysis));
    assert!(required_caps.contains(&prowzi_core::orchestration::SpecializedFunction::TechnicalAnalysis));
    
    let estimated_time = market_intel_msg.estimated_processing_time();
    assert_eq!(estimated_time, Duration::from_millis(50)); // Standard analysis
    
    let resource_reqs = market_intel_msg.resource_requirements();
    assert_eq!(resource_reqs.cpu_intensity, 0.4);
    assert!(resource_reqs.gpu_required);
    
    println!("Coordination message routing validated successfully");
}

#[tokio::test]
async fn test_orchestrator_shutdown_gracefully() {
    let (orchestrator, shutdown_tx) = QuantumOrchestrator::new();
    
    // Start orchestrator in background
    let orchestrator_handle = tokio::spawn(async move {
        orchestrator.start().await
    });
    
    // Wait briefly
    sleep(Duration::from_millis(50)).await;
    
    // Send shutdown signal
    shutdown_tx.send(true).unwrap();
    
    // Wait for graceful shutdown
    let result = tokio::time::timeout(Duration::from_secs(5), orchestrator_handle).await;
    assert!(result.is_ok());
    
    let orchestrator_result = result.unwrap();
    assert!(orchestrator_result.is_ok());
    
    println!("Orchestrator shutdown gracefully");
}