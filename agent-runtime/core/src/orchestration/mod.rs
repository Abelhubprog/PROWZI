//! Quantum Orchestration Module
//! 
//! Advanced multi-agent coordination system with AI-enhanced capabilities,
//! self-healing infrastructure, and quantum-speed communication protocols.

pub mod quantum_orchestrator;
pub mod agent_types;
pub mod coordination_messages;
pub mod performance_monitor;

pub use quantum_orchestrator::*;
pub use agent_types::*;
pub use coordination_messages::*;
pub use performance_monitor::*;

/// Re-export for convenience
pub type Orchestrator = QuantumOrchestrator;