//! Prowzi Core Library
//! 
//! Core components for the Prowzi autonomous AI agent runtime, including
//! actor system, budget management, collaboration coordination, and
//! zero-copy message processing.

pub mod actor;
pub mod budget;
pub mod collaboration;
pub mod gpu;
pub mod messages;
pub mod mission;
pub mod orchestrator;
pub mod performance;
pub mod risk;
pub mod security;
pub mod zero_copy;

pub use actor::*;
pub use budget::*;
pub use collaboration::*;
pub use gpu::*;
pub use messages::*;
pub use mission::*;
pub use orchestrator::*;
pub use performance::*;
pub use risk::*;
pub use security::*;
pub use zero_copy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_library() {
        // Basic test to ensure the library compiles
        assert!(true);
    }
}
