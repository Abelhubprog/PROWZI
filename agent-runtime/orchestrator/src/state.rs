
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Mission, budget::BudgetManager, scheduler::MissionScheduler};

pub struct AppState {
    pub budget_manager: Arc<BudgetManager>,
    pub mission_store: Arc<tokio::sync::RwLock<MissionStore>>,
    pub scheduler: Arc<MissionScheduler>,
}

pub struct MissionStore {
    pub missions: HashMap<String, Mission>,
}

impl MissionStore {
    pub fn new() -> Self {
        Self {
            missions: HashMap::new(),
        }
    }
    
    pub fn get(&self, mission_id: &str) -> Option<&Mission> {
        self.missions.get(mission_id)
    }
    
    pub fn insert(&mut self, mission: Mission) {
        self.missions.insert(mission.id.clone(), mission);
    }
    
    pub fn update_status(&mut self, mission_id: &str, status: crate::MissionStatus) {
        if let Some(mission) = self.missions.get_mut(mission_id) {
            mission.status = status;
            mission.updated_at = chrono::Utc::now();
        }
    }
}
