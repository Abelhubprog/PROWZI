use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

mod budget;
mod scheduler;
mod state;

use budget::{BudgetManager, TokenBucket};
use scheduler::MissionScheduler;
use state::{AppState, MissionStore};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Initialize components
    let budget_manager = Arc::new(BudgetManager::new());
    let mission_store = Arc::new(RwLock::new(MissionStore::new()));
    let scheduler = Arc::new(MissionScheduler::new());

    let app_state = Arc::new(AppState {
        budget_manager,
        mission_store,
        scheduler,
    });

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/missions/:id/spawn", post(spawn_agents))
        .route("/missions/:id/throttle", post(throttle_mission))
        .route("/missions/:id/budget", get(get_budget_status))
        .route("/metrics", get(metrics))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(RequestBodyLimitLayer::new(5 * 1024 * 1024)), // 5MB limit
        )
        .with_state(app_state);

    // Start server
    let addr = "0.0.0.0:8080".parse().unwrap();
    tracing::info!("Orchestrator listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn spawn_agents(
    Path(mission_id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(request): Json<SpawnRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Validate mission exists
    let missions = state.mission_store.read().await;
    let mission = missions
        .get(&mission_id)
        .ok_or(AppError::MissionNotFound)?;

    // Check budget
    let budget_available = state.budget_manager
        .check_budget(&mission_id, &request.resource_requirements)
        .await?;

    if !budget_available {
        return Err(AppError::BudgetExceeded);
    }

    // Reserve budget
    state.budget_manager
        .reserve_budget(&mission_id, &request.resource_requirements)
        .await?;

    // Schedule spawn
    let spawn_result = state.scheduler
        .schedule_spawn(&mission_id, request)
        .await?;

    // Update metrics
    metrics::AGENTS_SPAWNED.inc();

    Ok(Json(spawn_result))
}

async fn throttle_mission(
    Path(mission_id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ThrottleRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Update budget limits
    state.budget_manager
        .update_limits(&mission_id, request.new_limits)
        .await?;

    // Apply throttling
    state.scheduler
        .throttle_mission(&mission_id, request.throttle_factor)
        .await?;

    Ok(StatusCode::OK)
}

async fn get_budget_status(
    Path(mission_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, AppError> {
    let status = state.budget_manager
        .get_budget_status(&mission_id)
        .await?;

    Ok(Json(status))
}

async fn metrics() -> impl IntoResponse {
    let metrics = prometheus::TextEncoder::new()
        .encode_to_string(&prometheus::gather())
        .unwrap();

    metrics
}
agent-runtime/orchestrator/src/budget.rs:

rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct TokenBucket {
    capacity: u64,
    tokens: u64,
    refill_rate: u64, // tokens per second
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

    pub async fn check_budget(
        &self,
        mission_id: &str,
        requirements: &ResourceRequirements,
    ) -> Result<bool, BudgetError> {
        let mut budgets = self.budgets.write().await;

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

        // Record metrics
        metrics::BUDGET_CONSUMED
            .with_label_values(&[mission_id, "cpu_ms"])
            .inc_by(requirements.cpu_ms as f64);

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
}

// Prometheus metrics
lazy_static! {
    static ref BUDGET_CONSUMED: prometheus::CounterVec = prometheus::register_counter_vec!(
        "prowzi_budget_consumed_total",
        "Total budget consumed by resource type",
        &["mission_id", "resource_type"]
    ).unwrap();

    static ref BUDGET_REJECTIONS: prometheus::CounterVec = prometheus::register_counter_vec!(
        "prowzi_budget_rejections_total",
        "Total budget rejections by reason",
        &["mission_id", "reason"]
    ).unwrap();
}
