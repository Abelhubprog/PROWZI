// Gateway Integration Example for Prowzi Auth
// This shows how to integrate the authentication service with the main gateway

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
    routing::{get, post},
    Json, Router,
};
use prowzi_auth::{
    AuthService, AuthConfig, AuthMiddleware, 
    middleware::{auth_middleware, optional_auth_middleware, rate_limit_middleware},
    errors::AuthError,
    models::Claims,
};
use std::sync::Arc;
use sqlx::PgPool;
use redis::Client as RedisClient;

#[derive(Clone)]
pub struct GatewayState {
    pub auth_service: Arc<AuthService>,
    pub auth_middleware: Arc<AuthMiddleware>,
    pub db: Arc<PgPool>,
    pub redis: Arc<RedisClient>,
}

pub async fn create_gateway_with_auth() -> Result<Router, Box<dyn std::error::Error>> {
    // Load configuration
    let auth_config = AuthConfig::from_env()?;
    
    // Initialize services
    let auth_service = Arc::new(AuthService::new(auth_config.clone()).await?);
    let db = Arc::new(create_db_pool(&auth_config.database_url).await?);
    let redis = Arc::new(RedisClient::open(auth_config.redis_url.clone())?);
    
    let auth_middleware = Arc::new(AuthMiddleware::new(
        &auth_config.jwt_public_key,
        redis.clone(),
    )?);

    let state = GatewayState {
        auth_service,
        auth_middleware,
        db,
        redis,
    };

    // Create router with authentication
    let app = Router::new()
        // Public routes (no auth required)
        .route("/health", get(health_check))
        .route("/auth/*path", any(proxy_to_auth_service))
        
        // Protected routes (auth required)
        .route("/api/missions", get(get_missions).post(create_mission))
        .route("/api/missions/:id", get(get_mission).put(update_mission))
        .route("/api/briefs", get(get_briefs))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            protected_route_middleware,
        ))
        
        // Optional auth routes (works with or without auth)
        .route("/api/public/stats", get(get_public_stats))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            optional_auth_route_middleware,
        ))
        
        // Apply rate limiting to all routes
        .layer(axum::middleware::from_fn_with_state(
            state.redis.clone(),
            rate_limit_middleware,
        ))
        
        .with_state(state);

    Ok(app)
}

// Middleware for protected routes
async fn protected_route_middleware(
    State(state): State<GatewayState>,
    request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    auth_middleware(
        State(state.auth_middleware),
        State(state.db),
        request,
        next,
    ).await
}

// Middleware for optional auth routes
async fn optional_auth_route_middleware(
    State(state): State<GatewayState>,
    request: Request,
    next: Next,
) -> Response {
    optional_auth_middleware(
        State(state.auth_middleware),
        State(state.db),
        request,
        next,
    ).await
}

// Example protected route handlers
async fn get_missions(
    headers: HeaderMap,
    State(state): State<GatewayState>,
) -> Result<Json<serde_json::Value>, AuthError> {
    // Claims are automatically injected by auth middleware
    let claims = extract_claims_from_headers(&headers)?;
    
    tracing::info!("Getting missions for user: {} tenant: {}", claims.sub, claims.tenant);
    
    // Query missions with tenant isolation automatically enforced by RLS
    let missions = sqlx::query_as::<_, Mission>(
        "SELECT * FROM missions ORDER BY created_at DESC LIMIT 10"
    )
    .fetch_all(&**state.db)
    .await
    .map_err(AuthError::DatabaseError)?;

    Ok(Json(serde_json::json!({
        "missions": missions,
        "user_id": claims.sub,
        "tenant_id": claims.tenant
    })))
}

async fn create_mission(
    headers: HeaderMap,
    State(state): State<GatewayState>,
    Json(payload): Json<CreateMissionRequest>,
) -> Result<Json<serde_json::Value>, AuthError> {
    let claims = extract_claims_from_headers(&headers)?;
    
    // Check if user has write permissions
    if !claims.scope.contains(&"write".to_string()) {
        return Err(AuthError::AccessDenied);
    }

    // Create mission (tenant_id automatically set by RLS)
    let mission_id = uuid::Uuid::new_v4();
    sqlx::query(
        "INSERT INTO missions (id, title, description, user_id, tenant_id) VALUES ($1, $2, $3, $4, $5)"
    )
    .bind(mission_id)
    .bind(&payload.title)
    .bind(&payload.description)
    .bind(uuid::Uuid::parse_str(&claims.sub).unwrap())
    .bind(&claims.tenant)
    .execute(&**state.db)
    .await
    .map_err(AuthError::DatabaseError)?;

    Ok(Json(serde_json::json!({
        "id": mission_id,
        "status": "created"
    })))
}

// Proxy requests to auth service
async fn proxy_to_auth_service(
    State(state): State<GatewayState>,
    request: Request,
) -> Response {
    // Forward request to auth service
    // In production, use a proper reverse proxy
    Response::builder()
        .status(StatusCode::NOT_IMPLEMENTED)
        .body("Auth proxy not implemented".into())
        .unwrap()
}

// Example models
#[derive(serde::Deserialize)]
struct CreateMissionRequest {
    title: String,
    description: String,
}

#[derive(serde::Serialize, sqlx::FromRow)]
struct Mission {
    id: uuid::Uuid,
    title: String,
    description: String,
    created_at: chrono::DateTime<chrono::Utc>,
}

// Utility functions
fn extract_claims_from_headers(headers: &HeaderMap) -> Result<Claims, AuthError> {
    // Extract claims from request extensions (set by auth middleware)
    // This is a simplified version - in practice, use request extensions
    Err(AuthError::AccessDenied)
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "prowzi-gateway",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn get_public_stats() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "total_missions": 1337,
        "active_users": 42
    }))
}

async fn create_db_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    sqlx::postgres::PgPoolOptions::new()
        .max_connections(20)
        .connect(database_url)
        .await
}
