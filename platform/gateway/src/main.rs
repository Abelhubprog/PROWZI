//! Prowzi API Gateway
//!
//! Central gateway service for the Prowzi platform that handles routing, authentication,
//! rate limiting, and load balancing for all internal services. Built with Axum for
//! high-performance async HTTP handling.

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Json},
    routing::{get, post, put, delete, any},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::{ServiceBuilder, limit::ConcurrencyLimitLayer};
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
    limit::RequestBodyLimitLayer,
};
use uuid::Uuid;
use std::time::{Duration, Instant};

// Import our new auth service
use prowzi_auth::{
    AuthService, AuthConfig, AuthMiddleware,
    middleware::{auth_middleware, optional_auth_middleware, rate_limit_middleware},
    errors::AuthError,
    models::Claims,
};

mod routing;
mod ratelimit;
mod health;
mod metrics;

use routing::{ServiceRegistry, RouteConfig};
use ratelimit::{RateLimiter, RateLimitConfig};
use health::HealthChecker;
use metrics::MetricsCollector;

/// Main application state
#[derive(Clone)]
pub struct AppState {
    pub service_registry: Arc<RwLock<ServiceRegistry>>,
    pub auth_service: Arc<AuthService>,
    pub auth_middleware: Arc<AuthMiddleware>,
    pub rate_limiter: Arc<RateLimiter>,
    pub health_checker: Arc<HealthChecker>,
    pub metrics: Arc<MetricsCollector>,
    pub db_pool: Arc<sqlx::PgPool>,
    pub redis_client: Arc<redis::Client>,
}

/// Gateway configuration
#[derive(Debug, Clone, Deserialize)]
pub struct GatewayConfig {
    pub host: String,
    pub port: u16,
    pub database_url: String,
    pub redis_url: String,
    pub jwt_public_key: String,
    pub rate_limit: RateLimitConfig,
    pub request_timeout_sec: u64,
    pub max_request_size_mb: usize,
    pub services: Vec<ServiceConfig>,
}

/// Service configuration for routing
#[derive(Debug, Clone, Deserialize)]
pub struct ServiceConfig {
    pub name: String,
    pub url: String,
    pub health_check_path: String,
    pub timeout_sec: u64,
    pub retries: u32,
}

/// API request wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiRequest {
    pub id: String,
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Option<serde_json::Value>,
    pub timestamp: i64,
}

/// API response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse {
    pub id: String,
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Option<serde_json::Value>,
    pub duration_ms: u64,
    pub timestamp: i64,
}

/// Error response format
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub request_id: String,
    pub timestamp: i64,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/prowzi".to_string()),
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            jwt_public_key: std::env::var("JWT_PUBLIC_KEY")
                .unwrap_or_else(|_| "".to_string()),
            rate_limit: RateLimitConfig::default(),
            request_timeout_sec: 30,
            max_request_size_mb: 10,
            services: vec![
                ServiceConfig {
                    name: "auth".to_string(),
                    url: "http://localhost:3001".to_string(),
                    health_check_path: "/health".to_string(),
                    timeout_sec: 30,
                    retries: 3,
                },
                ServiceConfig {
                    name: "orchestrator".to_string(),
                    url: "http://localhost:8081".to_string(),
                    health_check_path: "/health".to_string(),
                    timeout_sec: 30,
                    retries: 3,
                },
                ServiceConfig {
                    name: "evaluator".to_string(),
                    url: "http://localhost:8082".to_string(),
                    health_check_path: "/health".to_string(),
                    timeout_sec: 30,
                    retries: 3,
                },
                ServiceConfig {
                    name: "curator".to_string(),
                    url: "http://localhost:8083".to_string(),
                    health_check_path: "/health".to_string(),
                    timeout_sec: 30,
                    retries: 3,
                },
            ],
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("gateway=debug,tower_http=debug,prowzi_auth=debug")
        .init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Load configuration
    let config = load_config().await?;
    
    // Initialize database connection pool
    let db_pool = Arc::new(
        sqlx::postgres::PgPoolOptions::new()
            .max_connections(20)
            .connect(&config.database_url)
            .await?
    );

    // Initialize Redis client
    let redis_client = Arc::new(redis::Client::open(config.redis_url.clone())?);

    // Initialize auth service with config
    let auth_config = AuthConfig {
        database_url: config.database_url.clone(),
        redis_url: config.redis_url.clone(),
        jwt_secret: std::env::var("JWT_SECRET")
            .unwrap_or_else(|_| "dev-secret".to_string()),
        jwt_public_key: config.jwt_public_key.clone(),
        refresh_secret: std::env::var("REFRESH_SECRET")
            .unwrap_or_else(|_| "dev-refresh-secret".to_string()),
        environment: std::env::var("ENVIRONMENT")
            .unwrap_or_else(|_| "development".to_string()),
    };

    let auth_service = Arc::new(AuthService::new(auth_config.clone()).await?);
    
    // Initialize auth middleware
    let auth_middleware = Arc::new(AuthMiddleware::new(
        &config.jwt_public_key,
        redis_client.clone(),
    )?);
    
    // Initialize other services
    let service_registry = Arc::new(RwLock::new(ServiceRegistry::new()));
    let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit.clone()));
    let health_checker = Arc::new(HealthChecker::new());
    let metrics = Arc::new(MetricsCollector::new());

    // Register services
    {
        let mut registry = service_registry.write().await;
        for service_config in &config.services {
            registry.register_service(
                service_config.name.clone(),
                RouteConfig {
                    url: service_config.url.clone(),
                    health_check_path: service_config.health_check_path.clone(),
                    timeout: Duration::from_secs(service_config.timeout_sec),
                    retries: service_config.retries,
                }
            );
        }
    }

    let app_state = AppState {
        service_registry,
        auth_service,
        auth_middleware,
        rate_limiter,
        health_checker,
        metrics,
        db_pool,
        redis_client,
    };

    // Build application router
    let app = create_router(app_state.clone(), &config);

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Gateway starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Create the main application router
fn create_router(state: AppState, config: &GatewayConfig) -> Router {
    Router::new()
        // Public routes (no auth required)
        .route("/health", get(health_check))
        .route("/health/deep", get(deep_health_check))
        .route("/metrics", get(get_metrics))
        .route("/metrics/prometheus", get(prometheus_metrics))
        
        // Auth service proxy (no auth required for auth endpoints)
        .route("/auth/*path", any(proxy_auth_service))
        .route("/.well-known/*path", any(proxy_auth_service))
        
        // Protected routes (require authentication)
        .route("/api/v1/*path", 
            get(proxy_request).post(proxy_request).put(proxy_request).delete(proxy_request))
        .route("/api/orchestrator/*path", 
            get(proxy_orchestrator).post(proxy_orchestrator).put(proxy_orchestrator).delete(proxy_orchestrator))
        .route("/api/evaluator/*path", 
            get(proxy_evaluator).post(proxy_evaluator).put(proxy_evaluator).delete(proxy_evaluator))
        .route("/api/curator/*path", 
            get(proxy_curator).post(proxy_curator).put(proxy_curator).delete(proxy_curator))
        .layer(middleware::from_fn_with_state(
            (state.auth_middleware.clone(), state.db_pool.clone()),
            protected_route_middleware,
        ))
        
        // Service management endpoints (admin only)
        .route("/gateway/services", get(list_services))
        .route("/gateway/services/:service/health", get(service_health))
        .route("/gateway/routes", get(list_routes))
        .layer(middleware::from_fn_with_state(
            (state.auth_middleware.clone(), state.db_pool.clone()),
            admin_route_middleware,
        ))
        
        // WebSocket proxy for real-time connections
        .route("/ws/*path", get(proxy_websocket))
        
        // Apply global middleware layers
        .layer(
            ServiceBuilder::new()
                // Request tracing
                .layer(TraceLayer::new_for_http())
                // Request timeout
                .layer(TimeoutLayer::new(Duration::from_secs(config.request_timeout_sec)))
                // Request size limit
                .layer(RequestBodyLimitLayer::new(config.max_request_size_mb * 1024 * 1024))
                // Concurrency limit
                .layer(ConcurrencyLimitLayer::new(1000))
                // CORS
                .layer(CorsLayer::permissive())
                // Rate limiting
                .layer(middleware::from_fn_with_state(
                    state.redis_client.clone(),
                    rate_limit_middleware,
                ))
                // Custom middleware for metrics and general request handling
                .layer(middleware::from_fn_with_state(state.clone(), request_middleware))
        )
        .with_state(state)
}

/// Request middleware for authentication, rate limiting, and metrics
async fn request_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    let start_time = Instant::now();
    let request_id = Uuid::new_v4().to_string();
    
    // Extract client IP for rate limiting
    let client_ip = extract_client_ip(&headers);
    
    // Apply rate limiting
    if !state.rate_limiter.check_rate_limit(&client_ip).await {
        tracing::warn!("Rate limit exceeded for IP: {}", client_ip);
        state.metrics.record_rate_limit_exceeded().await;
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Record request metrics
    state.metrics.record_request_start(&request_id).await;

    // Continue with request processing
    let response = next.run(request).await;
    
    // Record completion metrics
    let duration = start_time.elapsed();
    state.metrics.record_request_complete(&request_id, duration, response.status()).await;
    
    Ok(response)
}

/// Extract client IP from headers or connection info
fn extract_client_ip(headers: &HeaderMap) -> String {
    headers
        .get("x-forwarded-for")
        .or_else(|| headers.get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or(s).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Basic health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "gateway",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().timestamp()
    }))
}

/// Deep health check that verifies downstream services
async fn deep_health_check(State(state): State<AppState>) -> impl IntoResponse {
    let service_health = state.health_checker.check_all_services().await;
    let overall_healthy = service_health.values().all(|&status| status);
    
    let status_code = if overall_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    
    (status_code, Json(serde_json::json!({
        "status": if overall_healthy { "healthy" } else { "unhealthy" },
        "service": "gateway",
        "version": env!("CARGO_PKG_VERSION"),
        "services": service_health,
        "timestamp": chrono::Utc::now().timestamp()
    })))
}

/// Get gateway metrics
async fn get_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.get_metrics().await;
    Json(metrics)
}

/// Get Prometheus-formatted metrics
async fn prometheus_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.get_prometheus_metrics().await;
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        metrics
    )
}

/// List registered services
async fn list_services(State(state): State<AppState>) -> impl IntoResponse {
    let registry = state.service_registry.read().await;
    let services = registry.list_services();
    Json(services)
}

/// Check health of a specific service
async fn service_health(
    Path(service): Path<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let is_healthy = state.health_checker.check_service(&service).await;
    
    let status_code = if is_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    
    (status_code, Json(serde_json::json!({
        "service": service,
        "healthy": is_healthy,
        "timestamp": chrono::Utc::now().timestamp()
    })))
}

/// List available routes
async fn list_routes(State(state): State<AppState>) -> impl IntoResponse {
    let registry = state.service_registry.read().await;
    let routes = registry.list_routes();
    Json(routes)
}

/// Generic API proxy handler
async fn proxy_request(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: Option<String>,
) -> Result<impl IntoResponse, StatusCode> {
    // Determine target service from path
    let service_name = path.split('/').next().unwrap_or("default");
    
    let registry = state.service_registry.read().await;
    if let Some(route_config) = registry.get_service(service_name) {
        let target_url = format!("{}/api/v1/{}", route_config.url, path);
        proxy_to_service(&target_url, method, headers, body, &route_config).await
    } else {
        tracing::warn!("Service not found: {}", service_name);
        Err(StatusCode::NOT_FOUND)
    }
}

/// Proxy requests to orchestrator service
async fn proxy_orchestrator(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: Option<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let registry = state.service_registry.read().await;
    if let Some(route_config) = registry.get_service("orchestrator") {
        let target_url = format!("{}/{}", route_config.url, path);
        proxy_to_service(&target_url, method, headers, body, &route_config).await
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Proxy requests to evaluator service
async fn proxy_evaluator(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: Option<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let registry = state.service_registry.read().await;
    if let Some(route_config) = registry.get_service("evaluator") {
        let target_url = format!("{}/{}", route_config.url, path);
        proxy_to_service(&target_url, method, headers, body, &route_config).await
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Proxy requests to curator service
async fn proxy_curator(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: Option<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let registry = state.service_registry.read().await;
    if let Some(route_config) = registry.get_service("curator") {
        let target_url = format!("{}/{}", route_config.url, path);
        proxy_to_service(&target_url, method, headers, body, &route_config).await
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Proxy WebSocket connections
async fn proxy_websocket(
    Path(path): Path<String>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    // WebSocket proxying implementation would go here
    // For now, return not implemented
    tracing::warn!("WebSocket proxying not yet implemented for path: {}", path);
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Generic service proxy function
async fn proxy_to_service(
    target_url: &str,
    method: axum::http::Method,
    headers: HeaderMap,
    body: Option<String>,
    route_config: &RouteConfig,
) -> Result<impl IntoResponse, StatusCode> {
    let client = reqwest::Client::new();
    
    let mut request_builder = match method {
        axum::http::Method::GET => client.get(target_url),
        axum::http::Method::POST => client.post(target_url),
        axum::http::Method::PUT => client.put(target_url),
        axum::http::Method::DELETE => client.delete(target_url),
        _ => return Err(StatusCode::METHOD_NOT_ALLOWED),
    };

    // Forward headers (excluding connection-specific headers)
    for (name, value) in headers.iter() {
        if !is_hop_by_hop_header(name.as_str()) {
            request_builder = request_builder.header(name, value);
        }
    }

    // Add body if present
    if let Some(body_content) = body {
        request_builder = request_builder.body(body_content);
    }

    // Set timeout
    request_builder = request_builder.timeout(route_config.timeout);

    // Execute request with retries
    let mut last_error = None;
    for attempt in 0..=route_config.retries {
        match request_builder.try_clone() {
            Some(req) => {
                match req.send().await {
                    Ok(response) => {
                        let status = response.status();
                        let body = response.text().await.unwrap_or_default();
                        
                        return Ok((
                            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                            body
                        ));
                    }
                    Err(e) => {
                        last_error = Some(e);
                        if attempt < route_config.retries {
                            tracing::warn!("Request failed, retrying... attempt {}/{}", attempt + 1, route_config.retries);
                            tokio::time::sleep(Duration::from_millis(100 * 2_u64.pow(attempt))).await;
                        }
                    }
                }
            }
            None => {
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    }

    tracing::error!("All retry attempts failed: {:?}", last_error);
    Err(StatusCode::BAD_GATEWAY)
}

/// Check if header is a hop-by-hop header that shouldn't be forwarded
fn is_hop_by_hop_header(name: &str) -> bool {
    matches!(
        name.to_lowercase().as_str(),
        "connection" | "keep-alive" | "proxy-authenticate" | "proxy-authorization" 
        | "te" | "trailers" | "transfer-encoding" | "upgrade"
    )
}

/// Protected route middleware (requires authentication)
async fn protected_route_middleware(
    State((auth_middleware, db_pool)): State<(Arc<AuthMiddleware>, Arc<sqlx::PgPool>)>,
    request: axum::extract::Request,
    next: middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    auth_middleware(
        State(auth_middleware),
        State(db_pool),
        request,
        next,
    ).await
    .map_err(|_| StatusCode::UNAUTHORIZED)
}

/// Admin route middleware (requires admin role)
async fn admin_route_middleware(
    State((auth_middleware, db_pool)): State<(Arc<AuthMiddleware>, Arc<sqlx::PgPool>)>,
    request: axum::extract::Request,
    next: middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    // First validate authentication
    let response = auth_middleware(
        State(auth_middleware),
        State(db_pool),
        request,
        next,
    ).await
    .map_err(|_| StatusCode::UNAUTHORIZED)?;

    // Check if user has admin role (this would be done in the middleware)
    // For now, we'll just pass through authenticated requests
    Ok(response)
}

/// Proxy requests to auth service
async fn proxy_auth_service(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: Option<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let registry = state.service_registry.read().await;
    if let Some(route_config) = registry.get_service("auth") {
        let target_url = format!("{}/{}", route_config.url, path);
        proxy_to_service(&target_url, method, headers, body, &route_config).await
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Load configuration from file or environment
async fn load_config() -> Result<GatewayConfig, Box<dyn std::error::Error>> {
    // Try to load from file first, fall back to defaults
    if let Ok(config_str) = tokio::fs::read_to_string("gateway-config.toml").await {
        Ok(toml::from_str(&config_str)?)
    } else {
        tracing::info!("No config file found, using defaults");
        Ok(GatewayConfig::default())
    }
}