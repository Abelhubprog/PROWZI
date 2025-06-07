use prowzi_auth::{
    config::AuthConfig,
    auth_service::AuthService,
    routes::auth_routes,
    middleware::create_middleware_stack,
    errors::AuthError,
};
use axum::Router;
use std::sync::Arc;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "prowzi_auth=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration from environment
    dotenvy::dotenv().ok();
    let config = AuthConfig::from_env()?;

    info!("Starting Prowzi Auth Service");
    info!("Environment: {:?}", config.environment);

    // Initialize auth service
    let auth_service = Arc::new(AuthService::new(config.clone()).await?);

    // Create application with middleware
    let app = Router::new()
        .merge(auth_routes(auth_service))
        .layer(create_middleware_stack());

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await?;
    info!("Auth service listening on {}", listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}
