use crate::{
    auth_service::AuthService,
    errors::AuthError,
    models::*,
    utils::generate_nonce,
};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
pub struct NonceQuery {
    pub address: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct NonceResponse {
    pub nonce: String,
    pub message: String,
}

/// Create authentication routes
pub fn auth_routes(auth_service: Arc<AuthService>) -> Router {
    Router::new()
        .route("/auth/nonce", get(get_nonce))
        .route("/auth/wallet", post(wallet_auth))
        .route("/auth/refresh", post(refresh_auth))
        .route("/auth/introspect", post(introspect))
        .route("/auth/revoke", post(revoke_token))
        .route("/.well-known/jwks.json", get(jwks))
        .route("/health", get(health_check))
        .with_state(auth_service)
}

/// Generate nonce for wallet authentication
async fn get_nonce(
    Query(params): Query<NonceQuery>,
) -> Result<impl IntoResponse, AuthError> {
    let nonce = generate_nonce();
    
    let message = if let Some(address) = params.address {
        format!(
            "prowzi.io wants you to sign in with your Ethereum account:\n{}\n\n\
            Sign in to Prowzi platform\n\n\
            URI: https://prowzi.io\n\
            Version: 1\n\
            Chain ID: 1\n\
            Nonce: {}\n\
            Issued At: {}",
            address,
            nonce,
            chrono::Utc::now().to_rfc3339()
        )
    } else {
        format!("Sign in to Prowzi\nNonce: {}", nonce)
    };

    Ok(Json(NonceResponse { nonce, message }))
}

/// Authenticate with wallet signature
async fn wallet_auth(
    State(auth): State<Arc<AuthService>>,
    Json(request): Json<WalletAuthRequest>,
) -> Result<impl IntoResponse, AuthError> {
    info!("Wallet authentication request for address: {}", request.address);

    let response = auth.authenticate_wallet(request).await?;
    
    Ok((StatusCode::OK, Json(response)))
}

/// Refresh access token
async fn refresh_auth(
    State(auth): State<Arc<AuthService>>,
    Json(request): Json<RefreshRequest>,
) -> Result<impl IntoResponse, AuthError> {
    let response = auth.refresh_token(&request.refresh_token).await?;
    Ok(Json(response))
}

/// Introspect token (OAuth 2.0 Token Introspection)
async fn introspect(
    State(auth): State<Arc<AuthService>>,
    Json(request): Json<IntrospectRequest>,
) -> Result<impl IntoResponse, AuthError> {
    let result = auth.introspect_token(&request.token).await?;
    Ok(Json(result))
}

/// Revoke token
async fn revoke_token(
    State(auth): State<Arc<AuthService>>,
    Json(request): Json<RevokeRequest>,
) -> Result<impl IntoResponse, AuthError> {
    auth.revoke_token(&request.jti).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// JSON Web Key Set endpoint
async fn jwks(
    State(auth): State<Arc<AuthService>>,
) -> impl IntoResponse {
    Json(auth.get_jwks())
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "prowzi-auth",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

#[derive(Debug, Deserialize)]
pub struct RevokeRequest {
    pub jti: String,
}
