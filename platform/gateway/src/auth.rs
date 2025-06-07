//! Authentication and authorization module for the Prowzi Gateway.
//!
//! This module handles JWT validation, tenant context setting for Row Level Security (RLS),
//! and provides middleware for securing API routes.

use axum::{
    async_trait,
    extract::{FromRequestParts, Query, TypedHeader},
    headers::{authorization::Bearer, Authorization},
    http::{request::Parts, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    RequestPartsExt,
};
use chrono::{DateTime, Utc};
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Postgres, Executor};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

/// JWT claims structure including tenant information
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Tenant ID for multi-tenancy and RLS
    pub tenant_id: String,
    /// Optional user roles
    #[serde(default)]
    pub roles: Vec<String>,
    /// Optional model overrides for LLM selection
    #[serde(default)]
    pub model_overrides: Option<serde_json::Value>,
    /// Issued at timestamp
    pub iat: i64,
    /// Expiration timestamp
    pub exp: i64,
    /// Issuer
    #[serde(default)]
    pub iss: String,
}

/// Query parameters for token extraction
#[derive(Debug, Deserialize)]
pub struct TokenQuery {
    token: Option<String>,
}

/// Authentication errors
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Missing authentication token")]
    MissingToken,
    
    #[error("Invalid token format")]
    InvalidTokenFormat,
    
    #[error("Token validation failed: {0}")]
    TokenValidation(String),
    
    #[error("Token expired")]
    TokenExpired,
    
    #[error("Missing required claim: {0}")]
    MissingClaim(String),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    #[error("JWT error: {0}")]
    JwtError(#[from] jsonwebtoken::errors::Error),
    
    #[error("Internal server error: {0}")]
    InternalError(String),
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingToken => (StatusCode::UNAUTHORIZED, "Missing authentication token"),
            AuthError::InvalidTokenFormat => (StatusCode::UNAUTHORIZED, "Invalid token format"),
            AuthError::TokenValidation(msg) => (StatusCode::UNAUTHORIZED, msg.as_str()),
            AuthError::TokenExpired => (StatusCode::UNAUTHORIZED, "Token expired"),
            AuthError::MissingClaim(claim) => (
                StatusCode::UNAUTHORIZED,
                &format!("Missing required claim: {}", claim),
            ),
            AuthError::DatabaseError(e) => {
                error!("Database error in auth: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Database error")
            }
            AuthError::JwtError(e) => {
                error!("JWT error: {}", e);
                (StatusCode::UNAUTHORIZED, "Invalid token")
            }
            AuthError::InternalError(msg) => {
                error!("Internal auth error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
            }
        };

        (status, message).into_response()
    }
}

/// Authentication extractor for Axum
pub struct Auth {
    /// Validated JWT claims
    pub claims: Claims,
}

#[async_trait]
impl<S> FromRequestParts<S> for Auth
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    #[instrument(skip(parts, _state), err)]
    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Try to extract the token from the Authorization header
        let auth_header = parts.extract::<TypedHeader<Authorization<Bearer>>>().await;
        
        // If header extraction fails, try query parameters
        let token = match auth_header {
            Ok(TypedHeader(Authorization(bearer))) => bearer.token().to_string(),
            Err(_) => {
                let query = parts.extract::<Query<TokenQuery>>().await
                    .map_err(|_| AuthError::MissingToken)?;
                
                query.token.clone().ok_or(AuthError::MissingToken)?
            }
        };
        
        // Get JWT secret from environment
        let jwt_secret = std::env::var("JWT_SECRET")
            .map_err(|_| AuthError::InternalError("JWT_SECRET not configured".to_string()))?;
        
        // Decode and validate the token
        let token_data = decode::<Claims>(
            &token,
            &DecodingKey::from_secret(jwt_secret.as_bytes()),
            &Validation::new(Algorithm::HS256),
        )
        .map_err(|e| match e.kind() {
            jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
            _ => AuthError::JwtError(e),
        })?;
        
        // Validate required claims
        if token_data.claims.tenant_id.is_empty() {
            return Err(AuthError::MissingClaim("tenant_id".to_string()));
        }
        
        // Return the authenticated user with claims
        Ok(Auth {
            claims: token_data.claims,
        })
    }
}

/// Authentication middleware that sets the tenant context for RLS
pub async fn tenant_context_middleware<B>(
    auth: Auth,
    db_pool: axum::extract::Extension<PgPool>,
    request: axum::http::Request<B>,
    next: Next<B>,
) -> Result<Response, AuthError> {
    let tenant_id = auth.claims.tenant_id.clone();
    let db = db_pool.0;
    
    debug!("Setting tenant context for tenant_id: {}", tenant_id);
    
    // Get a connection from the pool
    let mut conn = db.acquire().await
        .map_err(|e| {
            error!("Failed to acquire database connection: {}", e);
            AuthError::DatabaseError(e)
        })?;
    
    // Set the tenant context for RLS
    conn.execute(
        &format!("SELECT set_config('prowzi.tenant', $1, true)"),
        &[&tenant_id]
    )
    .await
    .map_err(|e| {
        error!("Failed to set tenant context: {}", e);
        AuthError::DatabaseError(e)
    })?;
    
    // Release the connection back to the pool
    drop(conn);
    
    // Add tenant_id to request extensions for use in handlers
    let mut request = request;
    request.extensions_mut().insert(TenantId(tenant_id));
    
    // Continue with the request
    Ok(next.run(request).await)
}

/// Tenant ID extractor for use in handlers
#[derive(Debug, Clone)]
pub struct TenantId(pub String);

/// Generate a new JWT token for a user
#[instrument(skip(secret))]
pub fn generate_token(
    user_id: &str,
    tenant_id: &str,
    roles: Vec<String>,
    model_overrides: Option<serde_json::Value>,
    secret: &str,
    expiry_hours: i64,
) -> Result<String, AuthError> {
    use jsonwebtoken::{encode, EncodingKey, Header};
    
    let now = Utc::now();
    let expiry = now + chrono::Duration::hours(expiry_hours);
    
    let claims = Claims {
        sub: user_id.to_string(),
        tenant_id: tenant_id.to_string(),
        roles,
        model_overrides,
        iat: now.timestamp(),
        exp: expiry.timestamp(),
        iss: "prowzi".to_string(),
    };
    
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(AuthError::JwtError)
}

/// Verify a token and return the claims without using the database
#[instrument(skip(token, secret))]
pub fn verify_token(token: &str, secret: &str) -> Result<Claims, AuthError> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::new(Algorithm::HS256),
    )
    .map_err(|e| match e.kind() {
        jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
        _ => AuthError::JwtError(e),
    })?;
    
    Ok(token_data.claims)
}

/// Test module
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    #[test]
    fn test_token_generation_and_verification() {
        let secret = "test_secret";
        let user_id = "user123";
        let tenant_id = "tenant456";
        let roles = vec!["user".to_string(), "admin".to_string()];
        
        let token = generate_token(
            user_id,
            tenant_id,
            roles.clone(),
            None,
            secret,
            24
        ).expect("Token generation should succeed");
        
        let claims = verify_token(&token, secret).expect("Token verification should succeed");
        
        assert_eq!(claims.sub, user_id);
        assert_eq!(claims.tenant_id, tenant_id);
        assert_eq!(claims.roles, roles);
        assert!(claims.exp > Utc::now().timestamp());
    }
    
    #[test]
    fn test_expired_token() {
        use jsonwebtoken::{encode, EncodingKey, Header};
        
        let secret = "test_secret";
        let now = Utc::now();
        let expired = now - chrono::Duration::hours(1);
        
        let claims = Claims {
            sub: "user123".to_string(),
            tenant_id: "tenant456".to_string(),
            roles: vec![],
            model_overrides: None,
            iat: now.timestamp(),
            exp: expired.timestamp(),
            iss: "prowzi".to_string(),
        };
        
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        ).expect("Token encoding should succeed");
        
        let result = verify_token(&token, secret);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }
}
