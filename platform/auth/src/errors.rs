use thiserror::Error;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Invalid wallet type")]
    InvalidWalletType,
    
    #[error("Invalid signature")]
    InvalidSignature,
    
    #[error("Invalid wallet address")]
    InvalidAddress,
    
    #[error("Signature verification failed")]
    SignatureMismatch,
    
    #[error("Invalid SIWE message format")]
    InvalidSiweMessage,
    
    #[error("Token has expired")]
    TokenExpired,
    
    #[error("Token has been revoked")]
    TokenRevoked,
    
    #[error("Invalid token")]
    InvalidToken,
    
    #[error("User not found")]
    UserNotFound,
    
    #[error("Tenant not found")]
    TenantNotFound,
    
    #[error("Access denied")]
    AccessDenied,
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    #[error("Redis error: {0}")]
    RedisError(#[from] redis::RedisError),
    
    #[error("JWT error: {0}")]
    JwtError(#[from] jsonwebtoken::errors::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Internal server error")]
    InternalError,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AuthError::InvalidWalletType 
            | AuthError::InvalidSignature 
            | AuthError::InvalidAddress 
            | AuthError::InvalidSiweMessage 
            | AuthError::InvalidToken => (StatusCode::BAD_REQUEST, self.to_string()),
            
            AuthError::SignatureMismatch 
            | AuthError::AccessDenied => (StatusCode::UNAUTHORIZED, self.to_string()),
            
            AuthError::TokenExpired 
            | AuthError::TokenRevoked => (StatusCode::UNAUTHORIZED, "Token invalid".to_string()),
            
            AuthError::UserNotFound 
            | AuthError::TenantNotFound => (StatusCode::NOT_FOUND, self.to_string()),
            
            AuthError::RateLimitExceeded => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
            
            AuthError::DatabaseError(_) 
            | AuthError::RedisError(_) 
            | AuthError::JwtError(_) 
            | AuthError::ConfigError(_) 
            | AuthError::InternalError => (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error".to_string()),
        };

        let body = Json(json!({
            "error": error_message,
            "type": format!("{:?}", self).split('(').next().unwrap_or("Unknown")
        }));

        (status, body).into_response()
    }
}
