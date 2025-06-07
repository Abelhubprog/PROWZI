// File: platform/auth/src/jwt_auth.rs

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{
    decode, encode, Algorithm, DecodingKey, EncodingKey, Header, TokenData, Validation,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,           // Subject (user ID)
    pub exp: i64,              // Expiration time
    pub iat: i64,              // Issued at
    pub nbf: i64,              // Not before
    pub jti: String,           // JWT ID
    pub iss: String,           // Issuer
    pub aud: Vec<String>,      // Audience
    pub roles: Vec<String>,    // User roles
    pub permissions: Vec<String>, // Specific permissions
    pub scope: String,         // OAuth2 scope
    pub session_id: String,    // Session tracking
    pub ip_address: Option<String>, // IP binding
}

#[derive(Debug, Clone)]
pub struct JwtConfig {
    pub secret: String,
    pub issuer: String,
    pub audience: Vec<String>,
    pub expiration_hours: i64,
    pub refresh_expiration_days: i64,
    pub algorithm: Algorithm,
}

#[derive(Debug, Clone)]
pub struct AuthState {
    pub jwt_config: JwtConfig,
    pub encoding_key: EncodingKey,
    pub decoding_key: DecodingKey,
    pub revoked_tokens: Arc<RwLock<HashSet<String>>>,
    pub active_sessions: Arc<RwLock<HashMap<String, SessionInfo>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub user_id: String,
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub ip_address: String,
    pub user_agent: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthRequest {
    pub username: String,
    pub password: String,
    pub device_id: Option<String>,
    pub two_factor_code: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub scope: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshRequest {
    pub refresh_token: String,
}

impl AuthState {
    pub fn new(config: JwtConfig) -> Self {
        let encoding_key = EncodingKey::from_secret(config.secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(config.secret.as_bytes());
        
        Self {
            jwt_config: config,
            encoding_key,
            decoding_key,
            revoked_tokens: Arc::new(RwLock::new(HashSet::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn generate_tokens(
        &self,
        user_id: &str,
        roles: Vec<String>,
        permissions: Vec<String>,
        ip_address: Option<String>,
    ) -> Result<AuthResponse, AuthError> {
        let now = Utc::now();
        let session_id = Uuid::new_v4().to_string();
        
        // Access token claims
        let access_claims = Claims {
            sub: user_id.to_string(),
            exp: (now + Duration::hours(self.jwt_config.expiration_hours)).timestamp(),
            iat: now.timestamp(),
            nbf: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
            iss: self.jwt_config.issuer.clone(),
            aud: self.jwt_config.audience.clone(),
            roles: roles.clone(),
            permissions: permissions.clone(),
            scope: "read write".to_string(),
            session_id: session_id.clone(),
            ip_address: ip_address.clone(),
        };
        
        // Refresh token claims
        let refresh_claims = Claims {
            sub: user_id.to_string(),
            exp: (now + Duration::days(self.jwt_config.refresh_expiration_days)).timestamp(),
            iat: now.timestamp(),
            nbf: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
            iss: self.jwt_config.issuer.clone(),
            aud: vec!["refresh".to_string()],
            roles: vec![],
            permissions: vec![],
            scope: "refresh".to_string(),
            session_id: session_id.clone(),
            ip_address: ip_address.clone(),
        };
        
        let header = Header::new(self.jwt_config.algorithm);
        
        let access_token = encode(&header, &access_claims, &self.encoding_key)
            .map_err(|e| AuthError::TokenGeneration(e.to_string()))?;
            
        let refresh_token = encode(&header, &refresh_claims, &self.encoding_key)
            .map_err(|e| AuthError::TokenGeneration(e.to_string()))?;
        
        // Store session info
        let session_info = SessionInfo {
            user_id: user_id.to_string(),
            session_id,
            created_at: now,
            last_activity: now,
            ip_address: ip_address.unwrap_or_else(|| "unknown".to_string()),
            user_agent: "unknown".to_string(),
        };
        
        let mut sessions = self.active_sessions.write().await;
        sessions.insert(user_id.to_string(), session_info);
        
        Ok(AuthResponse {
            access_token,
            refresh_token,
            token_type: "Bearer".to_string(),
            expires_in: self.jwt_config.expiration_hours * 3600,
            scope: "read write".to_string(),
        })
    }
    
    pub async fn verify_token(&self, token: &str) -> Result<Claims, AuthError> {
        // Check if token is revoked
        let revoked = self.revoked_tokens.read().await;
        if revoked.contains(token) {
            return Err(AuthError::TokenRevoked);
        }
        
        let mut validation = Validation::new(self.jwt_config.algorithm);
        validation.set_audience(&self.jwt_config.audience);
        validation.set_issuer(&[&self.jwt_config.issuer]);
        validation.validate_exp = true;
        validation.validate_nbf = true;
        
        let token_data: TokenData<Claims> = decode(token, &self.decoding_key, &validation)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;
        
        // Verify session is still active
        let sessions = self.active_sessions.read().await;
        if !sessions.contains_key(&token_data.claims.sub) {
            return Err(AuthError::SessionExpired);
        }
        
        Ok(token_data.claims)
    }
    
    pub async fn refresh_token(&self, refresh_token: &str) -> Result<AuthResponse, AuthError> {
        let mut validation = Validation::new(self.jwt_config.algorithm);
        validation.set_audience(&["refresh"]);
        validation.set_issuer(&[&self.jwt_config.issuer]);
        
        let token_data: TokenData<Claims> = decode(refresh_token, &self.decoding_key, &validation)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;
        
        if token_data.claims.scope != "refresh" {
            return Err(AuthError::InvalidToken("Not a refresh token".to_string()));
        }
        
        // Get user info and generate new tokens
        // In production, fetch from database
        let roles = vec!["user".to_string()];
        let permissions = vec!["read".to_string(), "write".to_string()];
        
        self.generate_tokens(
            &token_data.claims.sub,
            roles,
            permissions,
            token_data.claims.ip_address,
        ).await
    }
    
    pub async fn revoke_token(&self, jti: &str) {
        let mut revoked = self.revoked_tokens.write().await;
        revoked.insert(jti.to_string());
    }
    
    pub async fn logout(&self, user_id: &str) {
        let mut sessions = self.active_sessions.write().await;
        sessions.remove(user_id);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Invalid token: {0}")]
    InvalidToken(String),
    #[error("Token expired")]
    TokenExpired,
    #[error("Token revoked")]
    TokenRevoked,
    #[error("Session expired")]
    SessionExpired,
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Token generation failed: {0}")]
    TokenGeneration(String),
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AuthError::InvalidToken(_) | AuthError::TokenExpired | AuthError::TokenRevoked => {
                (StatusCode::UNAUTHORIZED, "Invalid or expired token")
            }
            AuthError::SessionExpired => (StatusCode::UNAUTHORIZED, "Session expired"),
            AuthError::Unauthorized => (StatusCode::FORBIDDEN, "Unauthorized"),
            AuthError::InvalidCredentials => (StatusCode::UNAUTHORIZED, "Invalid credentials"),
            AuthError::TokenGeneration(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Token generation failed"),
        };
        
        let body = Json(serde_json::json!({
            "error": error_message,
            "status": status.as_u16(),
        }));
        
        (status, body).into_response()
    }
}

/// Authentication middleware
pub async fn auth_middleware(
    State(auth_state): State<Arc<AuthState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|header| header.to_str().ok());
    
    let token = match auth_header {
        Some(header) if header.starts_with("Bearer ") => &header[7..],
        _ => return Err(AuthError::Unauthorized),
    };
    
    let claims = auth_state.verify_token(token).await?;
    
    // Add claims to request extensions
    request.extensions_mut().insert(claims);
    
    Ok(next.run(request).await)
}

/// Role-based authorization middleware
pub fn require_role(required_role: &'static str) -> impl Fn(Request, Next) -> impl Future<Output = Result<Response, AuthError>> {
    move |request: Request, next: Next| async move {
        let claims = request
            .extensions()
            .get::<Claims>()
            .ok_or(AuthError::Unauthorized)?;
        
        if !claims.roles.contains(&required_role.to_string()) {
            return Err(AuthError::Unauthorized);
        }
        
        Ok(next.run(request).await)
    }
}

/// Permission-based authorization middleware
pub fn require_permission(required_permission: &'static str) -> impl Fn(Request, Next) -> impl Future<Output = Result<Response, AuthError>> {
    move |request: Request, next: Next| async move {
        let claims = request
            .extensions()
            .get::<Claims>()
            .ok_or(AuthError::Unauthorized)?;
        
        if !claims.permissions.contains(&required_permission.to_string()) {
            return Err(AuthError::Unauthorized);
        }
        
        Ok(next.run(request).await)
    }
}

use std::collections::HashMap;
use std::future::Future;

/// Rate limiting per user
pub struct RateLimiter {
    limits: Arc<RwLock<HashMap<String, RateLimit>>>,
}

#[derive(Debug)]
struct RateLimit {
    count: u32,
    reset_at: DateTime<Utc>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limits: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn check_rate_limit(&self, user_id: &str, limit: u32) -> Result<(), AuthError> {
        let mut limits = self.limits.write().await;
        let now = Utc::now();
        
        let rate_limit = limits.entry(user_id.to_string()).or_insert(RateLimit {
            count: 0,
            reset_at: now + Duration::minutes(1),
        });
        
        if now > rate_limit.reset_at {
            rate_limit.count = 0;
            rate_limit.reset_at = now + Duration::minutes(1);
        }
        
        if rate_limit.count >= limit {
            return Err(AuthError::Unauthorized);
        }
        
        rate_limit.count += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_token_generation_and_verification() {
        let config = JwtConfig {
            secret: "test_secret_key_123456789012345678901234567890".to_string(),
            issuer: "prowzi".to_string(),
            audience: vec!["prowzi-api".to_string()],
            expiration_hours: 1,
            refresh_expiration_days: 7,
            algorithm: Algorithm::HS256,
        };
        
        let auth_state = AuthState::new(config);
        
        let response = auth_state.generate_tokens(
            "user123",
            vec!["user".to_string()],
            vec!["read".to_string()],
            Some("127.0.0.1".to_string()),
        ).await.unwrap();
        
        let claims = auth_state.verify_token(&response.access_token).await.unwrap();
        assert_eq!(claims.sub, "user123");
        assert!(claims.roles.contains(&"user".to_string()));
    }
}