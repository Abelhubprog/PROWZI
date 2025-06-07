use crate::{errors::AuthError, models::Claims};
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;
use tower::ServiceBuilder;

pub struct AuthMiddleware {
    jwt_decode_key: Arc<DecodingKey>,
    redis: Arc<redis::Client>,
}

impl AuthMiddleware {
    pub fn new(jwt_public_key: &str, redis_client: Arc<redis::Client>) -> Result<Self, AuthError> {
        let jwt_decode_key = Arc::new(DecodingKey::from_rsa_pem(jwt_public_key.as_bytes())
            .map_err(|e| AuthError::ConfigError(format!("Invalid JWT public key: {}", e)))?);

        Ok(Self {
            jwt_decode_key,
            redis: redis_client,
        })
    }

    /// Extract and validate JWT token from Authorization header
    pub async fn validate_token(&self, headers: &HeaderMap) -> Result<Claims, AuthError> {
        // Extract token from Authorization header
        let auth_header = headers
            .get("authorization")
            .ok_or(AuthError::AccessDenied)?
            .to_str()
            .map_err(|_| AuthError::InvalidToken)?;

        let token = if auth_header.starts_with("Bearer ") {
            &auth_header[7..]
        } else {
            return Err(AuthError::InvalidToken);
        };

        // Decode JWT
        let token_data = decode::<Claims>(
            token,
            &self.jwt_decode_key,
            &Validation::new(Algorithm::RS256),
        )?;

        // Check if token is revoked
        let mut conn = self.redis.get_async_connection().await
            .map_err(AuthError::RedisError)?;
        
        use redis::AsyncCommands;
        let is_revoked: bool = conn.exists(format!("jwt:revoked:{}", token_data.claims.jti)).await
            .map_err(AuthError::RedisError)?;

        if is_revoked {
            return Err(AuthError::TokenRevoked);
        }

        Ok(token_data.claims)
    }

    /// Set tenant context in database connection
    pub async fn set_tenant_context(
        &self,
        db: &sqlx::PgPool,
        tenant_id: &str,
    ) -> Result<(), AuthError> {
        sqlx::query("SELECT set_config('prowzi.tenant', $1, false)")
            .bind(tenant_id)
            .execute(db)
            .await?;
        Ok(())
    }
}

/// Middleware function for protected routes
pub async fn auth_middleware(
    State(auth): State<Arc<AuthMiddleware>>,
    State(db): State<Arc<sqlx::PgPool>>,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    // Validate token and extract claims
    let claims = auth.validate_token(request.headers()).await?;

    // Set tenant context for RLS
    auth.set_tenant_context(&db, &claims.tenant).await?;

    // Add claims to request extensions for downstream handlers
    request.extensions_mut().insert(claims);

    // Continue to next middleware/handler
    Ok(next.run(request).await)
}

/// Middleware for optional authentication (allows anonymous access)
pub async fn optional_auth_middleware(
    State(auth): State<Arc<AuthMiddleware>>,
    State(db): State<Arc<sqlx::PgPool>>,
    mut request: Request,
    next: Next,
) -> Response {
    // Try to validate token, but don't fail if missing
    if let Ok(claims) = auth.validate_token(request.headers()).await {
        // Set tenant context if authenticated
        if auth.set_tenant_context(&db, &claims.tenant).await.is_ok() {
            request.extensions_mut().insert(claims);
        }
    }

    next.run(request).await
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    headers: HeaderMap,
    State(redis): State<Arc<redis::Client>>,
    request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    // Get client IP (in production, use X-Forwarded-For)
    let client_ip = headers
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    let mut conn = redis.get_async_connection().await
        .map_err(AuthError::RedisError)?;

    use redis::AsyncCommands;

    // Rate limiting: 100 requests per minute per IP
    let key = format!("rate_limit:{}", client_ip);
    let current: i32 = conn.get(&key).await.unwrap_or(0);

    if current >= 100 {
        return Err(AuthError::RateLimitExceeded);
    }

    // Increment counter
    let _: () = conn.incr(&key, 1).await.map_err(AuthError::RedisError)?;
    let _: () = conn.expire(&key, 60).await.map_err(AuthError::RedisError)?;

    Ok(next.run(request).await)
}

/// Create service layer with all middleware
pub fn create_middleware_stack() -> ServiceBuilder<
    tower::layer::util::Stack<
        tower::layer::util::Stack<
            tower_http::cors::CorsLayer,
            tower_http::trace::TraceLayer,
        >,
        RequestBodyLimitLayer,
    >,
> {
    ServiceBuilder::new()
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::cors::CorsLayer::permissive()) // Configure properly for production
        .layer(RequestBodyLimitLayer::new(1024 * 1024)) // 1MB limit
}

/// Scope-based authorization
pub fn require_scope(required_scope: &str) -> impl Fn(Claims) -> Result<(), AuthError> + '_ {
    move |claims: Claims| {
        if claims.scope.contains(&required_scope.to_string()) {
            Ok(())
        } else {
            Err(AuthError::AccessDenied)
        }
    }
}

/// Extract claims from request extensions
pub fn extract_claims(request: &Request) -> Option<&Claims> {
    request.extensions().get::<Claims>()
}
