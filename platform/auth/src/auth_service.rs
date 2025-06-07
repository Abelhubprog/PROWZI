use crate::{
    config::AuthConfig,
    errors::AuthError,
    models::*,
    utils::{verify_ethereum_signature, verify_solana_signature, verify_siwe_message},
};
use sqlx::{PgPool, Row};
use redis::AsyncCommands;
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation, EncodingKey, DecodingKey};
use std::sync::Arc;
use chrono::{Utc, Duration};
use tracing::{info, warn, error};

pub struct AuthService {
    jwt_secret: Arc<EncodingKey>,
    jwt_decode_key: Arc<DecodingKey>,
    refresh_secret: Arc<EncodingKey>,
    refresh_decode_key: Arc<DecodingKey>,
    db: Arc<PgPool>,
    redis: Arc<redis::Client>,
    config: AuthConfig,
}

impl AuthService {
    pub async fn new(config: AuthConfig) -> Result<Self, AuthError> {
        let jwt_secret = Arc::new(EncodingKey::from_rsa_pem(
            config.jwt_private_key.as_bytes()
        ).map_err(|e| AuthError::ConfigError(format!("Invalid JWT private key: {}", e)))?);

        let jwt_decode_key = Arc::new(DecodingKey::from_rsa_pem(
            config.jwt_public_key.as_bytes()
        ).map_err(|e| AuthError::ConfigError(format!("Invalid JWT public key: {}", e)))?);

        let refresh_secret = Arc::new(EncodingKey::from_rsa_pem(
            config.refresh_private_key.as_bytes()
        ).map_err(|e| AuthError::ConfigError(format!("Invalid refresh private key: {}", e)))?);

        let refresh_decode_key = Arc::new(DecodingKey::from_rsa_pem(
            config.refresh_public_key.as_bytes()
        ).map_err(|e| AuthError::ConfigError(format!("Invalid refresh public key: {}", e)))?);

        let db = Arc::new(create_pool(&config.database_url).await?);
        let redis = Arc::new(redis::Client::open(config.redis_url.clone())
            .map_err(|e| AuthError::ConfigError(format!("Invalid Redis URL: {}", e)))?);

        Ok(Self {
            jwt_secret,
            jwt_decode_key,
            refresh_secret,
            refresh_decode_key,
            db,
            redis,
            config,
        })
    }

    pub async fn authenticate_wallet(
        &self,
        request: WalletAuthRequest,
    ) -> Result<AuthResponse, AuthError> {
        info!("Authenticating wallet: {} type: {}", request.address, request.wallet_type);

        // Verify signature
        let user_id = match request.wallet_type.as_str() {
            "ethereum" => verify_ethereum_signature(&request).await?,
            "solana" => verify_solana_signature(&request).await?,
            _ => return Err(AuthError::InvalidWalletType),
        };

        // Get or create user with tenant context
        let user = self.get_or_create_user(&user_id, &request.address, request.tenant_id.as_deref()).await?;

        // Generate tokens
        let (access_token, refresh_token) = self.generate_tokens(&user).await?;

        info!("User authenticated: {} tenant: {}", user.id, user.tenant_id);

        Ok(AuthResponse {
            access_token,
            refresh_token,
            expires_in: self.config.jwt_expiry_seconds,
            scope: user.scope.clone(),
            user: UserInfo {
                id: user.id.to_string(),
                email: user.email.clone(),
                wallet_address: user.wallet_address.clone(),
                tier: user.tier.clone(),
                tenant_id: user.tenant_id.clone(),
            },
        })
    }

    async fn get_or_create_user(
        &self,
        user_id: &str,
        wallet_address: &str,
        tenant_id: Option<&str>,
    ) -> Result<User, AuthError> {
        // First try to find existing user
        let existing_user = sqlx::query_as::<_, User>(
            "SELECT id, email, wallet_address, tier, preferences, created_at, last_active, tenant_id, 
             ARRAY['read', 'write'] as scope FROM users WHERE wallet_address = $1"
        )
        .bind(wallet_address)
        .fetch_optional(&**self.db)
        .await?;

        if let Some(mut user) = existing_user {
            // Update last active
            sqlx::query("UPDATE users SET last_active = NOW() WHERE id = $1")
                .bind(user.id)
                .execute(&**self.db)
                .await?;
            
            user.last_active = Some(Utc::now());
            
            // Set tenant-based scope
            user.scope = self.get_user_scope(&user.tier, &user.tenant_id).await;
            
            return Ok(user);
        }

        // Create new user
        let tenant_id = tenant_id.unwrap_or("default");
        let new_user_id = uuid::Uuid::new_v4();
        
        let user = sqlx::query_as::<_, User>(
            "INSERT INTO users (id, wallet_address, tier, tenant_id, created_at, last_active) 
             VALUES ($1, $2, 'free', $3, NOW(), NOW()) 
             RETURNING id, email, wallet_address, tier, preferences, created_at, last_active, tenant_id,
             ARRAY['read'] as scope"
        )
        .bind(new_user_id)
        .bind(wallet_address)
        .bind(tenant_id)
        .fetch_one(&**self.db)
        .await?;

        info!("Created new user: {} for tenant: {}", user.id, user.tenant_id);
        Ok(user)
    }

    async fn get_user_scope(&self, tier: &str, tenant_id: &str) -> Vec<String> {
        // Determine scope based on user tier and tenant
        match tier {
            "enterprise" => vec!["read".to_string(), "write".to_string(), "admin".to_string()],
            "pro" => vec!["read".to_string(), "write".to_string()],
            "free" => vec!["read".to_string()],
            _ => vec!["read".to_string()],
        }
    }

    async fn generate_tokens(&self, user: &User) -> Result<(String, String), AuthError> {
        let now = Utc::now();
        let jti = uuid::Uuid::new_v4().to_string();

        // Access token
        let access_claims = Claims {
            sub: user.id.to_string(),
            tenant: user.tenant_id.clone(),
            scope: user.scope.clone(),
            exp: (now + Duration::seconds(self.config.jwt_expiry_seconds)).timestamp(),
            iat: now.timestamp(),
            jti: jti.clone(),
        };

        let access_token = encode(
            &Header::new(Algorithm::RS256),
            &access_claims,
            &self.jwt_secret,
        )?;

        // Refresh token (7 days, sliding window)
        let refresh_claims = Claims {
            sub: user.id.to_string(),
            tenant: user.tenant_id.clone(),
            scope: vec!["refresh".to_string()],
            exp: (now + Duration::seconds(self.config.refresh_expiry_seconds)).timestamp(),
            iat: now.timestamp(),
            jti: uuid::Uuid::new_v4().to_string(),
        };

        let refresh_token = encode(
            &Header::new(Algorithm::RS256),
            &refresh_claims,
            &self.refresh_secret,
        )?;

        // Store access token in Redis for revocation with TTL
        let mut conn = self.redis.get_async_connection().await?;
        conn.setex(
            format!("jwt:access:{}", jti),
            self.config.jwt_expiry_seconds as usize,
            &access_token,
        ).await?;

        Ok((access_token, refresh_token))
    }

    pub async fn refresh_token(
        &self,
        refresh_token: &str,
    ) -> Result<AuthResponse, AuthError> {
        // Decode refresh token
        let token_data = decode::<Claims>(
            refresh_token,
            &self.refresh_decode_key,
            &Validation::new(Algorithm::RS256),
        )?;

        // Check if revoked
        let mut conn = self.redis.get_async_connection().await?;
        if conn.exists(format!("jwt:revoked:{}", token_data.claims.jti)).await? {
            return Err(AuthError::TokenRevoked);
        }

        // Get user
        let user = self.get_user(&token_data.claims.sub).await?;

        // Generate new tokens
        let (access_token, new_refresh_token) = self.generate_tokens(&user).await?;

        // Revoke old refresh token
        conn.setex(
            format!("jwt:revoked:{}", token_data.claims.jti),
            self.config.refresh_expiry_seconds as usize,
            "1",
        ).await?;

        Ok(AuthResponse {
            access_token,
            refresh_token: new_refresh_token,
            expires_in: self.config.jwt_expiry_seconds,
            scope: user.scope.clone(),
            user: UserInfo {
                id: user.id.to_string(),
                email: user.email.clone(),
                wallet_address: user.wallet_address.clone(),
                tier: user.tier.clone(),
                tenant_id: user.tenant_id.clone(),
            },
        })
    }

    pub async fn introspect_token(
        &self,
        token: &str,
    ) -> Result<IntrospectResponse, AuthError> {
        match decode::<Claims>(
            token,
            &self.jwt_decode_key,
            &Validation::new(Algorithm::RS256),
        ) {
            Ok(token_data) => {
                // Check if revoked
                let mut conn = self.redis.get_async_connection().await?;
                let is_revoked: bool = conn.exists(format!("jwt:revoked:{}", token_data.claims.jti)).await?;
                
                if is_revoked {
                    return Ok(IntrospectResponse {
                        active: false,
                        scope: vec![],
                        client_id: None,
                        username: None,
                        exp: None,
                        iat: None,
                        sub: None,
                        tenant: None,
                    });
                }

                Ok(IntrospectResponse {
                    active: true,
                    scope: token_data.claims.scope,
                    client_id: Some("prowzi".to_string()),
                    username: Some(token_data.claims.sub.clone()),
                    exp: Some(token_data.claims.exp),
                    iat: Some(token_data.claims.iat),
                    sub: Some(token_data.claims.sub),
                    tenant: Some(token_data.claims.tenant),
                })
            }
            Err(_) => Ok(IntrospectResponse {
                active: false,
                scope: vec![],
                client_id: None,
                username: None,
                exp: None,
                iat: None,
                sub: None,
                tenant: None,
            }),
        }
    }

    async fn get_user(&self, user_id: &str) -> Result<User, AuthError> {
        let user_uuid = uuid::Uuid::parse_str(user_id)
            .map_err(|_| AuthError::UserNotFound)?;

        let user = sqlx::query_as::<_, User>(
            "SELECT id, email, wallet_address, tier, preferences, created_at, last_active, tenant_id,
             ARRAY['read', 'write'] as scope FROM users WHERE id = $1"
        )
        .bind(user_uuid)
        .fetch_optional(&**self.db)
        .await?
        .ok_or(AuthError::UserNotFound)?;

        Ok(user)
    }

    pub async fn revoke_token(&self, jti: &str) -> Result<(), AuthError> {
        let mut conn = self.redis.get_async_connection().await?;
        conn.setex(
            format!("jwt:revoked:{}", jti),
            self.config.refresh_expiry_seconds as usize,
            "1",
        ).await?;
        Ok(())
    }

    pub fn get_jwks(&self) -> JwksResponse {
        // In production, this should return the actual public key in JWK format
        // For now, return a placeholder
        JwksResponse {
            keys: vec![],
        }
    }
}

async fn create_pool(database_url: &str) -> Result<PgPool, AuthError> {
    sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
        .map_err(AuthError::DatabaseError)
}
