use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: Option<String>,
    pub wallet_address: Option<String>,
    pub tier: String,
    pub preferences: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub last_active: Option<DateTime<Utc>>,
    pub tenant_id: String,
    pub scope: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Tenant {
    pub id: String,
    pub name: String,
    pub tier: String,
    pub settings: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub is_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,        // user_id
    pub tenant: String,     // tenant_id
    pub scope: Vec<String>, // permissions
    pub exp: i64,          // expiry
    pub iat: i64,          // issued at
    pub jti: String,       // JWT ID for revocation
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WalletAuthRequest {
    #[serde(rename = "type")]
    pub wallet_type: String, // "ethereum" or "solana"
    pub address: String,
    pub message: String,
    pub signature: String,
    pub tenant_id: Option<String>, // Optional tenant context
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: i64,
    pub scope: Vec<String>,
    pub user: UserInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserInfo {
    pub id: String,
    pub email: Option<String>,
    pub wallet_address: Option<String>,
    pub tier: String,
    pub tenant_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshRequest {
    pub refresh_token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntrospectRequest {
    pub token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntrospectResponse {
    pub active: bool,
    pub scope: Vec<String>,
    pub client_id: Option<String>,
    pub username: Option<String>,
    pub exp: Option<i64>,
    pub iat: Option<i64>,
    pub sub: Option<String>,
    pub tenant: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JwksResponse {
    pub keys: Vec<JwkKey>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JwkKey {
    pub kty: String,
    pub use_: String,
    pub kid: String,
    pub n: String,
    pub e: String,
}
