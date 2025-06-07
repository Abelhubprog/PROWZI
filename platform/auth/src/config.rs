use serde::{Deserialize, Serialize};
use std::env;
use crate::errors::AuthError;

#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub database_url: String,
    pub redis_url: String,
    pub jwt_private_key: String,
    pub jwt_public_key: String,
    pub refresh_private_key: String,
    pub refresh_public_key: String,
    pub jwt_expiry_seconds: i64,
    pub refresh_expiry_seconds: i64,
    pub rate_limit_requests: u32,
    pub rate_limit_window_seconds: u64,
    pub cors_origins: Vec<String>,
    pub environment: Environment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

impl Default for Environment {
    fn default() -> Self {
        Environment::Development
    }
}

impl AuthConfig {
    pub fn from_env() -> Result<Self, AuthError> {
        let jwt_private_key = env::var("JWT_PRIVATE_KEY")
            .map_err(|_| AuthError::ConfigError("JWT_PRIVATE_KEY not set".to_string()))?;
        
        let jwt_public_key = env::var("JWT_PUBLIC_KEY")
            .map_err(|_| AuthError::ConfigError("JWT_PUBLIC_KEY not set".to_string()))?;
        
        let refresh_private_key = env::var("REFRESH_PRIVATE_KEY")
            .map_err(|_| AuthError::ConfigError("REFRESH_PRIVATE_KEY not set".to_string()))?;
        
        let refresh_public_key = env::var("REFRESH_PUBLIC_KEY")
            .map_err(|_| AuthError::ConfigError("REFRESH_PUBLIC_KEY not set".to_string()))?;

        let environment = match env::var("ENVIRONMENT")
            .unwrap_or_else(|_| "development".to_string())
            .to_lowercase()
            .as_str()
        {
            "production" => Environment::Production,
            "staging" => Environment::Staging,
            _ => Environment::Development,
        };

        let cors_origins = env::var("CORS_ORIGINS")
            .unwrap_or_else(|_| match environment {
                Environment::Development => "http://localhost:3000,http://localhost:5173".to_string(),
                Environment::Staging => "https://staging.prowzi.io".to_string(),
                Environment::Production => "https://prowzi.io".to_string(),
            })
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        Ok(Self {
            database_url: env::var("DATABASE_URL")
                .map_err(|_| AuthError::ConfigError("DATABASE_URL not set".to_string()))?,
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            jwt_private_key,
            jwt_public_key,
            refresh_private_key,
            refresh_public_key,
            jwt_expiry_seconds: env::var("JWT_EXPIRY_SECONDS")
                .unwrap_or_else(|_| "3600".to_string())
                .parse()
                .unwrap_or(3600),
            refresh_expiry_seconds: env::var("REFRESH_EXPIRY_SECONDS")
                .unwrap_or_else(|_| "604800".to_string())
                .parse()
                .unwrap_or(604800),
            rate_limit_requests: env::var("RATE_LIMIT_REQUESTS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100),
            rate_limit_window_seconds: env::var("RATE_LIMIT_WINDOW_SECONDS")
                .unwrap_or_else(|_| "60".to_string())
                .parse()
                .unwrap_or(60),
            cors_origins,
            environment,
        })
    }

    pub fn is_production(&self) -> bool {
        matches!(self.environment, Environment::Production)
    }

    pub fn is_development(&self) -> bool {
        matches!(self.environment, Environment::Development)
    }
}
