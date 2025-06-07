//! Security Module for Prowzi Agent Runtime
//! 
//! Provides comprehensive security controls, threat detection, and compliance
//! mechanisms for autonomous trading agents.

use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use uuid::Uuid;
use ring::{digest, hmac, rand, signature};
use base64::{Engine as _, engine::general_purpose};

/// Security configuration for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable API key authentication
    pub api_key_auth: bool,
    /// Enable rate limiting
    pub rate_limiting: bool,
    /// Maximum requests per minute
    pub max_requests_per_minute: u32,
    /// Enable IP whitelisting
    pub ip_whitelisting: bool,
    /// Allowed IP addresses
    pub allowed_ips: Vec<String>,
    /// Enable request signing
    pub request_signing: bool,
    /// Encryption key for sensitive data
    pub encryption_enabled: bool,
    /// Session timeout in minutes
    pub session_timeout_minutes: u32,
    /// Enable audit logging
    pub audit_logging: bool,
    /// Multi-factor authentication required
    pub mfa_required: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_key_auth: true,
            rate_limiting: true,
            max_requests_per_minute: 60,
            ip_whitelisting: false,
            allowed_ips: Vec::new(),
            request_signing: true,
            encryption_enabled: true,
            session_timeout_minutes: 30,
            audit_logging: true,
            mfa_required: false,
        }
    }
}

/// Security event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEvent {
    AuthenticationFailure {
        user_id: Option<String>,
        ip_address: IpAddr,
        reason: String,
    },
    UnauthorizedAccess {
        user_id: String,
        resource: String,
        action: String,
    },
    RateLimitExceeded {
        user_id: String,
        ip_address: IpAddr,
        endpoint: String,
        requests_count: u32,
    },
    SuspiciousActivity {
        user_id: String,
        activity_type: String,
        risk_score: f64,
        details: String,
    },
    DataBreach {
        affected_data: String,
        severity: SecuritySeverity,
        mitigation_actions: Vec<String>,
    },
    MaliciousRequest {
        ip_address: IpAddr,
        user_agent: String,
        payload_hash: String,
    },
    SystemIntrusion {
        detection_method: String,
        affected_systems: Vec<String>,
        attack_vector: String,
    },
}

/// Security severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Security audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub entry_id: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub action: String,
    pub resource: String,
    pub ip_address: Option<IpAddr>,
    pub user_agent: Option<String>,
    pub request_id: Option<String>,
    pub response_status: Option<u16>,
    pub details: HashMap<String, String>,
}

/// Rate limiting state
#[derive(Debug)]
struct RateLimitState {
    requests: Vec<DateTime<Utc>>,
    blocked_until: Option<DateTime<Utc>>,
}

/// Authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token_id: String,
    pub user_id: String,
    pub agent_id: Option<String>,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub permissions: Vec<String>,
    pub ip_address: Option<IpAddr>,
}

/// Security manager for the agent runtime
pub struct SecurityManager {
    config: SecurityConfig,
    rate_limits: HashMap<String, RateLimitState>,
    blocked_ips: HashSet<IpAddr>,
    security_events: Vec<(DateTime<Utc>, SecurityEvent)>,
    audit_log: Vec<AuditEntry>,
    active_tokens: HashMap<String, AuthToken>,
    signing_key: hmac::Key,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Result<Self, SecurityError> {
        let rng = rand::SystemRandom::new();
        let key_bytes = rand::generate(&rng)
            .map_err(|_| SecurityError::CryptoError("Failed to generate signing key".to_string()))?;
        let signing_key = hmac::Key::new(hmac::HMAC_SHA256, key_bytes.expose());

        Ok(Self {
            config,
            rate_limits: HashMap::new(),
            blocked_ips: HashSet::new(),
            security_events: Vec::new(),
            audit_log: Vec::new(),
            active_tokens: HashMap::new(),
            signing_key,
        })
    }

    /// Authenticate a request
    pub async fn authenticate(
        &mut self,
        token: &str,
        ip_address: IpAddr,
    ) -> Result<AuthToken, SecurityError> {
        // Check if IP is blocked
        if self.blocked_ips.contains(&ip_address) {
            self.record_security_event(SecurityEvent::AuthenticationFailure {
                user_id: None,
                ip_address,
                reason: "IP address blocked".to_string(),
            }).await;
            return Err(SecurityError::AccessDenied("IP blocked".to_string()));
        }

        // Validate token
        let auth_token = self.active_tokens.get(token)
            .ok_or(SecurityError::InvalidToken)?
            .clone();

        // Check token expiration
        if auth_token.expires_at < Utc::now() {
            self.active_tokens.remove(token);
            return Err(SecurityError::TokenExpired);
        }

        // Check IP address if configured
        if self.config.ip_whitelisting {
            if let Some(token_ip) = auth_token.ip_address {
                if token_ip != ip_address {
                    self.record_security_event(SecurityEvent::SuspiciousActivity {
                        user_id: auth_token.user_id.clone(),
                        activity_type: "IP address mismatch".to_string(),
                        risk_score: 0.8,
                        details: format!("Token IP: {}, Request IP: {}", token_ip, ip_address),
                    }).await;
                    return Err(SecurityError::AccessDenied("IP mismatch".to_string()));
                }
            }
        }

        Ok(auth_token)
    }

    /// Check rate limits for a user
    pub async fn check_rate_limit(
        &mut self,
        user_id: &str,
        ip_address: IpAddr,
    ) -> Result<(), SecurityError> {
        if !self.config.rate_limiting {
            return Ok(());
        }

        let now = Utc::now();
        let window_start = now - Duration::minutes(1);

        let rate_state = self.rate_limits.entry(user_id.to_string())
            .or_insert_with(|| RateLimitState {
                requests: Vec::new(),
                blocked_until: None,
            });

        // Check if user is temporarily blocked
        if let Some(blocked_until) = rate_state.blocked_until {
            if now < blocked_until {
                return Err(SecurityError::RateLimitExceeded);
            } else {
                rate_state.blocked_until = None;
            }
        }

        // Remove old requests outside the window
        rate_state.requests.retain(|&timestamp| timestamp > window_start);

        // Check if rate limit is exceeded
        if rate_state.requests.len() >= self.config.max_requests_per_minute as usize {
            rate_state.blocked_until = Some(now + Duration::minutes(5)); // Block for 5 minutes
            
            self.record_security_event(SecurityEvent::RateLimitExceeded {
                user_id: user_id.to_string(),
                ip_address,
                endpoint: "general".to_string(),
                requests_count: rate_state.requests.len() as u32,
            }).await;

            return Err(SecurityError::RateLimitExceeded);
        }

        // Add current request
        rate_state.requests.push(now);
        Ok(())
    }

    /// Validate request signature
    pub fn validate_signature(
        &self,
        message: &[u8],
        signature: &str,
    ) -> Result<(), SecurityError> {
        if !self.config.request_signing {
            return Ok(());
        }

        let signature_bytes = general_purpose::STANDARD.decode(signature)
            .map_err(|_| SecurityError::InvalidSignature)?;

        hmac::verify(&self.signing_key, message, &signature_bytes)
            .map_err(|_| SecurityError::InvalidSignature)?;

        Ok(())
    }

    /// Create a new authentication token
    pub fn create_token(
        &mut self,
        user_id: String,
        agent_id: Option<String>,
        permissions: Vec<String>,
        ip_address: Option<IpAddr>,
    ) -> AuthToken {
        let token = AuthToken {
            token_id: Uuid::new_v4().to_string(),
            user_id,
            agent_id,
            issued_at: Utc::now(),
            expires_at: Utc::now() + Duration::minutes(self.config.session_timeout_minutes as i64),
            permissions,
            ip_address,
        };

        self.active_tokens.insert(token.token_id.clone(), token.clone());
        token
    }

    /// Revoke an authentication token
    pub fn revoke_token(&mut self, token_id: &str) -> Result<(), SecurityError> {
        self.active_tokens.remove(token_id)
            .ok_or(SecurityError::TokenNotFound)?;
        Ok(())
    }

    /// Check if user has permission for an action
    pub fn check_permission(
        &self,
        token: &AuthToken,
        required_permission: &str,
    ) -> Result<(), SecurityError> {
        if token.permissions.contains(&"admin".to_string()) ||
           token.permissions.contains(&required_permission.to_string()) {
            Ok(())
        } else {
            Err(SecurityError::InsufficientPermissions)
        }
    }

    /// Record a security event
    pub async fn record_security_event(&mut self, event: SecurityEvent) {
        self.security_events.push((Utc::now(), event));
        
        // Keep only recent events (last 7 days)
        let cutoff = Utc::now() - Duration::days(7);
        self.security_events.retain(|(timestamp, _)| *timestamp > cutoff);
    }

    /// Add audit log entry
    pub fn add_audit_entry(&mut self, entry: AuditEntry) {
        if self.config.audit_logging {
            self.audit_log.push(entry);
            
            // Keep only recent entries (last 30 days)
            let cutoff = Utc::now() - Duration::days(30);
            self.audit_log.retain(|entry| entry.timestamp > cutoff);
        }
    }

    /// Block an IP address
    pub async fn block_ip(&mut self, ip_address: IpAddr, reason: String) {
        self.blocked_ips.insert(ip_address);
        
        self.record_security_event(SecurityEvent::SuspiciousActivity {
            user_id: "system".to_string(),
            activity_type: "IP blocked".to_string(),
            risk_score: 0.9,
            details: format!("IP {} blocked: {}", ip_address, reason),
        }).await;
    }

    /// Unblock an IP address
    pub fn unblock_ip(&mut self, ip_address: IpAddr) {
        self.blocked_ips.remove(&ip_address);
    }

    /// Get recent security events
    pub fn get_security_events(&self, severity_filter: Option<SecuritySeverity>) -> Vec<&(DateTime<Utc>, SecurityEvent)> {
        self.security_events.iter()
            .filter(|(_, event)| {
                severity_filter.as_ref().map_or(true, |filter| {
                    self.get_event_severity(event) >= *filter
                })
            })
            .collect()
    }

    /// Get audit log entries
    pub fn get_audit_entries(&self, limit: Option<usize>) -> Vec<&AuditEntry> {
        let entries: Vec<&AuditEntry> = self.audit_log.iter().collect();
        if let Some(limit) = limit {
            entries.into_iter().take(limit).collect()
        } else {
            entries
        }
    }

    /// Encrypt sensitive data
    pub fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, SecurityError> {
        if !self.config.encryption_enabled {
            return Ok(data.to_vec());
        }

        // TODO: Implement actual encryption (AES-GCM)
        // For now, return base64 encoded data as placeholder
        Ok(general_purpose::STANDARD.encode(data).into_bytes())
    }

    /// Decrypt sensitive data
    pub fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, SecurityError> {
        if !self.config.encryption_enabled {
            return Ok(encrypted_data.to_vec());
        }

        // TODO: Implement actual decryption (AES-GCM)
        // For now, return base64 decoded data as placeholder
        let encoded = String::from_utf8(encrypted_data.to_vec())
            .map_err(|_| SecurityError::DecryptionFailed)?;
        general_purpose::STANDARD.decode(encoded)
            .map_err(|_| SecurityError::DecryptionFailed)
    }

    /// Generate secure hash of data
    pub fn hash_data(&self, data: &[u8]) -> String {
        let digest = digest::digest(&digest::SHA256, data);
        general_purpose::STANDARD.encode(digest.as_ref())
    }

    /// Get event severity for filtering
    fn get_event_severity(&self, event: &SecurityEvent) -> SecuritySeverity {
        match event {
            SecurityEvent::AuthenticationFailure { .. } => SecuritySeverity::Medium,
            SecurityEvent::UnauthorizedAccess { .. } => SecuritySeverity::High,
            SecurityEvent::RateLimitExceeded { .. } => SecuritySeverity::Low,
            SecurityEvent::SuspiciousActivity { risk_score, .. } => {
                if *risk_score > 0.8 { SecuritySeverity::High }
                else if *risk_score > 0.5 { SecuritySeverity::Medium }
                else { SecuritySeverity::Low }
            },
            SecurityEvent::DataBreach { severity, .. } => severity.clone(),
            SecurityEvent::MaliciousRequest { .. } => SecuritySeverity::High,
            SecurityEvent::SystemIntrusion { .. } => SecuritySeverity::Critical,
        }
    }

    /// Update security configuration
    pub fn update_config(&mut self, config: SecurityConfig) {
        self.config = config;
    }

    /// Get current security configuration
    pub fn get_config(&self) -> &SecurityConfig {
        &self.config
    }
}

/// Security-related errors
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("Access denied: {0}")]
    AccessDenied(String),

    #[error("Invalid authentication token")]
    InvalidToken,

    #[error("Authentication token expired")]
    TokenExpired,

    #[error("Authentication token not found")]
    TokenNotFound,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Insufficient permissions")]
    InsufficientPermissions,

    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    #[error("Encryption failed")]
    EncryptionFailed,

    #[error("Decryption failed")]
    DecryptionFailed,

    #[error("Security policy violation: {0}")]
    PolicyViolation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[tokio::test]
    async fn test_authentication() {
        let config = SecurityConfig::default();
        let mut security_manager = SecurityManager::new(config).unwrap();

        let ip = IpAddr::from_str("127.0.0.1").unwrap();
        let token = security_manager.create_token(
            "test_user".to_string(),
            Some("test_agent".to_string()),
            vec!["read".to_string()],
            Some(ip),
        );

        let auth_result = security_manager.authenticate(&token.token_id, ip).await;
        assert!(auth_result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let mut config = SecurityConfig::default();
        config.max_requests_per_minute = 2;
        
        let mut security_manager = SecurityManager::new(config).unwrap();
        let ip = IpAddr::from_str("127.0.0.1").unwrap();

        // First two requests should succeed
        assert!(security_manager.check_rate_limit("test_user", ip).await.is_ok());
        assert!(security_manager.check_rate_limit("test_user", ip).await.is_ok());

        // Third request should fail
        assert!(security_manager.check_rate_limit("test_user", ip).await.is_err());
    }

    #[test]
    fn test_permission_check() {
        let config = SecurityConfig::default();
        let security_manager = SecurityManager::new(config).unwrap();

        let token = AuthToken {
            token_id: "test".to_string(),
            user_id: "test_user".to_string(),
            agent_id: None,
            issued_at: Utc::now(),
            expires_at: Utc::now() + Duration::hours(1),
            permissions: vec!["read".to_string()],
            ip_address: None,
        };

        assert!(security_manager.check_permission(&token, "read").is_ok());
        assert!(security_manager.check_permission(&token, "write").is_err());
    }

    #[test]
    fn test_data_encryption() {
        let config = SecurityConfig::default();
        let security_manager = SecurityManager::new(config).unwrap();

        let data = b"sensitive data";
        let encrypted = security_manager.encrypt_data(data).unwrap();
        let decrypted = security_manager.decrypt_data(&encrypted).unwrap();

        assert_eq!(data, decrypted.as_slice());
    }

    #[test]
    fn test_data_hashing() {
        let config = SecurityConfig::default();
        let security_manager = SecurityManager::new(config).unwrap();

        let data = b"test data";
        let hash1 = security_manager.hash_data(data);
        let hash2 = security_manager.hash_data(data);

        assert_eq!(hash1, hash2);
        assert!(!hash1.is_empty());
    }

    #[tokio::test]
    async fn test_ip_blocking() {
        let config = SecurityConfig::default();
        let mut security_manager = SecurityManager::new(config).unwrap();

        let ip = IpAddr::from_str("192.168.1.100").unwrap();
        
        security_manager.block_ip(ip, "Suspicious activity".to_string()).await;
        
        let token = security_manager.create_token(
            "test_user".to_string(),
            None,
            vec!["read".to_string()],
            Some(ip),
        );

        let auth_result = security_manager.authenticate(&token.token_id, ip).await;
        assert!(auth_result.is_err());
    }
}
