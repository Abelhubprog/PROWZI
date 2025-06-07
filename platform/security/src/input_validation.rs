//! Comprehensive Input Validation and Sanitization
//! 
//! Production-grade input validation for all user inputs across the Prowzi platform.
//! Prevents injection attacks, data corruption, and security vulnerabilities.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Invalid length: expected {expected}, got {actual}")]
    InvalidLength { expected: String, actual: usize },
    
    #[error("Invalid format: {field} does not match required pattern")]
    InvalidFormat { field: String },
    
    #[error("Invalid range: {field} must be between {min} and {max}")]
    InvalidRange { field: String, min: String, max: String },
    
    #[error("Forbidden characters detected in {field}")]
    ForbiddenCharacters { field: String },
    
    #[error("SQL injection attempt detected in {field}")]
    SqlInjectionAttempt { field: String },
    
    #[error("XSS attempt detected in {field}")]
    XssAttempt { field: String },
    
    #[error("Path traversal attempt detected in {field}")]
    PathTraversalAttempt { field: String },
    
    #[error("Invalid email format")]
    InvalidEmail,
    
    #[error("Invalid URL format")]
    InvalidUrl,
    
    #[error("Invalid JSON structure")]
    InvalidJson,
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// Input validator with comprehensive security checks
pub struct InputValidator {
    sql_injection_patterns: Vec<Regex>,
    xss_patterns: Vec<Regex>,
    path_traversal_patterns: Vec<Regex>,
    email_regex: Regex,
    url_regex: Regex,
}

impl InputValidator {
    pub fn new() -> Self {
        let sql_injection_patterns = vec![
            Regex::new(r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)").unwrap(),
            Regex::new(r"(?i)(script|javascript|vbscript|onload|onerror|onclick)").unwrap(),
            Regex::new(r"['\";]").unwrap(),
            Regex::new(r"--").unwrap(),
            Regex::new(r"/\*.*\*/").unwrap(),
        ];
        
        let xss_patterns = vec![
            Regex::new(r"(?i)<script[^>]*>.*?</script>").unwrap(),
            Regex::new(r"(?i)javascript:").unwrap(),
            Regex::new(r"(?i)on\w+\s*=").unwrap(),
            Regex::new(r"(?i)<iframe[^>]*>").unwrap(),
            Regex::new(r"(?i)<object[^>]*>").unwrap(),
            Regex::new(r"(?i)<embed[^>]*>").unwrap(),
        ];
        
        let path_traversal_patterns = vec![
            Regex::new(r"\.\.(/|\\)").unwrap(),
            Regex::new(r"\.\.%2f").unwrap(),
            Regex::new(r"\.\.%5c").unwrap(),
            Regex::new(r"%2e%2e%2f").unwrap(),
            Regex::new(r"%2e%2e%5c").unwrap(),
        ];
        
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
        let url_regex = Regex::new(r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$").unwrap();
        
        Self {
            sql_injection_patterns,
            xss_patterns,
            path_traversal_patterns,
            email_regex,
            url_regex,
        }
    }
    
    /// Validate and sanitize mission name
    pub fn validate_mission_name(&self, name: &str) -> Result<String, ValidationError> {
        // Length validation
        if name.len() < 3 || name.len() > 100 {
            return Err(ValidationError::InvalidLength {
                expected: "3-100 characters".to_string(),
                actual: name.len(),
            });
        }
        
        // Pattern validation - alphanumeric, spaces, hyphens, underscores only
        let valid_pattern = Regex::new(r"^[a-zA-Z0-9\s\-_]+$").unwrap();
        if !valid_pattern.is_match(name) {
            return Err(ValidationError::InvalidFormat {
                field: "mission_name".to_string(),
            });
        }
        
        // Security checks
        self.check_injection_attempts(name, "mission_name")?;
        
        // Sanitize and return
        Ok(name.trim().to_string())
    }
    
    /// Validate trading amount in USDC
    pub fn validate_trading_amount(&self, amount: f64) -> Result<f64, ValidationError> {
        // Minimum $10 requirement
        if amount < 10.0 {
            return Err(ValidationError::InvalidRange {
                field: "trading_amount".to_string(),
                min: "10.0".to_string(),
                max: "1000000.0".to_string(),
            });
        }
        
        // Maximum $1M limit for safety
        if amount > 1_000_000.0 {
            return Err(ValidationError::InvalidRange {
                field: "trading_amount".to_string(),
                min: "10.0".to_string(),
                max: "1000000.0".to_string(),
            });
        }
        
        // Check for reasonable precision (max 6 decimal places)
        let rounded = (amount * 1_000_000.0).round() / 1_000_000.0;
        Ok(rounded)
    }
    
    /// Validate Solana wallet address
    pub fn validate_wallet_address(&self, address: &str) -> Result<String, ValidationError> {
        // Solana addresses are base58 encoded, 32-44 characters
        if address.len() < 32 || address.len() > 44 {
            return Err(ValidationError::InvalidLength {
                expected: "32-44 characters".to_string(),
                actual: address.len(),
            });
        }
        
        // Base58 character set validation
        let base58_pattern = Regex::new(r"^[1-9A-HJ-NP-Za-km-z]+$").unwrap();
        if !base58_pattern.is_match(address) {
            return Err(ValidationError::InvalidFormat {
                field: "wallet_address".to_string(),
            });
        }
        
        // Additional security checks
        self.check_injection_attempts(address, "wallet_address")?;
        
        Ok(address.to_string())
    }
    
    /// Validate email address
    pub fn validate_email(&self, email: &str) -> Result<String, ValidationError> {
        let email = email.trim().to_lowercase();
        
        if !self.email_regex.is_match(&email) {
            return Err(ValidationError::InvalidEmail);
        }
        
        // Security checks
        self.check_injection_attempts(&email, "email")?;
        
        Ok(email)
    }
    
    /// Validate JSON configuration
    pub fn validate_json_config(&self, json_str: &str) -> Result<serde_json::Value, ValidationError> {
        // Parse JSON
        let json_value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|_| ValidationError::InvalidJson)?;
        
        // Security checks on serialized form
        self.check_injection_attempts(json_str, "json_config")?;
        
        // Validate structure doesn't contain dangerous keys
        self.validate_json_structure(&json_value)?;
        
        Ok(json_value)
    }
    
    /// Validate IP address
    pub fn validate_ip_address(&self, ip: &str) -> Result<IpAddr, ValidationError> {
        ip.parse::<IpAddr>().map_err(|_| ValidationError::InvalidFormat {
            field: "ip_address".to_string(),
        })
    }
    
    /// Validate user agent string
    pub fn validate_user_agent(&self, user_agent: &str) -> Result<String, ValidationError> {
        // Length validation
        if user_agent.len() > 500 {
            return Err(ValidationError::InvalidLength {
                expected: "0-500 characters".to_string(),
                actual: user_agent.len(),
            });
        }
        
        // Security checks
        self.check_injection_attempts(user_agent, "user_agent")?;
        
        // Sanitize potentially dangerous characters
        let sanitized = user_agent
            .chars()
            .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
            .collect::<String>();
        
        Ok(sanitized)
    }
    
    /// Validate UUID
    pub fn validate_uuid(&self, uuid_str: &str) -> Result<Uuid, ValidationError> {
        Uuid::parse_str(uuid_str).map_err(|_| ValidationError::InvalidFormat {
            field: "uuid".to_string(),
        })
    }
    
    /// Comprehensive security check for injection attempts
    fn check_injection_attempts(&self, input: &str, field: &str) -> Result<(), ValidationError> {
        let input_lower = input.to_lowercase();
        
        // Check for SQL injection patterns
        for pattern in &self.sql_injection_patterns {
            if pattern.is_match(&input_lower) {
                return Err(ValidationError::SqlInjectionAttempt {
                    field: field.to_string(),
                });
            }
        }
        
        // Check for XSS patterns
        for pattern in &self.xss_patterns {
            if pattern.is_match(&input_lower) {
                return Err(ValidationError::XssAttempt {
                    field: field.to_string(),
                });
            }
        }
        
        // Check for path traversal patterns
        for pattern in &self.path_traversal_patterns {
            if pattern.is_match(&input_lower) {
                return Err(ValidationError::PathTraversalAttempt {
                    field: field.to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate JSON structure for dangerous keys
    fn validate_json_structure(&self, value: &serde_json::Value) -> Result<(), ValidationError> {
        match value {
            serde_json::Value::Object(obj) => {
                for (key, val) in obj {
                    // Check for dangerous keys
                    let dangerous_keys = ["__proto__", "constructor", "prototype", "eval", "function"];
                    if dangerous_keys.contains(&key.as_str()) {
                        return Err(ValidationError::ForbiddenCharacters {
                            field: "json_key".to_string(),
                        });
                    }
                    
                    // Recursively validate nested objects
                    self.validate_json_structure(val)?;
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.validate_json_structure(item)?;
                }
            }
            serde_json::Value::String(s) => {
                // Validate string values for injection attempts
                self.check_injection_attempts(s, "json_string")?;
            }
            _ => {} // Numbers, booleans, null are safe
        }
        
        Ok(())
    }
}

/// Rate limiter for input validation
pub struct ValidationRateLimiter {
    requests: HashMap<String, Vec<std::time::Instant>>,
    window_size: std::time::Duration,
    max_requests: usize,
}

impl ValidationRateLimiter {
    pub fn new(window_size: std::time::Duration, max_requests: usize) -> Self {
        Self {
            requests: HashMap::new(),
            window_size,
            max_requests,
        }
    }
    
    pub fn check_rate_limit(&mut self, key: &str) -> Result<(), ValidationError> {
        let now = std::time::Instant::now();
        let requests = self.requests.entry(key.to_string()).or_insert_with(Vec::new);
        
        // Remove old requests outside the window
        requests.retain(|&time| now.duration_since(time) <= self.window_size);
        
        if requests.len() >= self.max_requests {
            return Err(ValidationError::RateLimitExceeded);
        }
        
        requests.push(now);
        Ok(())
    }
}

/// Sanitize output to prevent data leakage
pub struct OutputSanitizer;

impl OutputSanitizer {
    /// Sanitize error messages to prevent information disclosure
    pub fn sanitize_error_message(error: &ValidationError) -> String {
        match error {
            ValidationError::SqlInjectionAttempt { .. } => "Invalid input format".to_string(),
            ValidationError::XssAttempt { .. } => "Invalid input format".to_string(),
            ValidationError::PathTraversalAttempt { .. } => "Invalid input format".to_string(),
            _ => error.to_string(),
        }
    }
    
    /// Remove sensitive data from logs
    pub fn sanitize_log_data(data: &str) -> String {
        let sensitive_patterns = vec![
            Regex::new(r"password=\w+").unwrap(),
            Regex::new(r"token=\w+").unwrap(),
            Regex::new(r"key=\w+").unwrap(),
            Regex::new(r"secret=\w+").unwrap(),
        ];
        
        let mut sanitized = data.to_string();
        for pattern in sensitive_patterns {
            sanitized = pattern.replace_all(&sanitized, "[REDACTED]").to_string();
        }
        
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mission_name_validation() {
        let validator = InputValidator::new();
        
        // Valid names
        assert!(validator.validate_mission_name("My Trading Mission").is_ok());
        assert!(validator.validate_mission_name("Trade_Bot_01").is_ok());
        
        // Invalid names
        assert!(validator.validate_mission_name("sh").is_err()); // Too short
        assert!(validator.validate_mission_name("a".repeat(101).as_str()).is_err()); // Too long
        assert!(validator.validate_mission_name("Mission<script>alert(1)</script>").is_err()); // XSS
        assert!(validator.validate_mission_name("Mission'; DROP TABLE missions; --").is_err()); // SQL injection
    }
    
    #[test]
    fn test_trading_amount_validation() {
        let validator = InputValidator::new();
        
        // Valid amounts
        assert!(validator.validate_trading_amount(10.0).is_ok());
        assert!(validator.validate_trading_amount(100.50).is_ok());
        assert!(validator.validate_trading_amount(1000.0).is_ok());
        
        // Invalid amounts
        assert!(validator.validate_trading_amount(9.99).is_err()); // Below minimum
        assert!(validator.validate_trading_amount(1_000_001.0).is_err()); // Above maximum
    }
    
    #[test]
    fn test_wallet_address_validation() {
        let validator = InputValidator::new();
        
        // Valid Solana address (example)
        assert!(validator.validate_wallet_address("11111111111111111111111111111112").is_ok());
        
        // Invalid addresses
        assert!(validator.validate_wallet_address("short").is_err()); // Too short
        assert!(validator.validate_wallet_address("0OIl").is_err()); // Invalid base58 characters
        assert!(validator.validate_wallet_address("'; DROP TABLE accounts; --").is_err()); // SQL injection
    }
    
    #[test]
    fn test_email_validation() {
        let validator = InputValidator::new();
        
        // Valid emails
        assert!(validator.validate_email("user@example.com").is_ok());
        assert!(validator.validate_email("test.email+tag@domain.co.uk").is_ok());
        
        // Invalid emails
        assert!(validator.validate_email("invalid-email").is_err());
        assert!(validator.validate_email("user@').is_err());
        assert!(validator.validate_email("user@example.com<script>alert(1)</script>").is_err());
    }
}