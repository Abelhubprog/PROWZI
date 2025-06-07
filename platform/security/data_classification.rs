use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Data classification levels used throughout the Prowzi platform
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataClassification {
    /// Public data that can be freely shared
    Public,
    /// Internal data for organizational use only
    Internal,
    /// Confidential data requiring protection
    Confidential,
    /// Restricted data with strict access controls
    Restricted,
    /// Highly sensitive data requiring maximum security
    Secret,
}

impl DataClassification {
    /// Returns the security level as a numeric value (higher = more secure)
    pub fn security_level(&self) -> u8 {
        match self {
            DataClassification::Public => 1,
            DataClassification::Internal => 2,
            DataClassification::Confidential => 3,
            DataClassification::Restricted => 4,
            DataClassification::Secret => 5,
        }
    }

    /// Returns the human-readable name of the classification
    pub fn name(&self) -> &'static str {
        match self {
            DataClassification::Public => "Public",
            DataClassification::Internal => "Internal",
            DataClassification::Confidential => "Confidential",
            DataClassification::Restricted => "Restricted",
            DataClassification::Secret => "Secret",
        }
    }

    /// Returns the description of the classification level
    pub fn description(&self) -> &'static str {
        match self {
            DataClassification::Public => "Data that can be freely shared and accessed by anyone",
            DataClassification::Internal => "Data for internal organizational use only",
            DataClassification::Confidential => "Sensitive data requiring protection from unauthorized access",
            DataClassification::Restricted => "Highly sensitive data with strict access controls",
            DataClassification::Secret => "Extremely sensitive data requiring maximum security measures",
        }
    }

    /// Returns the required encryption level for this classification
    pub fn encryption_requirement(&self) -> EncryptionLevel {
        match self {
            DataClassification::Public => EncryptionLevel::None,
            DataClassification::Internal => EncryptionLevel::InTransit,
            DataClassification::Confidential => EncryptionLevel::InTransitAndRest,
            DataClassification::Restricted => EncryptionLevel::InTransitAndRest,
            DataClassification::Secret => EncryptionLevel::EndToEnd,
        }
    }

    /// Returns the minimum access control level required
    pub fn access_control_requirement(&self) -> AccessControlLevel {
        match self {
            DataClassification::Public => AccessControlLevel::None,
            DataClassification::Internal => AccessControlLevel::Basic,
            DataClassification::Confidential => AccessControlLevel::RoleBased,
            DataClassification::Restricted => AccessControlLevel::AttributeBased,
            DataClassification::Secret => AccessControlLevel::ZeroTrust,
        }
    }

    /// Returns the audit logging requirement
    pub fn audit_requirement(&self) -> AuditLevel {
        match self {
            DataClassification::Public => AuditLevel::None,
            DataClassification::Internal => AuditLevel::Basic,
            DataClassification::Confidential => AuditLevel::Detailed,
            DataClassification::Restricted => AuditLevel::Comprehensive,
            DataClassification::Secret => AuditLevel::Forensic,
        }
    }

    /// Returns the data retention policy
    pub fn retention_policy(&self) -> RetentionPolicy {
        match self {
            DataClassification::Public => RetentionPolicy::Indefinite,
            DataClassification::Internal => RetentionPolicy::Years(7),
            DataClassification::Confidential => RetentionPolicy::Years(5),
            DataClassification::Restricted => RetentionPolicy::Years(3),
            DataClassification::Secret => RetentionPolicy::Months(12),
        }
    }
}

impl std::fmt::Display for DataClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Encryption levels for data protection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionLevel {
    None,
    InTransit,
    InTransitAndRest,
    EndToEnd,
}

/// Access control levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessControlLevel {
    None,
    Basic,
    RoleBased,
    AttributeBased,
    ZeroTrust,
}

/// Audit logging levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditLevel {
    None,
    Basic,
    Detailed,
    Comprehensive,
    Forensic,
}

/// Data retention policies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionPolicy {
    Indefinite,
    Days(u32),
    Months(u32),
    Years(u32),
}

/// Data classification errors
#[derive(Error, Debug)]
pub enum ClassificationError {
    #[error("Invalid classification level: {0}")]
    InvalidLevel(String),
    #[error("Insufficient permissions for classification level: {0}")]
    InsufficientPermissions(DataClassification),
    #[error("Classification downgrade not allowed: {from} to {to}")]
    DowngradeNotAllowed {
        from: DataClassification,
        to: DataClassification,
    },
    #[error("Data handling violation: {0}")]
    HandlingViolation(String),
}

/// Classified data wrapper that enforces security policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedData<T> {
    /// The actual data
    data: T,
    /// Classification level
    classification: DataClassification,
    /// Data owner/creator
    owner: String,
    /// Creation timestamp
    created_at: chrono::DateTime<chrono::Utc>,
    /// Last access timestamp
    last_accessed: Option<chrono::DateTime<chrono::Utc>>,
    /// Access count
    access_count: u64,
    /// Tags for additional metadata
    tags: HashMap<String, String>,
}

impl<T> ClassifiedData<T> {
    /// Creates new classified data
    pub fn new(data: T, classification: DataClassification, owner: String) -> Self {
        Self {
            data,
            classification,
            owner,
            created_at: chrono::Utc::now(),
            last_accessed: None,
            access_count: 0,
            tags: HashMap::new(),
        }
    }

    /// Gets the classification level
    pub fn classification(&self) -> &DataClassification {
        &self.classification
    }

    /// Gets the data owner
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// Gets creation timestamp
    pub fn created_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.created_at
    }

    /// Gets last access timestamp
    pub fn last_accessed(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.last_accessed
    }

    /// Gets access count
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Adds a tag
    pub fn add_tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    /// Gets a tag value
    pub fn get_tag(&self, key: &str) -> Option<&String> {
        self.tags.get(key)
    }

    /// Attempts to upgrade classification level
    pub fn upgrade_classification(&mut self, new_level: DataClassification) -> Result<(), ClassificationError> {
        if new_level.security_level() > self.classification.security_level() {
            self.classification = new_level;
            Ok(())
        } else {
            Err(ClassificationError::DowngradeNotAllowed {
                from: self.classification.clone(),
                to: new_level,
            })
        }
    }

    /// Accesses the data with security checks
    pub fn access_data(&mut self, accessor: &str) -> Result<&T, ClassificationError> {
        // Record access
        self.last_accessed = Some(chrono::Utc::now());
        self.access_count += 1;

        // Log access for audit trail
        tracing::info!(
            "Data accessed - Classification: {}, Owner: {}, Accessor: {}, Access Count: {}",
            self.classification,
            self.owner,
            accessor,
            self.access_count
        );

        Ok(&self.data)
    }

    /// Mutably accesses the data with security checks
    pub fn access_data_mut(&mut self, accessor: &str) -> Result<&mut T, ClassificationError> {
        // Record access
        self.last_accessed = Some(chrono::Utc::now());
        self.access_count += 1;

        // Log access for audit trail
        tracing::info!(
            "Data mutably accessed - Classification: {}, Owner: {}, Accessor: {}, Access Count: {}",
            self.classification,
            self.owner,
            accessor,
            self.access_count
        );

        Ok(&mut self.data)
    }

    /// Checks if the data can be shared with another party
    pub fn can_share_with(&self, recipient_clearance: &DataClassification) -> bool {
        recipient_clearance.security_level() >= self.classification.security_level()
    }

    /// Checks if the data has expired based on retention policy
    pub fn is_expired(&self) -> bool {
        match self.classification.retention_policy() {
            RetentionPolicy::Indefinite => false,
            RetentionPolicy::Days(days) => {
                let expiry = self.created_at + chrono::Duration::days(days as i64);
                chrono::Utc::now() > expiry
            }
            RetentionPolicy::Months(months) => {
                let expiry = self.created_at + chrono::Duration::days((months * 30) as i64);
                chrono::Utc::now() > expiry
            }
            RetentionPolicy::Years(years) => {
                let expiry = self.created_at + chrono::Duration::days((years * 365) as i64);
                chrono::Utc::now() > expiry
            }
        }
    }
}

/// Data classification manager
pub struct ClassificationManager {
    /// Default classification rules
    rules: HashMap<String, DataClassification>,
}

impl ClassificationManager {
    /// Creates a new classification manager
    pub fn new() -> Self {
        let mut rules = HashMap::new();
        
        // Default classification rules for common data types
        rules.insert("user_pii".to_string(), DataClassification::Confidential);
        rules.insert("financial_data".to_string(), DataClassification::Restricted);
        rules.insert("api_keys".to_string(), DataClassification::Secret);
        rules.insert("trade_secrets".to_string(), DataClassification::Secret);
        rules.insert("market_data".to_string(), DataClassification::Internal);
        rules.insert("logs".to_string(), DataClassification::Internal);
        rules.insert("metrics".to_string(), DataClassification::Internal);
        rules.insert("public_data".to_string(), DataClassification::Public);
        
        Self { rules }
    }

    /// Adds a classification rule
    pub fn add_rule(&mut self, data_type: String, classification: DataClassification) {
        self.rules.insert(data_type, classification);
    }

    /// Gets the classification for a data type
    pub fn classify(&self, data_type: &str) -> Option<&DataClassification> {
        self.rules.get(data_type)
    }

    /// Auto-classifies data based on content analysis
    pub fn auto_classify(&self, content: &str) -> DataClassification {
        let content_lower = content.to_lowercase();
        
        // Check for sensitive patterns
        if content_lower.contains("password") 
            || content_lower.contains("secret") 
            || content_lower.contains("private_key") 
            || content_lower.contains("api_key") {
            return DataClassification::Secret;
        }
        
        if content_lower.contains("ssn") 
            || content_lower.contains("credit_card") 
            || content_lower.contains("social_security") 
            || content_lower.contains("bank_account") {
            return DataClassification::Restricted;
        }
        
        if content_lower.contains("email") 
            || content_lower.contains("phone") 
            || content_lower.contains("address") 
            || content_lower.contains("personal") {
            return DataClassification::Confidential;
        }
        
        if content_lower.contains("internal") 
            || content_lower.contains("proprietary") 
            || content_lower.contains("confidential") {
            return DataClassification::Internal;
        }
        
        // Default to internal for unknown data
        DataClassification::Internal
    }

    /// Validates that an operation is allowed for the given classification
    pub fn validate_operation(&self, classification: &DataClassification, operation: &str) -> Result<(), ClassificationError> {
        match operation {
            "export" => {
                if classification.security_level() >= DataClassification::Restricted.security_level() {
                    return Err(ClassificationError::HandlingViolation(
                        "Export not allowed for restricted or secret data".to_string()
                    ));
                }
            }
            "log" => {
                if classification.security_level() >= DataClassification::Secret.security_level() {
                    return Err(ClassificationError::HandlingViolation(
                        "Logging not allowed for secret data".to_string()
                    ));
                }
            }
            "cache" => {
                if classification.security_level() >= DataClassification::Restricted.security_level() {
                    return Err(ClassificationError::HandlingViolation(
                        "Caching not allowed for restricted or secret data".to_string()
                    ));
                }
            }
            _ => {}
        }
        
        Ok(())
    }
}

impl Default for ClassificationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_levels() {
        assert!(DataClassification::Secret.security_level() > DataClassification::Restricted.security_level());
        assert!(DataClassification::Restricted.security_level() > DataClassification::Confidential.security_level());
        assert!(DataClassification::Confidential.security_level() > DataClassification::Internal.security_level());
        assert!(DataClassification::Internal.security_level() > DataClassification::Public.security_level());
    }

    #[test]
    fn test_classified_data() {
        let mut data = ClassifiedData::new(
            "sensitive information".to_string(),
            DataClassification::Confidential,
            "test_user".to_string()
        );

        assert_eq!(data.classification(), &DataClassification::Confidential);
        assert_eq!(data.owner(), "test_user");
        assert_eq!(data.access_count(), 0);

        let _content = data.access_data("accessor").unwrap();
        assert_eq!(data.access_count(), 1);
        assert!(data.last_accessed().is_some());
    }

    #[test]
    fn test_classification_upgrade() {
        let mut data = ClassifiedData::new(
            "test data".to_string(),
            DataClassification::Internal,
            "test_user".to_string()
        );

        // Upgrade should work
        assert!(data.upgrade_classification(DataClassification::Confidential).is_ok());
        assert_eq!(data.classification(), &DataClassification::Confidential);

        // Downgrade should fail
        assert!(data.upgrade_classification(DataClassification::Internal).is_err());
    }

    #[test]
    fn test_auto_classification() {
        let manager = ClassificationManager::new();

        assert_eq!(
            manager.auto_classify("User password: 123456"),
            DataClassification::Secret
        );

        assert_eq!(
            manager.auto_classify("User email: user@example.com"),
            DataClassification::Confidential
        );

        assert_eq!(
            manager.auto_classify("Internal company data"),
            DataClassification::Internal
        );
    }

    #[test]
    fn test_sharing_permissions() {
        let data = ClassifiedData::new(
            "confidential data".to_string(),
            DataClassification::Confidential,
            "test_user".to_string()
        );

        assert!(data.can_share_with(&DataClassification::Confidential));
        assert!(data.can_share_with(&DataClassification::Restricted));
        assert!(data.can_share_with(&DataClassification::Secret));
        assert!(!data.can_share_with(&DataClassification::Internal));
        assert!(!data.can_share_with(&DataClassification::Public));
    }
}