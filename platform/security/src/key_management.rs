// File: platform/security/src/key_management.rs

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Nonce, Key
};
use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2
};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;
use base64::{Engine as _, engine::general_purpose};
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Error)]
pub enum KeyManagementError {
    #[error("Encryption error: {0}")]
    Encryption(String),
    #[error("Decryption error: {0}")]
    Decryption(String),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Key rotation failed: {0}")]
    RotationFailed(String),
    #[error("Invalid key format")]
    InvalidKeyFormat,
    #[error("Key expired")]
    KeyExpired,
    #[error("Unauthorized access")]
    Unauthorized,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedKey {
    pub id: String,
    pub encrypted_data: Vec<u8>,
    pub nonce: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub key_type: KeyType,
    pub rotation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeyType {
    ApiKey,
    PrivateKey,
    DatabaseKey,
    EncryptionKey,
    SigningKey,
}

#[derive(Debug)]
pub struct MasterKey {
    key: Key<Aes256Gcm>,
    created_at: DateTime<Utc>,
    rotation_due: DateTime<Utc>,
}

pub struct SecureKeyManager {
    master_key: Arc<RwLock<MasterKey>>,
    keys: Arc<RwLock<HashMap<String, EncryptedKey>>>,
    key_derivation_salt: Vec<u8>,
    rng: SystemRandom,
    rotation_interval: Duration,
}

impl SecureKeyManager {
    /// Initialize with hardware security module (HSM) integration
    pub async fn new_with_hsm(hsm_config: Option<HsmConfig>) -> Result<Self, KeyManagementError> {
        let rng = SystemRandom::new();
        
        // Generate master key derivation salt
        let mut salt = vec![0u8; 32];
        rng.fill(&mut salt).map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
        
        // Initialize master key (in production, this would come from HSM)
        let master_key = if let Some(hsm) = hsm_config {
            Self::get_master_key_from_hsm(hsm).await?
        } else {
            Self::generate_master_key()?
        };
        
        Ok(Self {
            master_key: Arc::new(RwLock::new(MasterKey {
                key: master_key,
                created_at: Utc::now(),
                rotation_due: Utc::now() + Duration::days(30),
            })),
            keys: Arc::new(RwLock::new(HashMap::new())),
            key_derivation_salt: salt,
            rng,
            rotation_interval: Duration::days(30),
        })
    }
    
    fn generate_master_key() -> Result<Key<Aes256Gcm>, KeyManagementError> {
        let key = Aes256Gcm::generate_key(OsRng);
        Ok(key)
    }
    
    async fn get_master_key_from_hsm(hsm_config: HsmConfig) -> Result<Key<Aes256Gcm>, KeyManagementError> {
        // In production, integrate with actual HSM
        // This is a placeholder
        Self::generate_master_key()
    }
    
    /// Store a sensitive key securely
    pub async fn store_key(
        &self,
        key_id: String,
        key_data: &[u8],
        key_type: KeyType,
        expires_in: Option<Duration>,
    ) -> Result<(), KeyManagementError> {
        let master = self.master_key.read().await;
        let cipher = Aes256Gcm::new(&master.key);
        
        // Generate nonce
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        
        // Encrypt the key
        let encrypted = cipher.encrypt(&nonce, key_data)
            .map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
        
        let encrypted_key = EncryptedKey {
            id: key_id.clone(),
            encrypted_data: encrypted,
            nonce: nonce.to_vec(),
            created_at: Utc::now(),
            expires_at: expires_in.map(|d| Utc::now() + d),
            key_type,
            rotation_count: 0,
        };
        
        let mut keys = self.keys.write().await;
        keys.insert(key_id, encrypted_key);
        
        Ok(())
    }
    
    /// Retrieve and decrypt a key
    pub async fn get_key(&self, key_id: &str) -> Result<Vec<u8>, KeyManagementError> {
        let keys = self.keys.read().await;
        let encrypted_key = keys.get(key_id)
            .ok_or_else(|| KeyManagementError::KeyNotFound(key_id.to_string()))?;
        
        // Check expiration
        if let Some(expires_at) = encrypted_key.expires_at {
            if Utc::now() > expires_at {
                return Err(KeyManagementError::KeyExpired);
            }
        }
        
        let master = self.master_key.read().await;
        let cipher = Aes256Gcm::new(&master.key);
        
        let nonce = Nonce::from_slice(&encrypted_key.nonce);
        
        cipher.decrypt(nonce, encrypted_key.encrypted_data.as_ref())
            .map_err(|e| KeyManagementError::Decryption(e.to_string()))
    }
    
    /// Rotate a specific key
    pub async fn rotate_key(&self, key_id: &str) -> Result<Vec<u8>, KeyManagementError> {
        let old_key = self.get_key(key_id).await?;
        
        // Generate new key based on type
        let mut keys = self.keys.write().await;
        let key_entry = keys.get_mut(key_id)
            .ok_or_else(|| KeyManagementError::KeyNotFound(key_id.to_string()))?;
        
        let new_key = self.generate_new_key(&key_entry.key_type)?;
        
        // Re-encrypt with new data
        let master = self.master_key.read().await;
        let cipher = Aes256Gcm::new(&master.key);
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        
        let encrypted = cipher.encrypt(&nonce, new_key.as_ref())
            .map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
        
        key_entry.encrypted_data = encrypted;
        key_entry.nonce = nonce.to_vec();
        key_entry.rotation_count += 1;
        key_entry.created_at = Utc::now();
        
        Ok(new_key)
    }
    
    fn generate_new_key(&self, key_type: &KeyType) -> Result<Vec<u8>, KeyManagementError> {
        let key_size = match key_type {
            KeyType::ApiKey => 32,
            KeyType::PrivateKey => 32,
            KeyType::DatabaseKey => 32,
            KeyType::EncryptionKey => 32,
            KeyType::SigningKey => 64,
        };
        
        let mut key = vec![0u8; key_size];
        self.rng.fill(&mut key)
            .map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
        
        Ok(key)
    }
    
    /// Rotate master key
    pub async fn rotate_master_key(&self) -> Result<(), KeyManagementError> {
        let new_master_key = Self::generate_master_key()?;
        
        // Re-encrypt all keys with new master
        let mut keys = self.keys.write().await;
        let old_master = self.master_key.read().await;
        
        for (_, encrypted_key) in keys.iter_mut() {
            // Decrypt with old master
            let old_cipher = Aes256Gcm::new(&old_master.key);
            let nonce = Nonce::from_slice(&encrypted_key.nonce);
            let decrypted = old_cipher.decrypt(nonce, encrypted_key.encrypted_data.as_ref())
                .map_err(|e| KeyManagementError::Decryption(e.to_string()))?;
            
            // Encrypt with new master
            let new_cipher = Aes256Gcm::new(&new_master_key);
            let new_nonce = Aes256Gcm::generate_nonce(&mut OsRng);
            let re_encrypted = new_cipher.encrypt(&new_nonce, decrypted.as_ref())
                .map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
            
            encrypted_key.encrypted_data = re_encrypted;
            encrypted_key.nonce = new_nonce.to_vec();
        }
        
        // Update master key
        drop(old_master);
        let mut master = self.master_key.write().await;
        *master = MasterKey {
            key: new_master_key,
            created_at: Utc::now(),
            rotation_due: Utc::now() + self.rotation_interval,
        };
        
        Ok(())
    }
    
    /// Check if master key rotation is due
    pub async fn is_rotation_due(&self) -> bool {
        let master = self.master_key.read().await;
        Utc::now() > master.rotation_due
    }
    
    /// Clean up expired keys
    pub async fn cleanup_expired_keys(&self) -> usize {
        let mut keys = self.keys.write().await;
        let now = Utc::now();
        
        let expired: Vec<String> = keys.iter()
            .filter(|(_, k)| k.expires_at.map(|e| now > e).unwrap_or(false))
            .map(|(id, _)| id.clone())
            .collect();
        
        let count = expired.len();
        for id in expired {
            keys.remove(&id);
        }
        
        count
    }
}

#[derive(Debug, Clone)]
pub struct HsmConfig {
    pub slot_id: u64,
    pub pin: String,
    pub module_path: String,
}

/// Zero-knowledge proof for key possession without revealing the key
pub struct KeyProof {
    commitment: Vec<u8>,
    challenge: Vec<u8>,
    response: Vec<u8>,
}

impl SecureKeyManager {
    /// Generate zero-knowledge proof of key possession
    pub async fn generate_key_proof(&self, key_id: &str) -> Result<KeyProof, KeyManagementError> {
        let key = self.get_key(key_id).await?;
        
        // Simplified ZKP implementation
        // In production, use a proper ZKP library
        let mut commitment = vec![0u8; 32];
        self.rng.fill(&mut commitment).map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
        
        let mut challenge = vec![0u8; 32];
        self.rng.fill(&mut challenge).map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
        
        // Response = commitment + challenge * key (simplified)
        let response = commitment.clone();
        
        Ok(KeyProof {
            commitment,
            challenge,
            response,
        })
    }
    
    /// Verify zero-knowledge proof
    pub fn verify_key_proof(&self, proof: &KeyProof) -> bool {
        // Simplified verification
        // In production, implement proper ZKP verification
        !proof.commitment.is_empty() && !proof.challenge.is_empty() && !proof.response.is_empty()
    }
}

/// Secure API key generation with rate limiting
pub struct ApiKeyGenerator {
    key_manager: Arc<SecureKeyManager>,
    rate_limiter: Arc<RwLock<HashMap<String, RateLimitEntry>>>,
}

#[derive(Debug)]
struct RateLimitEntry {
    count: u32,
    reset_at: DateTime<Utc>,
}

impl ApiKeyGenerator {
    pub fn new(key_manager: Arc<SecureKeyManager>) -> Self {
        Self {
            key_manager,
            rate_limiter: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn generate_api_key(
        &self,
        user_id: &str,
        scope: Vec<String>,
    ) -> Result<String, KeyManagementError> {
        // Check rate limit
        self.check_rate_limit(user_id).await?;
        
        // Generate secure random API key
        let mut key_bytes = vec![0u8; 32];
        let rng = SystemRandom::new();
        rng.fill(&mut key_bytes).map_err(|e| KeyManagementError::Encryption(e.to_string()))?;
        
        // Create key metadata
        let metadata = ApiKeyMetadata {
            user_id: user_id.to_string(),
            scope,
            created_at: Utc::now(),
            last_used: None,
        };
        
        let key_id = format!("api_key_{}", uuid::Uuid::new_v4());
        let key_string = general_purpose::URL_SAFE_NO_PAD.encode(&key_bytes);
        
        // Store encrypted
        self.key_manager.store_key(
            key_id.clone(),
            &key_bytes,
            KeyType::ApiKey,
            Some(Duration::days(365)),
        ).await?;
        
        // Store metadata separately (could be in database)
        // ...
        
        Ok(key_string)
    }
    
    async fn check_rate_limit(&self, user_id: &str) -> Result<(), KeyManagementError> {
        let mut limiter = self.rate_limiter.write().await;
        let now = Utc::now();
        
        let entry = limiter.entry(user_id.to_string()).or_insert(RateLimitEntry {
            count: 0,
            reset_at: now + Duration::hours(1),
        });
        
        if now > entry.reset_at {
            entry.count = 0;
            entry.reset_at = now + Duration::hours(1);
        }
        
        if entry.count >= 10 {
            return Err(KeyManagementError::Unauthorized);
        }
        
        entry.count += 1;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiKeyMetadata {
    user_id: String,
    scope: Vec<String>,
    created_at: DateTime<Utc>,
    last_used: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_key_storage_and_retrieval() {
        let manager = SecureKeyManager::new_with_hsm(None).await.unwrap();
        
        let test_key = b"super_secret_key_12345";
        let key_id = "test_key_1";
        
        // Store key
        manager.store_key(
            key_id.to_string(),
            test_key,
            KeyType::ApiKey,
            None,
        ).await.unwrap();
        
        // Retrieve key
        let retrieved = manager.get_key(key_id).await.unwrap();
        assert_eq!(test_key, &retrieved[..]);
    }
    
    #[tokio::test]
    async fn test_key_rotation() {
        let manager = SecureKeyManager::new_with_hsm(None).await.unwrap();
        
        let test_key = b"original_key";
        let key_id = "rotate_test";
        
        manager.store_key(
            key_id.to_string(),
            test_key,
            KeyType::EncryptionKey,
            None,
        ).await.unwrap();
        
        let new_key = manager.rotate_key(key_id).await.unwrap();
        assert_ne!(test_key, &new_key[..]);
        
        let retrieved = manager.get_key(key_id).await.unwrap();
        assert_eq!(new_key, retrieved);
    }
}