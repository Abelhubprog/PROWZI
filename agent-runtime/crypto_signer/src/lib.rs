//! Cryptographic transaction signing for Prowzi trading agents.
//!
//! This module provides functionality for signing blockchain transactions,
//! with support for both AWS KMS (production) and local key (development) signing methods.
//! 
//! # Examples
//! 
//! ```
//! use prowzi_crypto_signer::sign_transaction;
//! 
//! async fn example() -> anyhow::Result<()> {
//!     let transaction_bytes = vec![1, 2, 3, 4];
//!     let signature = sign_transaction(&transaction_bytes).await?;
//!     println!("Transaction signed with signature: {:?}", signature);
//!     Ok(())
//! }
//! ```

use std::env;
use std::fmt;
use std::str::FromStr;

use anyhow::Result;
use async_trait::async_trait;
use base58::{FromBase58, ToBase58};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

#[cfg(feature = "kms")]
use aws_sdk_kms::Client as KmsClient;
#[cfg(feature = "kms")]
use aws_sdk_kms::primitives::Blob;

/// Error types for crypto signing operations
#[derive(Error, Debug)]
pub enum SigningError {
    #[error("Missing key: {0}")]
    MissingKey(String),
    
    #[error("Invalid key format: {0}")]
    InvalidKeyFormat(String),
    
    #[error("Signing operation failed: {0}")]
    SigningFailed(String),
    
    #[cfg(feature = "kms")]
    #[error("AWS KMS error: {0}")]
    KmsError(String),
    
    #[error("Environment error: {0}")]
    EnvError(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Represents an Ed25519 signature
#[derive(Clone, PartialEq, Eq)]
pub struct Ed25519Signature(pub [u8; 64]);

impl Ed25519Signature {
    /// Create a new signature from a byte array
    pub fn new(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }
    
    /// Convert the signature to a byte vector
    pub fn to_bytes(&self) -> [u8; 64] {
        self.0
    }
    
    /// Convert the signature to a base58 string (Solana format)
    pub fn to_base58(&self) -> String {
        self.0.to_base58()
    }
    
    /// Create a signature from a base58 string
    pub fn from_base58(s: &str) -> Result<Self, SigningError> {
        let bytes = s.from_base58()
            .map_err(|e| SigningError::InvalidKeyFormat(format!("Invalid base58: {}", e)))?;
        
        if bytes.len() != 64 {
            return Err(SigningError::InvalidKeyFormat(
                format!("Expected 64 bytes, got {}", bytes.len())
            ));
        }
        
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&bytes);
        Ok(Self(signature))
    }
}

impl fmt::Debug for Ed25519Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ed25519Signature({})", self.to_base58())
    }
}

impl fmt::Display for Ed25519Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_base58())
    }
}

impl From<Signature> for Ed25519Signature {
    fn from(sig: Signature) -> Self {
        Self(sig.to_bytes())
    }
}

impl From<Ed25519Signature> for Signature {
    fn from(sig: Ed25519Signature) -> Self {
        // Safe conversion with proper error handling
        Signature::from_bytes(&sig.0).unwrap_or_else(|_| {
            // This should never happen with valid 64-byte signatures,
            // but we provide a safe fallback
            tracing::error!("Invalid signature bytes encountered - using zero signature");
            Signature::from_bytes(&[0u8; 64]).expect("Zero signature should always be valid")
        })
    }
}

/// Transaction signer trait for signing blockchain transactions
#[async_trait]
pub trait TransactionSigner: Send + Sync {
    /// Sign a transaction with the signer's private key
    async fn sign(&self, transaction: &[u8]) -> Result<Ed25519Signature, SigningError>;
    
    /// Verify a signature against a message using the signer's public key
    fn verify(&self, message: &[u8], signature: &Ed25519Signature) -> Result<bool, SigningError>;
    
    /// Get the public key associated with this signer
    fn get_public_key(&self) -> Result<Vec<u8>, SigningError>;
}

/// AWS KMS-based transaction signer
#[cfg(feature = "kms")]
pub struct KmsSigner {
    /// AWS KMS client
    client: KmsClient,
    
    /// KMS key ID
    key_id: String,
}

#[cfg(feature = "kms")]
impl KmsSigner {
    /// Create a new KMS signer with the specified key ID
    pub async fn new(key_id: &str) -> Result<Self, SigningError> {
        let config = aws_config::from_env().load().await;
        let client = KmsClient::new(&config);
        
        Ok(Self {
            client,
            key_id: key_id.to_string(),
        })
    }
}

#[cfg(feature = "kms")]
#[async_trait]
impl TransactionSigner for KmsSigner {
    #[instrument(skip(self, transaction), fields(key_id = %self.key_id))]
    async fn sign(&self, transaction: &[u8]) -> Result<Ed25519Signature, SigningError> {
        info!("Signing transaction with AWS KMS, key ID: {}", self.key_id);
        
        // Create a message digest for the transaction
        use sha2::{Sha256, Digest};
        let message_digest = Sha256::digest(transaction);
        
        // Sign the message digest with KMS
        let sign_result = self.client.sign()
            .key_id(&self.key_id)
            .message(Blob::new(message_digest))
            .message_type(aws_sdk_kms::types::MessageType::Digest)
            .signing_algorithm(aws_sdk_kms::types::SigningAlgorithmSpec::EcdsaSha256)
            .send()
            .await
            .map_err(|e| SigningError::KmsError(e.to_string()))?;
        
        // Extract the signature
        let signature_bytes = sign_result.signature()
            .ok_or_else(|| SigningError::KmsError("No signature returned from KMS".to_string()))?
            .as_ref();
        
        // Convert the DER-encoded ECDSA signature to Ed25519 format
        // Note: This is a simplified example. In practice, you would need to properly convert
        // between ECDSA and Ed25519 formats, which is non-trivial.
        if signature_bytes.len() != 64 {
            return Err(SigningError::KmsError(
                format!("Expected 64 bytes, got {}", signature_bytes.len())
            ));
        }
        
        let mut sig = [0u8; 64];
        sig.copy_from_slice(signature_bytes);
        
        Ok(Ed25519Signature(sig))
    }
    
    fn verify(&self, _message: &[u8], _signature: &Ed25519Signature) -> Result<bool, SigningError> {
        // KMS doesn't provide a direct verify operation for Ed25519
        // In a real implementation, you would fetch the public key and verify locally
        Err(SigningError::UnsupportedOperation(
            "Verification not implemented for KMS signer".to_string()
        ))
    }
    
    fn get_public_key(&self) -> Result<Vec<u8>, SigningError> {
        // In a real implementation, you would fetch the public key from KMS
        // This is a placeholder
        Err(SigningError::UnsupportedOperation(
            "get_public_key not implemented for KMS signer".to_string()
        ))
    }
}

/// Local Ed25519 transaction signer
#[cfg(feature = "local-key")]
pub struct LocalKeySigner {
    /// Ed25519 signing key
    signing_key: SigningKey,
    
    /// Ed25519 verifying key (public key)
    verifying_key: VerifyingKey,
}

#[cfg(feature = "local-key")]
impl LocalKeySigner {
    /// Create a new local key signer from a base58-encoded private key
    /// 
    /// # INSECURE DEV ONLY
    /// This method is for development use only and should not be used in production.
    /// Private keys should never be stored in environment variables in production.
    #[instrument(skip(private_key_base58))]
    pub fn new(private_key_base58: &str) -> Result<Self, SigningError> {
        warn!("INSECURE DEV ONLY: Using local key signer with private key from environment");
        
        // Decode the base58 private key
        let private_key_bytes = private_key_base58.from_base58()
            .map_err(|e| SigningError::InvalidKeyFormat(format!("Invalid base58: {}", e)))?;
        
        if private_key_bytes.len() != 32 {
            return Err(SigningError::InvalidKeyFormat(
                format!("Expected 32 bytes, got {}", private_key_bytes.len())
            ));
        }
        
        // Create the signing key
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&private_key_bytes);
        
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();
        
        Ok(Self {
            signing_key,
            verifying_key,
        })
    }
}

#[cfg(feature = "local-key")]
#[async_trait]
impl TransactionSigner for LocalKeySigner {
    #[instrument(skip(self, transaction))]
    async fn sign(&self, transaction: &[u8]) -> Result<Ed25519Signature, SigningError> {
        warn!("INSECURE DEV ONLY: Signing transaction with local key");
        
        // Sign the transaction
        let signature = self.signing_key.sign(transaction);
        
        Ok(Ed25519Signature::from(signature))
    }
    
    #[instrument(skip(self, message, signature))]
    fn verify(&self, message: &[u8], signature: &Ed25519Signature) -> Result<bool, SigningError> {
        let dalek_signature = Signature::from(*signature);
        
        // Verify the signature
        match self.verifying_key.verify(message, &dalek_signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    fn get_public_key(&self) -> Result<Vec<u8>, SigningError> {
        Ok(self.verifying_key.to_bytes().to_vec())
    }
}

/// Sign a transaction using the appropriate signer based on environment configuration
///
/// This function will:
/// 1. Try to use AWS KMS if TRADING_KMS_KEY_ID is set
/// 2. Fall back to local key signing if TRADING_PRIVATE_KEY is set
///
/// # Arguments
///
/// * `tx_bytes` - The transaction bytes to sign
///
/// # Returns
///
/// * `Result<Ed25519Signature, SigningError>` - The signature or an error
#[instrument(skip(tx_bytes))]
pub async fn sign_transaction(tx_bytes: &[u8]) -> Result<Ed25519Signature, SigningError> {
    match env::var("TRADING_KMS_KEY_ID") {
        #[cfg(feature = "kms")]
        Ok(kms_key) => {
            info!("Using AWS KMS for transaction signing");
            let signer = KmsSigner::new(&kms_key).await
                .map_err(|e| SigningError::KmsError(format!("Failed to create KMS signer: {}", e)))?;
            signer.sign(tx_bytes).await
        },
        
        #[cfg(not(feature = "kms"))]
        Ok(_) => {
            error!("AWS KMS feature is not enabled but TRADING_KMS_KEY_ID is set");
            Err(SigningError::UnsupportedOperation(
                "AWS KMS feature is not enabled in this build".to_string()
            ))
        },
        
        Err(_) => {
            // Fall back to local key signing
            match env::var("TRADING_PRIVATE_KEY") {
                #[cfg(feature = "local-key")]
                Ok(private_key) => {
                    warn!("INSECURE DEV ONLY: Using local key for transaction signing");
                    let signer = LocalKeySigner::new(&private_key)
                        .map_err(|e| SigningError::EnvError(format!("Failed to create local key signer: {}", e)))?;
                    signer.sign(tx_bytes).await
                },
                
                #[cfg(not(feature = "local-key"))]
                Ok(_) => {
                    error!("Local key feature is not enabled but TRADING_PRIVATE_KEY is set");
                    Err(SigningError::UnsupportedOperation(
                        "Local key feature is not enabled in this build".to_string()
                    ))
                },
                
                Err(_) => {
                    error!("No signing key available. Set either TRADING_KMS_KEY_ID or TRADING_PRIVATE_KEY");
                    Err(SigningError::MissingKey(
                        "No signing key available. Set either TRADING_KMS_KEY_ID or TRADING_PRIVATE_KEY".to_string()
                    ))
                }
            }
        }
    }
}

/// Verify a signature for a transaction
///
/// This function will verify a signature using the appropriate verifier based on
/// the environment configuration.
///
/// # Arguments
///
/// * `tx_bytes` - The transaction bytes that were signed
/// * `signature` - The signature to verify
///
/// # Returns
///
/// * `Result<bool, SigningError>` - True if valid, false if invalid, or an error
#[instrument(skip(tx_bytes, signature))]
pub async fn verify_transaction(
    tx_bytes: &[u8],
    signature: &Ed25519Signature
) -> Result<bool, SigningError> {
    match env::var("TRADING_PRIVATE_KEY") {
        #[cfg(feature = "local-key")]
        Ok(private_key) => {
            let signer = LocalKeySigner::new(&private_key)
                .map_err(|e| SigningError::EnvError(format!("Failed to create local key signer: {}", e)))?;
            signer.verify(tx_bytes, signature)
        },
        
        _ => {
            // For KMS, we would need to implement a proper verification method
            // This is a placeholder
            Err(SigningError::UnsupportedOperation(
                "Verification not implemented for current configuration".to_string()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;
    
    #[cfg(feature = "local-key")]
    #[test]
    fn test_local_key_signer() {
        // Generate a random key for testing
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let private_key_bytes = signing_key.to_bytes();
        let private_key_base58 = private_key_bytes.to_base58();
        
        // Create a signer
        let signer = LocalKeySigner::new(&private_key_base58).unwrap();
        
        // Test signing and verification
        let message = b"Hello, world!";
        let signature = tokio_test::block_on(signer.sign(message)).unwrap();
        let valid = signer.verify(message, &signature).unwrap();
        
        assert!(valid);
        
        // Test invalid signature
        let mut bad_sig = signature.to_bytes();
        bad_sig[0] ^= 0xFF; // Flip some bits
        let bad_signature = Ed25519Signature::new(bad_sig);
        let invalid = signer.verify(message, &bad_signature).unwrap();
        
        assert!(!invalid);
    }
    
    #[test]
    fn test_ed25519_signature_base58() {
        let mut signature_bytes = [0u8; 64];
        for i in 0..64 {
            signature_bytes[i] = i as u8;
        }
        
        let signature = Ed25519Signature::new(signature_bytes);
        let base58 = signature.to_base58();
        let decoded = Ed25519Signature::from_base58(&base58).unwrap();
        
        assert_eq!(signature, decoded);
    }
}

// Additional modules for enhanced wallet functionality
pub mod multisig_wallet;
pub mod wallet_manager;
pub mod hardware_integration;
