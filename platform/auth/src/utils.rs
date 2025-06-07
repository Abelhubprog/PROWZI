use crate::{errors::AuthError, models::WalletAuthRequest};
use ethers::prelude::*;
use solana_sdk::{pubkey::Pubkey, signature::Signature};
use std::str::FromStr;

/// Verify Ethereum wallet signature using EIP-191 or EIP-4361 (SIWE)
pub async fn verify_ethereum_signature(request: &WalletAuthRequest) -> Result<String, AuthError> {
    // Parse the signature
    let signature = request.signature.parse::<ethers::types::Signature>()
        .map_err(|_| AuthError::InvalidSignature)?;

    // Hash the message (Ethereum uses keccak256 with prefix)
    let message_hash = ethers::utils::hash_message(&request.message);

    // Recover the address from signature
    let recovered_address = signature.recover(message_hash)
        .map_err(|_| AuthError::InvalidSignature)?;

    // Parse expected address
    let expected_address = request.address.parse::<Address>()
        .map_err(|_| AuthError::InvalidAddress)?;

    // Verify addresses match
    if recovered_address != expected_address {
        return Err(AuthError::SignatureMismatch);
    }

    // If message looks like SIWE, verify its format
    if request.message.contains("Sign in with Ethereum") || request.message.contains("prowzi.io") {
        verify_siwe_message(&request.message)?;
    }

    Ok(format!("eth:{}", request.address.to_lowercase()))
}

/// Verify Solana wallet signature
pub async fn verify_solana_signature(request: &WalletAuthRequest) -> Result<String, AuthError> {
    // Parse public key
    let pubkey = Pubkey::from_str(&request.address)
        .map_err(|_| AuthError::InvalidAddress)?;

    // Parse signature  
    let signature = Signature::from_str(&request.signature)
        .map_err(|_| AuthError::InvalidSignature)?;

    // Verify signature
    let message_bytes = request.message.as_bytes();
    if !signature.verify(pubkey.as_ref(), message_bytes) {
        return Err(AuthError::SignatureMismatch);
    }

    // Verify message contains expected content
    if !request.message.contains("Sign in to Prowzi") {
        return Err(AuthError::InvalidSiweMessage);
    }

    Ok(format!("sol:{}", request.address))
}

/// Verify SIWE (Sign-In with Ethereum) message format
pub fn verify_siwe_message(message: &str) -> Result<(), AuthError> {
    // Basic SIWE validation - in production, use the SIWE library
    let required_fields = [
        "prowzi.io",
        "wants you to sign in",
        "URI:",
        "Version:",
        "Chain ID:",
        "Nonce:",
        "Issued At:",
    ];

    for field in &required_fields {
        if !message.contains(field) {
            return Err(AuthError::InvalidSiweMessage);
        }
    }

    // Verify domain
    if !message.contains("prowzi.io") {
        return Err(AuthError::InvalidSiweMessage);
    }

    Ok(())
}

/// Generate a cryptographically secure nonce for SIWE
pub fn generate_nonce() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let nonce: [u8; 16] = rng.gen();
    hex::encode(nonce)
}

/// Validate tenant ID format
pub fn validate_tenant_id(tenant_id: &str) -> Result<(), AuthError> {
    if tenant_id.is_empty() || tenant_id.len() > 100 {
        return Err(AuthError::ConfigError("Invalid tenant ID length".to_string()));
    }

    // Only allow alphanumeric, hyphens, and underscores
    if !tenant_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Err(AuthError::ConfigError("Invalid tenant ID format".to_string()));
    }

    Ok(())
}
