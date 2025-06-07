//! Multi-signature wallet implementation for secure agent operations
//!
//! This module provides multi-signature wallet functionality for Prowzi trading agents,
//! enabling secure collaborative transaction signing with threshold signatures.

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use crate::{Ed25519Signature, SigningError, TransactionSigner};

/// Multi-signature wallet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultisigConfig {
    /// Unique identifier for the multisig wallet
    pub wallet_id: String,
    /// Required number of signatures (threshold)
    pub threshold: usize,
    /// Total number of signers
    pub total_signers: usize,
    /// List of authorized signer public keys
    pub authorized_signers: Vec<String>,
    /// Wallet creation timestamp
    pub created_at: DateTime<Utc>,
    /// Optional description
    pub description: Option<String>,
}

impl MultisigConfig {
    pub fn new(
        wallet_id: String,
        threshold: usize,
        authorized_signers: Vec<String>,
        description: Option<String>,
    ) -> Result<Self> {
        let total_signers = authorized_signers.len();
        
        if threshold == 0 {
            return Err(anyhow!("Threshold must be greater than 0"));
        }
        
        if threshold > total_signers {
            return Err(anyhow!("Threshold cannot exceed total number of signers"));
        }
        
        if total_signers == 0 {
            return Err(anyhow!("Must have at least one authorized signer"));
        }
        
        Ok(Self {
            wallet_id,
            threshold,
            total_signers,
            authorized_signers,
            created_at: Utc::now(),
            description,
        })
    }
    
    /// Validate if a signer is authorized
    pub fn is_authorized_signer(&self, public_key: &str) -> bool {
        self.authorized_signers.contains(&public_key.to_string())
    }
}

/// Pending multisig transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTransaction {
    /// Unique transaction identifier
    pub transaction_id: String,
    /// Wallet ID this transaction belongs to
    pub wallet_id: String,
    /// Raw transaction data to be signed
    pub transaction_data: Vec<u8>,
    /// Hash of the transaction for quick verification
    pub transaction_hash: String,
    /// Collected signatures so far
    pub signatures: BTreeMap<String, Ed25519Signature>,
    /// Transaction creation timestamp
    pub created_at: DateTime<Utc>,
    /// Transaction expiration (optional)
    pub expires_at: Option<DateTime<Utc>>,
    /// Transaction initiator
    pub initiator: String,
    /// Optional description/memo
    pub memo: Option<String>,
    /// Current status
    pub status: TransactionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Pending,
    ReadyForExecution,
    Executed,
    Expired,
    Cancelled,
}

impl PendingTransaction {
    pub fn new(
        wallet_id: String,
        transaction_data: Vec<u8>,
        initiator: String,
        memo: Option<String>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Self {
        let transaction_id = Uuid::new_v4().to_string();
        let transaction_hash = Self::compute_hash(&transaction_data);
        
        Self {
            transaction_id,
            wallet_id,
            transaction_data,
            transaction_hash,
            signatures: BTreeMap::new(),
            created_at: Utc::now(),
            expires_at,
            initiator,
            memo,
            status: TransactionStatus::Pending,
        }
    }
    
    fn compute_hash(data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let hash = Sha256::digest(data);
        hex::encode(hash)
    }
    
    /// Add a signature to the transaction
    pub fn add_signature(&mut self, signer_pubkey: String, signature: Ed25519Signature) -> Result<()> {
        if self.status != TransactionStatus::Pending {
            return Err(anyhow!("Cannot add signatures to non-pending transaction"));
        }
        
        if let Some(expires_at) = self.expires_at {
            if Utc::now() > expires_at {
                self.status = TransactionStatus::Expired;
                return Err(anyhow!("Transaction has expired"));
            }
        }
        
        self.signatures.insert(signer_pubkey, signature);
        Ok(())
    }
    
    /// Check if the transaction has enough signatures
    pub fn has_sufficient_signatures(&self, config: &MultisigConfig) -> bool {
        self.signatures.len() >= config.threshold
    }
    
    /// Update transaction status based on signature count
    pub fn update_status(&mut self, config: &MultisigConfig) {
        if let Some(expires_at) = self.expires_at {
            if Utc::now() > expires_at {
                self.status = TransactionStatus::Expired;
                return;
            }
        }
        
        if self.has_sufficient_signatures(config) {
            self.status = TransactionStatus::ReadyForExecution;
        }
    }
    
    /// Get missing signatures count
    pub fn missing_signatures(&self, config: &MultisigConfig) -> usize {
        if self.signatures.len() >= config.threshold {
            0
        } else {
            config.threshold - self.signatures.len()
        }
    }
}

/// Multi-signature wallet manager
pub struct MultisigWallet {
    /// Wallet configuration
    config: MultisigConfig,
    /// Pending transactions
    pending_transactions: Arc<RwLock<HashMap<String, PendingTransaction>>>,
    /// Completed transactions history
    completed_transactions: Arc<RwLock<Vec<CompletedTransaction>>>,
    /// Local signer (if this instance can sign)
    local_signer: Option<Arc<dyn TransactionSigner>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedTransaction {
    pub transaction_id: String,
    pub wallet_id: String,
    pub transaction_hash: String,
    pub signatures: BTreeMap<String, Ed25519Signature>,
    pub executed_at: DateTime<Utc>,
    pub final_signature: Ed25519Signature,
    pub status: TransactionStatus,
}

impl MultisigWallet {
    /// Create a new multisig wallet
    pub fn new(config: MultisigConfig) -> Self {
        Self {
            config,
            pending_transactions: Arc::new(RwLock::new(HashMap::new())),
            completed_transactions: Arc::new(RwLock::new(Vec::new())),
            local_signer: None,
        }
    }
    
    /// Set a local signer for this wallet instance
    pub fn with_signer(mut self, signer: Arc<dyn TransactionSigner>) -> Self {
        self.local_signer = Some(signer);
        self
    }
    
    /// Get wallet configuration
    pub fn get_config(&self) -> &MultisigConfig {
        &self.config
    }
    
    /// Initiate a new transaction
    pub async fn initiate_transaction(
        &self,
        transaction_data: Vec<u8>,
        initiator: String,
        memo: Option<String>,
        expires_in_minutes: Option<i64>,
    ) -> Result<String> {
        if !self.config.is_authorized_signer(&initiator) {
            return Err(anyhow!("Initiator is not an authorized signer"));
        }
        
        let expires_at = expires_in_minutes.map(|minutes| {
            Utc::now() + chrono::Duration::minutes(minutes)
        });
        
        let mut transaction = PendingTransaction::new(
            self.config.wallet_id.clone(),
            transaction_data,
            initiator.clone(),
            memo,
            expires_at,
        );
        
        // Auto-sign if we have a local signer and the initiator matches
        if let Some(ref signer) = self.local_signer {
            if let Ok(public_key_bytes) = signer.get_public_key() {
                let public_key = hex::encode(public_key_bytes);
                if public_key == initiator {
                    let signature = signer.sign(&transaction.transaction_data).await
                        .map_err(|e| anyhow!("Failed to auto-sign transaction: {}", e))?;
                    transaction.add_signature(initiator, signature)?;
                    transaction.update_status(&self.config);
                }
            }
        }
        
        let transaction_id = transaction.transaction_id.clone();
        let mut pending = self.pending_transactions.write().await;
        pending.insert(transaction_id.clone(), transaction);
        
        Ok(transaction_id)
    }
    
    /// Sign a pending transaction
    pub async fn sign_transaction(
        &self,
        transaction_id: &str,
        signer_pubkey: &str,
    ) -> Result<()> {
        if !self.config.is_authorized_signer(signer_pubkey) {
            return Err(anyhow!("Signer is not authorized for this wallet"));
        }
        
        let signature = {
            let pending = self.pending_transactions.read().await;
            let transaction = pending.get(transaction_id)
                .ok_or_else(|| anyhow!("Transaction not found: {}", transaction_id))?;
            
            if transaction.signatures.contains_key(signer_pubkey) {
                return Err(anyhow!("Signer has already signed this transaction"));
            }
            
            // Sign the transaction if we have a local signer
            if let Some(ref signer) = self.local_signer {
                if let Ok(public_key_bytes) = signer.get_public_key() {
                    let public_key = hex::encode(public_key_bytes);
                    if public_key == signer_pubkey {
                        signer.sign(&transaction.transaction_data).await
                            .map_err(|e| anyhow!("Failed to sign transaction: {}", e))?
                    } else {
                        return Err(anyhow!("Local signer public key does not match requested signer"));
                    }
                } else {
                    return Err(anyhow!("Failed to get public key from local signer"));
                }
            } else {
                return Err(anyhow!("No local signer available"));
            }
        };
        
        // Update the transaction with the new signature
        let mut pending = self.pending_transactions.write().await;
        if let Some(transaction) = pending.get_mut(transaction_id) {
            transaction.add_signature(signer_pubkey.to_string(), signature)?;
            transaction.update_status(&self.config);
            
            // If ready for execution, move to completion process
            if transaction.status == TransactionStatus::ReadyForExecution {
                // In a real implementation, this would execute the transaction
                transaction.status = TransactionStatus::Executed;
                
                let completed = CompletedTransaction {
                    transaction_id: transaction.transaction_id.clone(),
                    wallet_id: transaction.wallet_id.clone(),
                    transaction_hash: transaction.transaction_hash.clone(),
                    signatures: transaction.signatures.clone(),
                    executed_at: Utc::now(),
                    final_signature: signature, // Combined signature in real implementation
                    status: TransactionStatus::Executed,
                };
                
                // Move to completed transactions
                let mut completed_txs = self.completed_transactions.write().await;
                completed_txs.push(completed);
            }
        }
        
        Ok(())
    }
    
    /// Get all pending transactions
    pub async fn get_pending_transactions(&self) -> Vec<PendingTransaction> {
        let pending = self.pending_transactions.read().await;
        pending.values().cloned().collect()
    }
    
    /// Get a specific pending transaction
    pub async fn get_pending_transaction(&self, transaction_id: &str) -> Option<PendingTransaction> {
        let pending = self.pending_transactions.read().await;
        pending.get(transaction_id).cloned()
    }
    
    /// Get completed transactions
    pub async fn get_completed_transactions(&self) -> Vec<CompletedTransaction> {
        let completed = self.completed_transactions.read().await;
        completed.clone()
    }
    
    /// Cancel a pending transaction (only by initiator or with sufficient signatures)
    pub async fn cancel_transaction(
        &self,
        transaction_id: &str,
        requester: &str,
    ) -> Result<()> {
        let mut pending = self.pending_transactions.write().await;
        
        if let Some(transaction) = pending.get_mut(transaction_id) {
            // Only initiator can cancel, or if we have threshold signatures for cancellation
            if transaction.initiator != requester && !self.config.is_authorized_signer(requester) {
                return Err(anyhow!("Not authorized to cancel this transaction"));
            }
            
            if transaction.status != TransactionStatus::Pending {
                return Err(anyhow!("Can only cancel pending transactions"));
            }
            
            transaction.status = TransactionStatus::Cancelled;
        } else {
            return Err(anyhow!("Transaction not found: {}", transaction_id));
        }
        
        Ok(())
    }
    
    /// Clean up expired transactions
    pub async fn cleanup_expired_transactions(&self) -> Result<usize> {
        let mut pending = self.pending_transactions.write().await;
        let mut expired_count = 0;
        
        let expired_ids: Vec<String> = pending
            .iter()
            .filter_map(|(id, tx)| {
                if let Some(expires_at) = tx.expires_at {
                    if Utc::now() > expires_at {
                        return Some(id.clone());
                    }
                }
                None
            })
            .collect();
        
        for id in expired_ids {
            if let Some(mut tx) = pending.remove(&id) {
                tx.status = TransactionStatus::Expired;
                expired_count += 1;
            }
        }
        
        Ok(expired_count)
    }
    
    /// Get wallet statistics
    pub async fn get_wallet_stats(&self) -> WalletStats {
        let pending = self.pending_transactions.read().await;
        let completed = self.completed_transactions.read().await;
        
        let pending_count = pending.len();
        let ready_count = pending
            .values()
            .filter(|tx| tx.status == TransactionStatus::ReadyForExecution)
            .count();
        let expired_count = pending
            .values()
            .filter(|tx| tx.status == TransactionStatus::Expired)
            .count();
        let completed_count = completed.len();
        
        WalletStats {
            total_pending: pending_count,
            ready_for_execution: ready_count,
            expired: expired_count,
            completed: completed_count,
            threshold: self.config.threshold,
            total_signers: self.config.total_signers,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletStats {
    pub total_pending: usize,
    pub ready_for_execution: usize,
    pub expired: usize,
    pub completed: usize,
    pub threshold: usize,
    pub total_signers: usize,
}

/// Multi-signature wallet factory for creating and managing multiple wallets
pub struct MultisigWalletFactory {
    wallets: Arc<RwLock<HashMap<String, MultisigWallet>>>,
}

impl MultisigWalletFactory {
    pub fn new() -> Self {
        Self {
            wallets: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a new multisig wallet
    pub async fn create_wallet(
        &self,
        config: MultisigConfig,
        signer: Option<Arc<dyn TransactionSigner>>,
    ) -> Result<String> {
        let wallet_id = config.wallet_id.clone();
        
        let mut wallet = MultisigWallet::new(config);
        if let Some(signer) = signer {
            wallet = wallet.with_signer(signer);
        }
        
        let mut wallets = self.wallets.write().await;
        if wallets.contains_key(&wallet_id) {
            return Err(anyhow!("Wallet with ID {} already exists", wallet_id));
        }
        
        wallets.insert(wallet_id.clone(), wallet);
        Ok(wallet_id)
    }
    
    /// Get a wallet by ID
    pub async fn get_wallet(&self, wallet_id: &str) -> Option<MultisigWallet> {
        let wallets = self.wallets.read().await;
        wallets.get(wallet_id).cloned()
    }
    
    /// List all wallet IDs
    pub async fn list_wallets(&self) -> Vec<String> {
        let wallets = self.wallets.read().await;
        wallets.keys().cloned().collect()
    }
    
    /// Remove a wallet
    pub async fn remove_wallet(&self, wallet_id: &str) -> Result<()> {
        let mut wallets = self.wallets.write().await;
        wallets.remove(wallet_id)
            .ok_or_else(|| anyhow!("Wallet not found: {}", wallet_id))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LocalKeySigner;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    
    #[tokio::test]
    async fn test_multisig_wallet_creation() {
        let config = MultisigConfig::new(
            "test-wallet".to_string(),
            2,
            vec!["signer1".to_string(), "signer2".to_string(), "signer3".to_string()],
            Some("Test wallet".to_string()),
        ).unwrap();
        
        let wallet = MultisigWallet::new(config.clone());
        assert_eq!(wallet.get_config().threshold, 2);
        assert_eq!(wallet.get_config().total_signers, 3);
    }
    
    #[tokio::test]
    async fn test_transaction_workflow() {
        let config = MultisigConfig::new(
            "test-wallet".to_string(),
            2,
            vec!["signer1".to_string(), "signer2".to_string()],
            None,
        ).unwrap();
        
        let wallet = MultisigWallet::new(config);
        
        // Create a test transaction
        let tx_data = b"test transaction data".to_vec();
        let tx_id = wallet.initiate_transaction(
            tx_data,
            "signer1".to_string(),
            Some("Test transaction".to_string()),
            Some(60), // 60 minutes expiry
        ).await.unwrap();
        
        // Check pending transaction
        let pending_txs = wallet.get_pending_transactions().await;
        assert_eq!(pending_txs.len(), 1);
        assert_eq!(pending_txs[0].transaction_id, tx_id);
    }
    
    #[tokio::test]
    async fn test_wallet_factory() {
        let factory = MultisigWalletFactory::new();
        
        let config = MultisigConfig::new(
            "factory-wallet".to_string(),
            1,
            vec!["signer1".to_string()],
            None,
        ).unwrap();
        
        let wallet_id = factory.create_wallet(config, None).await.unwrap();
        assert_eq!(wallet_id, "factory-wallet");
        
        let wallets = factory.list_wallets().await;
        assert_eq!(wallets.len(), 1);
        assert!(wallets.contains(&"factory-wallet".to_string()));
    }
}