//! Comprehensive wallet management system for Prowzi agents
//!
//! This module provides a centralized wallet management system that handles
//! multiple wallet types, hot/cold storage separation, and secure operations.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};

use crate::{Ed25519Signature, SigningError, TransactionSigner};
use crate::multisig_wallet::{MultisigWallet, MultisigConfig, MultisigWalletFactory};

/// Wallet types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalletType {
    /// Single-signature hot wallet for frequent operations
    HotWallet,
    /// Single-signature cold wallet for secure storage
    ColdWallet,
    /// Multi-signature wallet for collaborative operations
    Multisig,
    /// Hardware wallet integration
    Hardware,
    /// Agent-controlled programmatic wallet
    Agent,
}

/// Wallet security level classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum SecurityLevel {
    /// Low security - for small amounts, development
    Low = 1,
    /// Medium security - for operational amounts
    Medium = 2,
    /// High security - for significant amounts
    High = 3,
    /// Critical security - for treasury and critical operations
    Critical = 4,
}

/// Wallet configuration and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    /// Unique wallet identifier
    pub wallet_id: String,
    /// Human-readable wallet name
    pub name: String,
    /// Wallet type
    pub wallet_type: WalletType,
    /// Security level
    pub security_level: SecurityLevel,
    /// Public key (base58 encoded)
    pub public_key: String,
    /// Associated agent ID (if applicable)
    pub agent_id: Option<String>,
    /// Wallet creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
    /// Wallet status
    pub status: WalletStatus,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
    /// Spending limits (in base units)
    pub spending_limits: SpendingLimits,
    /// Risk controls
    pub risk_controls: RiskControls,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalletStatus {
    Active,
    Frozen,
    Compromised,
    Archived,
    PendingActivation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendingLimits {
    /// Maximum amount per transaction
    pub max_transaction: u64,
    /// Maximum amount per hour
    pub max_hourly: u64,
    /// Maximum amount per day
    pub max_daily: u64,
    /// Time window for rate limiting (minutes)
    pub rate_limit_window: u64,
}

impl Default for SpendingLimits {
    fn default() -> Self {
        Self {
            max_transaction: 1_000_000, // 1M base units
            max_hourly: 10_000_000,     // 10M base units
            max_daily: 100_000_000,     // 100M base units
            rate_limit_window: 60,      // 1 hour
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskControls {
    /// Require manual approval for amounts above this threshold
    pub manual_approval_threshold: u64,
    /// Require multi-party approval
    pub require_multi_party: bool,
    /// Maximum number of pending transactions
    pub max_pending_transactions: usize,
    /// Allowed destination addresses (if empty, all allowed)
    pub allowed_destinations: Vec<String>,
    /// Blocked destination addresses
    pub blocked_destinations: Vec<String>,
    /// Geographic restrictions
    pub geo_restrictions: Vec<String>,
}

impl Default for RiskControls {
    fn default() -> Self {
        Self {
            manual_approval_threshold: 10_000_000, // 10M base units
            require_multi_party: false,
            max_pending_transactions: 10,
            allowed_destinations: Vec::new(),
            blocked_destinations: Vec::new(),
            geo_restrictions: Vec::new(),
        }
    }
}

/// Transaction request for wallet operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionRequest {
    /// Unique request identifier
    pub request_id: String,
    /// Source wallet ID
    pub wallet_id: String,
    /// Transaction data to sign
    pub transaction_data: Vec<u8>,
    /// Requested by (agent ID or user)
    pub requested_by: String,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Optional memo
    pub memo: Option<String>,
    /// Priority level
    pub priority: TransactionPriority,
    /// Approval requirements
    pub approval_requirements: ApprovalRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionPriority {
    Low,
    Normal,
    High,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequirements {
    /// Requires manual approval
    pub manual_approval: bool,
    /// Required approvers
    pub required_approvers: Vec<String>,
    /// Approval timeout (minutes)
    pub timeout_minutes: Option<u64>,
}

/// Transaction approval record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionApproval {
    pub approver: String,
    pub approved_at: DateTime<Utc>,
    pub signature: Option<Ed25519Signature>,
    pub notes: Option<String>,
}

/// Pending transaction with approval tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingTransactionRequest {
    pub request: TransactionRequest,
    pub approvals: Vec<TransactionApproval>,
    pub status: PendingStatus,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PendingStatus {
    PendingApproval,
    Approved,
    Rejected,
    Expired,
    Executed,
}

/// Comprehensive wallet manager
pub struct WalletManager {
    /// Registered wallets
    wallets: Arc<RwLock<HashMap<String, WalletInfo>>>,
    /// Wallet signers
    signers: Arc<RwLock<HashMap<String, Arc<dyn TransactionSigner>>>>,
    /// Multisig wallet factory
    multisig_factory: MultisigWalletFactory,
    /// Pending transaction requests
    pending_requests: Arc<RwLock<HashMap<String, PendingTransactionRequest>>>,
    /// Transaction history
    transaction_history: Arc<RwLock<Vec<TransactionRecord>>>,
    /// Risk engine
    risk_engine: Arc<RiskEngine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionRecord {
    pub transaction_id: String,
    pub wallet_id: String,
    pub request_id: String,
    pub transaction_hash: String,
    pub executed_at: DateTime<Utc>,
    pub executed_by: String,
    pub signature: Ed25519Signature,
    pub amount: Option<u64>,
    pub destination: Option<String>,
    pub status: TransactionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Submitted,
    Confirmed,
    Failed,
    Cancelled,
}

impl WalletManager {
    /// Create a new wallet manager
    pub fn new() -> Self {
        Self {
            wallets: Arc::new(RwLock::new(HashMap::new())),
            signers: Arc::new(RwLock::new(HashMap::new())),
            multisig_factory: MultisigWalletFactory::new(),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            transaction_history: Arc::new(RwLock::new(Vec::new())),
            risk_engine: Arc::new(RiskEngine::new()),
        }
    }
    
    /// Register a new wallet
    pub async fn register_wallet(
        &self,
        wallet_info: WalletInfo,
        signer: Option<Arc<dyn TransactionSigner>>,
    ) -> Result<()> {
        let wallet_id = wallet_info.wallet_id.clone();
        
        // Validate wallet info
        self.validate_wallet_info(&wallet_info)?;
        
        // Store wallet info
        {
            let mut wallets = self.wallets.write().await;
            if wallets.contains_key(&wallet_id) {
                return Err(anyhow!("Wallet {} already exists", wallet_id));
            }
            wallets.insert(wallet_id.clone(), wallet_info.clone());
        }
        
        // Store signer if provided
        if let Some(signer) = signer {
            let mut signers = self.signers.write().await;
            signers.insert(wallet_id.clone(), signer);
        }
        
        // Create multisig wallet if needed
        if wallet_info.wallet_type == WalletType::Multisig {
            // Extract multisig configuration from metadata
            if let Some(threshold_str) = wallet_info.metadata.get("threshold") {
                if let Some(signers_str) = wallet_info.metadata.get("authorized_signers") {
                    let threshold: usize = threshold_str.parse()
                        .map_err(|_| anyhow!("Invalid threshold in metadata"))?;
                    let authorized_signers: Vec<String> = serde_json::from_str(signers_str)
                        .map_err(|_| anyhow!("Invalid authorized_signers in metadata"))?;
                    
                    let multisig_config = MultisigConfig::new(
                        wallet_id.clone(),
                        threshold,
                        authorized_signers,
                        Some(wallet_info.name.clone()),
                    )?;
                    
                    self.multisig_factory.create_wallet(multisig_config, None).await?;
                }
            }
        }
        
        tracing::info!("Registered wallet: {} ({})", wallet_id, wallet_info.wallet_type.name());
        Ok(())
    }
    
    fn validate_wallet_info(&self, wallet_info: &WalletInfo) -> Result<()> {
        if wallet_info.wallet_id.is_empty() {
            return Err(anyhow!("Wallet ID cannot be empty"));
        }
        
        if wallet_info.name.is_empty() {
            return Err(anyhow!("Wallet name cannot be empty"));
        }
        
        if wallet_info.public_key.is_empty() {
            return Err(anyhow!("Public key cannot be empty"));
        }
        
        // Validate spending limits
        if wallet_info.spending_limits.max_transaction > wallet_info.spending_limits.max_daily {
            return Err(anyhow!("Max transaction cannot exceed max daily limit"));
        }
        
        Ok(())
    }
    
    /// Get wallet information
    pub async fn get_wallet(&self, wallet_id: &str) -> Option<WalletInfo> {
        let wallets = self.wallets.read().await;
        wallets.get(wallet_id).cloned()
    }
    
    /// List all wallets
    pub async fn list_wallets(&self) -> Vec<WalletInfo> {
        let wallets = self.wallets.read().await;
        wallets.values().cloned().collect()
    }
    
    /// List wallets by type
    pub async fn list_wallets_by_type(&self, wallet_type: WalletType) -> Vec<WalletInfo> {
        let wallets = self.wallets.read().await;
        wallets.values()
            .filter(|w| w.wallet_type == wallet_type)
            .cloned()
            .collect()
    }
    
    /// List wallets by agent
    pub async fn list_wallets_by_agent(&self, agent_id: &str) -> Vec<WalletInfo> {
        let wallets = self.wallets.read().await;
        wallets.values()
            .filter(|w| w.agent_id.as_ref() == Some(&agent_id.to_string()))
            .cloned()
            .collect()
    }
    
    /// Request a transaction signature
    pub async fn request_transaction(
        &self,
        wallet_id: &str,
        transaction_data: Vec<u8>,
        requested_by: &str,
        memo: Option<String>,
        priority: TransactionPriority,
    ) -> Result<String> {
        let wallet = self.get_wallet(wallet_id).await
            .ok_or_else(|| anyhow!("Wallet not found: {}", wallet_id))?;
        
        // Check wallet status
        if wallet.status != WalletStatus::Active {
            return Err(anyhow!("Wallet is not active: {:?}", wallet.status));
        }
        
        // Run risk assessment
        let risk_assessment = self.risk_engine.assess_transaction_risk(
            &wallet,
            &transaction_data,
            requested_by,
        ).await?;
        
        if risk_assessment.blocked {
            return Err(anyhow!("Transaction blocked by risk engine: {}", risk_assessment.reason));
        }
        
        // Create transaction request
        let request_id = Uuid::new_v4().to_string();
        let approval_requirements = ApprovalRequirements {
            manual_approval: risk_assessment.requires_manual_approval,
            required_approvers: risk_assessment.required_approvers,
            timeout_minutes: Some(60), // 1 hour default timeout
        };
        
        let request = TransactionRequest {
            request_id: request_id.clone(),
            wallet_id: wallet_id.to_string(),
            transaction_data,
            requested_by: requested_by.to_string(),
            requested_at: Utc::now(),
            memo,
            priority,
            approval_requirements,
        };
        
        let expires_at = Some(Utc::now() + chrono::Duration::hours(1));
        
        let pending_request = PendingTransactionRequest {
            request,
            approvals: Vec::new(),
            status: if risk_assessment.requires_manual_approval {
                PendingStatus::PendingApproval
            } else {
                PendingStatus::Approved
            },
            created_at: Utc::now(),
            expires_at,
        };
        
        // Store pending request
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(request_id.clone(), pending_request);
        }
        
        // Auto-execute if no approval required
        if !risk_assessment.requires_manual_approval {
            self.execute_transaction(&request_id).await?;
        }
        
        Ok(request_id)
    }
    
    /// Approve a pending transaction
    pub async fn approve_transaction(
        &self,
        request_id: &str,
        approver: &str,
        notes: Option<String>,
    ) -> Result<()> {
        let mut pending = self.pending_requests.write().await;
        
        if let Some(pending_request) = pending.get_mut(request_id) {
            if pending_request.status != PendingStatus::PendingApproval {
                return Err(anyhow!("Transaction is not pending approval"));
            }
            
            // Check if approver is authorized
            if !pending_request.request.approval_requirements.required_approvers.contains(&approver.to_string()) {
                return Err(anyhow!("Approver is not authorized"));
            }
            
            // Check if already approved by this approver
            if pending_request.approvals.iter().any(|a| a.approver == approver) {
                return Err(anyhow!("Already approved by this approver"));
            }
            
            // Add approval
            let approval = TransactionApproval {
                approver: approver.to_string(),
                approved_at: Utc::now(),
                signature: None, // Could be enhanced with approval signatures
                notes,
            };
            
            pending_request.approvals.push(approval);
            
            // Check if all required approvals are obtained
            let required_count = pending_request.request.approval_requirements.required_approvers.len();
            if pending_request.approvals.len() >= required_count {
                pending_request.status = PendingStatus::Approved;
                
                // Execute the transaction
                drop(pending);
                self.execute_transaction(request_id).await?;
            }
        } else {
            return Err(anyhow!("Transaction request not found"));
        }
        
        Ok(())
    }
    
    /// Execute an approved transaction
    pub async fn execute_transaction(&self, request_id: &str) -> Result<String> {
        let (request, wallet_id) = {
            let pending = self.pending_requests.read().await;
            let pending_request = pending.get(request_id)
                .ok_or_else(|| anyhow!("Transaction request not found"))?;
            
            if pending_request.status != PendingStatus::Approved {
                return Err(anyhow!("Transaction is not approved"));
            }
            
            (pending_request.request.clone(), pending_request.request.wallet_id.clone())
        };
        
        // Get the signer
        let signers = self.signers.read().await;
        let signer = signers.get(&wallet_id)
            .ok_or_else(|| anyhow!("No signer available for wallet: {}", wallet_id))?;
        
        // Sign the transaction
        let signature = signer.sign(&request.transaction_data).await
            .map_err(|e| anyhow!("Failed to sign transaction: {}", e))?;
        
        // Create transaction record
        let transaction_id = Uuid::new_v4().to_string();
        let transaction_hash = Self::compute_transaction_hash(&request.transaction_data);
        
        let record = TransactionRecord {
            transaction_id: transaction_id.clone(),
            wallet_id: wallet_id.clone(),
            request_id: request_id.to_string(),
            transaction_hash,
            executed_at: Utc::now(),
            executed_by: request.requested_by.clone(),
            signature,
            amount: None, // Could be extracted from transaction data
            destination: None, // Could be extracted from transaction data
            status: TransactionStatus::Submitted,
        };
        
        // Store transaction record
        {
            let mut history = self.transaction_history.write().await;
            history.push(record);
        }
        
        // Update pending request status
        {
            let mut pending = self.pending_requests.write().await;
            if let Some(pending_request) = pending.get_mut(request_id) {
                pending_request.status = PendingStatus::Executed;
            }
        }
        
        // Update wallet last used timestamp
        {
            let mut wallets = self.wallets.write().await;
            if let Some(wallet) = wallets.get_mut(&wallet_id) {
                wallet.last_used = Some(Utc::now());
            }
        }
        
        tracing::info!("Executed transaction: {} for wallet: {}", transaction_id, wallet_id);
        Ok(transaction_id)
    }
    
    fn compute_transaction_hash(data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let hash = Sha256::digest(data);
        hex::encode(hash)
    }
    
    /// Get pending transaction requests
    pub async fn get_pending_requests(&self) -> Vec<PendingTransactionRequest> {
        let pending = self.pending_requests.read().await;
        pending.values().cloned().collect()
    }
    
    /// Get pending requests for a specific wallet
    pub async fn get_pending_requests_for_wallet(&self, wallet_id: &str) -> Vec<PendingTransactionRequest> {
        let pending = self.pending_requests.read().await;
        pending.values()
            .filter(|r| r.request.wallet_id == wallet_id)
            .cloned()
            .collect()
    }
    
    /// Get transaction history
    pub async fn get_transaction_history(&self, wallet_id: Option<&str>) -> Vec<TransactionRecord> {
        let history = self.transaction_history.read().await;
        
        if let Some(wallet_id) = wallet_id {
            history.iter()
                .filter(|r| r.wallet_id == wallet_id)
                .cloned()
                .collect()
        } else {
            history.clone()
        }
    }
    
    /// Freeze a wallet
    pub async fn freeze_wallet(&self, wallet_id: &str, reason: &str) -> Result<()> {
        let mut wallets = self.wallets.write().await;
        
        if let Some(wallet) = wallets.get_mut(wallet_id) {
            wallet.status = WalletStatus::Frozen;
            wallet.metadata.insert("freeze_reason".to_string(), reason.to_string());
            wallet.metadata.insert("frozen_at".to_string(), Utc::now().to_rfc3339());
            
            tracing::warn!("Wallet {} frozen: {}", wallet_id, reason);
        } else {
            return Err(anyhow!("Wallet not found: {}", wallet_id));
        }
        
        Ok(())
    }
    
    /// Unfreeze a wallet
    pub async fn unfreeze_wallet(&self, wallet_id: &str) -> Result<()> {
        let mut wallets = self.wallets.write().await;
        
        if let Some(wallet) = wallets.get_mut(wallet_id) {
            wallet.status = WalletStatus::Active;
            wallet.metadata.remove("freeze_reason");
            wallet.metadata.remove("frozen_at");
            wallet.metadata.insert("unfrozen_at".to_string(), Utc::now().to_rfc3339());
            
            tracing::info!("Wallet {} unfrozen", wallet_id);
        } else {
            return Err(anyhow!("Wallet not found: {}", wallet_id));
        }
        
        Ok(())
    }
    
    /// Clean up expired pending requests
    pub async fn cleanup_expired_requests(&self) -> Result<usize> {
        let mut pending = self.pending_requests.write().await;
        let mut expired_count = 0;
        
        let expired_ids: Vec<String> = pending
            .iter()
            .filter_map(|(id, req)| {
                if let Some(expires_at) = req.expires_at {
                    if Utc::now() > expires_at && req.status == PendingStatus::PendingApproval {
                        return Some(id.clone());
                    }
                }
                None
            })
            .collect();
        
        for id in expired_ids {
            if let Some(mut req) = pending.get_mut(&id) {
                req.status = PendingStatus::Expired;
                expired_count += 1;
            }
        }
        
        Ok(expired_count)
    }
}

/// Risk assessment engine for wallet operations
pub struct RiskEngine {
    // Risk rules and configuration would be stored here
}

#[derive(Debug)]
pub struct RiskAssessment {
    pub blocked: bool,
    pub reason: String,
    pub requires_manual_approval: bool,
    pub required_approvers: Vec<String>,
    pub risk_score: f64,
}

impl RiskEngine {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn assess_transaction_risk(
        &self,
        wallet: &WalletInfo,
        transaction_data: &[u8],
        requested_by: &str,
    ) -> Result<RiskAssessment> {
        let mut risk_score = 0.0;
        let mut requires_manual_approval = false;
        let mut required_approvers = Vec::new();
        
        // Simple risk assessment logic
        let transaction_size = transaction_data.len();
        
        // Size-based risk
        if transaction_size > 10000 {
            risk_score += 0.3;
        }
        
        // Security level-based risk
        match wallet.security_level {
            SecurityLevel::Critical => {
                requires_manual_approval = true;
                required_approvers.push("risk_manager".to_string());
                risk_score += 0.5;
            },
            SecurityLevel::High => {
                if transaction_size > 5000 {
                    requires_manual_approval = true;
                    required_approvers.push("senior_agent".to_string());
                }
                risk_score += 0.3;
            },
            _ => {}
        }
        
        // Check risk controls
        if wallet.risk_controls.require_multi_party {
            requires_manual_approval = true;
            required_approvers.push("multi_party_approver".to_string());
        }
        
        let blocked = risk_score > 0.8;
        let reason = if blocked {
            "High risk score".to_string()
        } else {
            "Risk assessment passed".to_string()
        };
        
        Ok(RiskAssessment {
            blocked,
            reason,
            requires_manual_approval,
            required_approvers,
            risk_score,
        })
    }
}

impl WalletType {
    pub fn name(&self) -> &'static str {
        match self {
            WalletType::HotWallet => "Hot Wallet",
            WalletType::ColdWallet => "Cold Wallet",
            WalletType::Multisig => "Multi-signature",
            WalletType::Hardware => "Hardware Wallet",
            WalletType::Agent => "Agent Wallet",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_wallet_manager_creation() {
        let manager = WalletManager::new();
        let wallets = manager.list_wallets().await;
        assert!(wallets.is_empty());
    }
    
    #[tokio::test]
    async fn test_wallet_registration() {
        let manager = WalletManager::new();
        
        let wallet_info = WalletInfo {
            wallet_id: "test-wallet".to_string(),
            name: "Test Wallet".to_string(),
            wallet_type: WalletType::HotWallet,
            security_level: SecurityLevel::Medium,
            public_key: "test-pubkey".to_string(),
            agent_id: Some("agent-1".to_string()),
            created_at: Utc::now(),
            last_used: None,
            status: WalletStatus::Active,
            metadata: HashMap::new(),
            spending_limits: SpendingLimits::default(),
            risk_controls: RiskControls::default(),
        };
        
        manager.register_wallet(wallet_info, None).await.unwrap();
        
        let wallets = manager.list_wallets().await;
        assert_eq!(wallets.len(), 1);
        assert_eq!(wallets[0].wallet_id, "test-wallet");
    }
    
    #[tokio::test]
    async fn test_risk_assessment() {
        let risk_engine = RiskEngine::new();
        
        let wallet_info = WalletInfo {
            wallet_id: "test-wallet".to_string(),
            name: "Test Wallet".to_string(),
            wallet_type: WalletType::HotWallet,
            security_level: SecurityLevel::High,
            public_key: "test-pubkey".to_string(),
            agent_id: None,
            created_at: Utc::now(),
            last_used: None,
            status: WalletStatus::Active,
            metadata: HashMap::new(),
            spending_limits: SpendingLimits::default(),
            risk_controls: RiskControls::default(),
        };
        
        let transaction_data = vec![0u8; 6000]; // Large transaction
        let assessment = risk_engine.assess_transaction_risk(&wallet_info, &transaction_data, "test-agent").await.unwrap();
        
        assert!(assessment.requires_manual_approval);
        assert!(!assessment.required_approvers.is_empty());
    }
}