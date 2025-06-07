//! Hardware wallet integration for enhanced security
//!
//! This module provides integration with hardware wallets like Ledger and Trezor
//! for maximum security in agent wallet operations.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::{Ed25519Signature, SigningError, TransactionSigner};

/// Hardware wallet types supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HardwareWalletType {
    Ledger,
    Trezor,
    YubiKey,
    SecureElement,
}

/// Hardware wallet device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareDevice {
    /// Device unique identifier
    pub device_id: String,
    /// Hardware wallet type
    pub wallet_type: HardwareWalletType,
    /// Device firmware version
    pub firmware_version: String,
    /// Device serial number (if available)
    pub serial_number: Option<String>,
    /// Device connection status
    pub connection_status: ConnectionStatus,
    /// Supported derivation paths
    pub supported_paths: Vec<String>,
    /// Last communication timestamp
    pub last_seen: DateTime<Utc>,
    /// Device capabilities
    pub capabilities: DeviceCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Locked,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Supports Ed25519 signing
    pub ed25519_support: bool,
    /// Supports ECDSA signing
    pub ecdsa_support: bool,
    /// Supports multiple accounts
    pub multi_account: bool,
    /// Supports custom derivation paths
    pub custom_derivation: bool,
    /// Supports transaction review on device
    pub tx_review: bool,
    /// Maximum transaction size supported
    pub max_tx_size: usize,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            ed25519_support: true,
            ecdsa_support: true,
            multi_account: true,
            custom_derivation: true,
            tx_review: true,
            max_tx_size: 65536, // 64KB
        }
    }
}

/// Hardware wallet account configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAccount {
    /// Account identifier
    pub account_id: String,
    /// Associated device ID
    pub device_id: String,
    /// Derivation path (e.g., "m/44'/501'/0'/0'")
    pub derivation_path: String,
    /// Account public key
    pub public_key: String,
    /// Account name/label
    pub name: String,
    /// Account creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
    /// Account status
    pub status: AccountStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccountStatus {
    Active,
    Inactive,
    Locked,
    Compromised,
}

/// Hardware wallet operation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOperation {
    /// Operation ID
    pub operation_id: String,
    /// Target device ID
    pub device_id: String,
    /// Account ID
    pub account_id: String,
    /// Operation type
    pub operation_type: OperationType,
    /// Operation data
    pub data: Vec<u8>,
    /// Requested timestamp
    pub requested_at: DateTime<Utc>,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Requires user confirmation
    pub user_confirmation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Sign,
    GetPublicKey,
    GetAddress,
    VerifyAddress,
    SignMessage,
}

/// Hardware wallet operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOperationResult {
    pub operation_id: String,
    pub status: OperationStatus,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperationStatus {
    Success,
    UserRejected,
    DeviceError,
    Timeout,
    InvalidOperation,
}

/// Hardware wallet manager trait
#[async_trait]
pub trait HardwareWalletManager: Send + Sync {
    /// Discover available hardware wallets
    async fn discover_devices(&self) -> Result<Vec<HardwareDevice>>;
    
    /// Connect to a specific device
    async fn connect_device(&self, device_id: &str) -> Result<()>;
    
    /// Disconnect from a device
    async fn disconnect_device(&self, device_id: &str) -> Result<()>;
    
    /// Get device status
    async fn get_device_status(&self, device_id: &str) -> Result<ConnectionStatus>;
    
    /// List accounts on a device
    async fn list_accounts(&self, device_id: &str) -> Result<Vec<HardwareAccount>>;
    
    /// Create a new account
    async fn create_account(
        &self,
        device_id: &str,
        derivation_path: &str,
        name: &str,
    ) -> Result<HardwareAccount>;
    
    /// Execute a hardware operation
    async fn execute_operation(&self, operation: HardwareOperation) -> Result<HardwareOperationResult>;
}

/// Ledger hardware wallet implementation
pub struct LedgerManager {
    devices: Arc<RwLock<HashMap<String, HardwareDevice>>>,
    accounts: Arc<RwLock<HashMap<String, HardwareAccount>>>,
    operations: Arc<RwLock<HashMap<String, HardwareOperationResult>>>,
}

impl LedgerManager {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            accounts: Arc::new(RwLock::new(HashMap::new())),
            operations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Simulate Ledger device communication
    async fn communicate_with_device(
        &self,
        device_id: &str,
        operation: &HardwareOperation,
    ) -> Result<serde_json::Value> {
        // In a real implementation, this would use the Ledger SDK/API
        // For now, we simulate the communication
        
        let devices = self.devices.read().await;
        let device = devices.get(device_id)
            .ok_or_else(|| anyhow!("Device not found: {}", device_id))?;
        
        if device.connection_status != ConnectionStatus::Connected {
            return Err(anyhow!("Device not connected: {}", device_id));
        }
        
        // Simulate device operation
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        
        match operation.operation_type {
            OperationType::Sign => {
                // Simulate signing operation
                let fake_signature = Ed25519Signature::new([42u8; 64]);
                Ok(serde_json::json!({
                    "signature": fake_signature.to_base58()
                }))
            },
            OperationType::GetPublicKey => {
                // Simulate public key retrieval
                Ok(serde_json::json!({
                    "public_key": "fake_pubkey_from_ledger"
                }))
            },
            OperationType::GetAddress => {
                // Simulate address derivation
                Ok(serde_json::json!({
                    "address": "fake_address_from_ledger"
                }))
            },
            OperationType::VerifyAddress => {
                // Simulate address verification
                Ok(serde_json::json!({
                    "verified": true
                }))
            },
            OperationType::SignMessage => {
                // Simulate message signing
                let fake_signature = Ed25519Signature::new([99u8; 64]);
                Ok(serde_json::json!({
                    "signature": fake_signature.to_base58()
                }))
            },
        }
    }
}

#[async_trait]
impl HardwareWalletManager for LedgerManager {
    async fn discover_devices(&self) -> Result<Vec<HardwareDevice>> {
        // In a real implementation, this would scan for Ledger devices
        // For now, we simulate finding one device
        
        let device = HardwareDevice {
            device_id: "ledger_001".to_string(),
            wallet_type: HardwareWalletType::Ledger,
            firmware_version: "2.1.0".to_string(),
            serial_number: Some("LEDGER123456".to_string()),
            connection_status: ConnectionStatus::Disconnected,
            supported_paths: vec![
                "m/44'/501'/0'/0'".to_string(),  // Solana
                "m/44'/60'/0'/0'".to_string(),   // Ethereum
                "m/44'/0'/0'/0'".to_string(),    // Bitcoin
            ],
            last_seen: Utc::now(),
            capabilities: DeviceCapabilities::default(),
        };
        
        let mut devices = self.devices.write().await;
        devices.insert(device.device_id.clone(), device.clone());
        
        Ok(vec![device])
    }
    
    async fn connect_device(&self, device_id: &str) -> Result<()> {
        let mut devices = self.devices.write().await;
        
        if let Some(device) = devices.get_mut(device_id) {
            // Simulate connection process
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            
            device.connection_status = ConnectionStatus::Connected;
            device.last_seen = Utc::now();
            
            tracing::info!("Connected to Ledger device: {}", device_id);
        } else {
            return Err(anyhow!("Device not found: {}", device_id));
        }
        
        Ok(())
    }
    
    async fn disconnect_device(&self, device_id: &str) -> Result<()> {
        let mut devices = self.devices.write().await;
        
        if let Some(device) = devices.get_mut(device_id) {
            device.connection_status = ConnectionStatus::Disconnected;
            tracing::info!("Disconnected from Ledger device: {}", device_id);
        } else {
            return Err(anyhow!("Device not found: {}", device_id));
        }
        
        Ok(())
    }
    
    async fn get_device_status(&self, device_id: &str) -> Result<ConnectionStatus> {
        let devices = self.devices.read().await;
        
        devices.get(device_id)
            .map(|device| device.connection_status.clone())
            .ok_or_else(|| anyhow!("Device not found: {}", device_id))
    }
    
    async fn list_accounts(&self, device_id: &str) -> Result<Vec<HardwareAccount>> {
        let accounts = self.accounts.read().await;
        
        let device_accounts: Vec<HardwareAccount> = accounts.values()
            .filter(|account| account.device_id == device_id)
            .cloned()
            .collect();
        
        Ok(device_accounts)
    }
    
    async fn create_account(
        &self,
        device_id: &str,
        derivation_path: &str,
        name: &str,
    ) -> Result<HardwareAccount> {
        // Check if device exists and is connected
        let device_status = self.get_device_status(device_id).await?;
        if device_status != ConnectionStatus::Connected {
            return Err(anyhow!("Device must be connected to create account"));
        }
        
        // Create operation to get public key for this derivation path
        let operation = HardwareOperation {
            operation_id: uuid::Uuid::new_v4().to_string(),
            device_id: device_id.to_string(),
            account_id: "".to_string(), // Will be set after creation
            operation_type: OperationType::GetPublicKey,
            data: derivation_path.as_bytes().to_vec(),
            requested_at: Utc::now(),
            timeout_seconds: 30,
            user_confirmation: false,
        };
        
        let result = self.execute_operation(operation).await?;
        
        if result.status != OperationStatus::Success {
            return Err(anyhow!("Failed to get public key from device"));
        }
        
        let public_key = result.result
            .and_then(|r| r.get("public_key"))
            .and_then(|pk| pk.as_str())
            .ok_or_else(|| anyhow!("Invalid public key response"))?;
        
        let account = HardwareAccount {
            account_id: uuid::Uuid::new_v4().to_string(),
            device_id: device_id.to_string(),
            derivation_path: derivation_path.to_string(),
            public_key: public_key.to_string(),
            name: name.to_string(),
            created_at: Utc::now(),
            last_used: None,
            status: AccountStatus::Active,
        };
        
        // Store the account
        let mut accounts = self.accounts.write().await;
        accounts.insert(account.account_id.clone(), account.clone());
        
        Ok(account)
    }
    
    async fn execute_operation(&self, operation: HardwareOperation) -> Result<HardwareOperationResult> {
        let operation_id = operation.operation_id.clone();
        
        // Validate device connection
        let device_status = self.get_device_status(&operation.device_id).await?;
        if device_status != ConnectionStatus::Connected {
            let result = HardwareOperationResult {
                operation_id,
                status: OperationStatus::DeviceError,
                result: None,
                error: Some("Device not connected".to_string()),
                completed_at: Utc::now(),
            };
            
            let mut operations = self.operations.write().await;
            operations.insert(result.operation_id.clone(), result.clone());
            
            return Ok(result);
        }
        
        // Execute the operation with timeout
        let timeout_duration = std::time::Duration::from_secs(operation.timeout_seconds);
        
        let operation_result = tokio::time::timeout(
            timeout_duration,
            self.communicate_with_device(&operation.device_id, &operation)
        ).await;
        
        let result = match operation_result {
            Ok(Ok(data)) => HardwareOperationResult {
                operation_id,
                status: OperationStatus::Success,
                result: Some(data),
                error: None,
                completed_at: Utc::now(),
            },
            Ok(Err(e)) => HardwareOperationResult {
                operation_id,
                status: OperationStatus::DeviceError,
                result: None,
                error: Some(e.to_string()),
                completed_at: Utc::now(),
            },
            Err(_) => HardwareOperationResult {
                operation_id,
                status: OperationStatus::Timeout,
                result: None,
                error: Some("Operation timed out".to_string()),
                completed_at: Utc::now(),
            },
        };
        
        // Store the operation result
        let mut operations = self.operations.write().await;
        operations.insert(result.operation_id.clone(), result.clone());
        
        Ok(result)
    }
}

/// Hardware wallet transaction signer
pub struct HardwareWalletSigner {
    manager: Arc<dyn HardwareWalletManager>,
    device_id: String,
    account_id: String,
}

impl HardwareWalletSigner {
    pub fn new(
        manager: Arc<dyn HardwareWalletManager>,
        device_id: String,
        account_id: String,
    ) -> Self {
        Self {
            manager,
            device_id,
            account_id,
        }
    }
}

#[async_trait]
impl TransactionSigner for HardwareWalletSigner {
    async fn sign(&self, transaction: &[u8]) -> Result<Ed25519Signature, SigningError> {
        let operation = HardwareOperation {
            operation_id: uuid::Uuid::new_v4().to_string(),
            device_id: self.device_id.clone(),
            account_id: self.account_id.clone(),
            operation_type: OperationType::Sign,
            data: transaction.to_vec(),
            requested_at: Utc::now(),
            timeout_seconds: 60,
            user_confirmation: true,
        };
        
        let result = self.manager.execute_operation(operation).await
            .map_err(|e| SigningError::SigningFailed(e.to_string()))?;
        
        match result.status {
            OperationStatus::Success => {
                let signature_str = result.result
                    .and_then(|r| r.get("signature"))
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| SigningError::SigningFailed("Invalid signature response".to_string()))?;
                
                Ed25519Signature::from_base58(signature_str)
                    .map_err(|e| SigningError::SigningFailed(format!("Invalid signature format: {}", e)))
            },
            OperationStatus::UserRejected => {
                Err(SigningError::SigningFailed("User rejected transaction".to_string()))
            },
            OperationStatus::DeviceError => {
                Err(SigningError::SigningFailed(
                    result.error.unwrap_or_else(|| "Unknown device error".to_string())
                ))
            },
            OperationStatus::Timeout => {
                Err(SigningError::SigningFailed("Hardware wallet operation timed out".to_string()))
            },
            OperationStatus::InvalidOperation => {
                Err(SigningError::SigningFailed("Invalid operation for hardware wallet".to_string()))
            },
        }
    }
    
    fn verify(&self, _message: &[u8], _signature: &Ed25519Signature) -> Result<bool, SigningError> {
        // Hardware wallets typically don't support verification directly
        // Verification would be done using the public key
        Err(SigningError::UnsupportedOperation(
            "Verification not supported by hardware wallet signer".to_string()
        ))
    }
    
    fn get_public_key(&self) -> Result<Vec<u8>, SigningError> {
        // This would need to be implemented as an async operation
        // For now, return an error indicating async operation required
        Err(SigningError::UnsupportedOperation(
            "Use hardware wallet manager to get public key asynchronously".to_string()
        ))
    }
}

/// Hardware wallet integration factory
pub struct HardwareWalletFactory;

impl HardwareWalletFactory {
    /// Create a manager for the specified hardware wallet type
    pub fn create_manager(wallet_type: HardwareWalletType) -> Result<Arc<dyn HardwareWalletManager>> {
        match wallet_type {
            HardwareWalletType::Ledger => {
                Ok(Arc::new(LedgerManager::new()))
            },
            HardwareWalletType::Trezor => {
                // Trezor implementation would go here
                Err(anyhow!("Trezor support not yet implemented"))
            },
            HardwareWalletType::YubiKey => {
                // YubiKey implementation would go here
                Err(anyhow!("YubiKey support not yet implemented"))
            },
            HardwareWalletType::SecureElement => {
                // Secure Element implementation would go here
                Err(anyhow!("Secure Element support not yet implemented"))
            },
        }
    }
    
    /// Auto-discover all supported hardware wallets
    pub async fn discover_all_devices() -> Result<Vec<HardwareDevice>> {
        let mut all_devices = Vec::new();
        
        // Try each supported hardware wallet type
        for wallet_type in [HardwareWalletType::Ledger] {
            if let Ok(manager) = Self::create_manager(wallet_type) {
                if let Ok(devices) = manager.discover_devices().await {
                    all_devices.extend(devices);
                }
            }
        }
        
        Ok(all_devices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ledger_manager() {
        let manager = LedgerManager::new();
        
        // Test device discovery
        let devices = manager.discover_devices().await.unwrap();
        assert!(!devices.is_empty());
        
        let device_id = &devices[0].device_id;
        
        // Test connection
        manager.connect_device(device_id).await.unwrap();
        
        let status = manager.get_device_status(device_id).await.unwrap();
        assert_eq!(status, ConnectionStatus::Connected);
        
        // Test account creation
        let account = manager.create_account(
            device_id,
            "m/44'/501'/0'/0'",
            "Test Account"
        ).await.unwrap();
        
        assert_eq!(account.device_id, *device_id);
        assert_eq!(account.derivation_path, "m/44'/501'/0'/0'");
        
        // Test disconnection
        manager.disconnect_device(device_id).await.unwrap();
    }
    
    #[tokio::test]
    async fn test_hardware_wallet_factory() {
        let devices = HardwareWalletFactory::discover_all_devices().await.unwrap();
        // Should find at least one simulated device
        assert!(!devices.is_empty());
        
        let manager = HardwareWalletFactory::create_manager(HardwareWalletType::Ledger).unwrap();
        let discovered = manager.discover_devices().await.unwrap();
        assert!(!discovered.is_empty());
    }
    
    #[tokio::test]
    async fn test_hardware_wallet_signer() {
        let manager = Arc::new(LedgerManager::new());
        
        // Discover and connect device
        let devices = manager.discover_devices().await.unwrap();
        let device_id = devices[0].device_id.clone();
        manager.connect_device(&device_id).await.unwrap();
        
        // Create account
        let account = manager.create_account(
            &device_id,
            "m/44'/501'/0'/0'",
            "Test Signer Account"
        ).await.unwrap();
        
        // Create signer
        let signer = HardwareWalletSigner::new(
            manager.clone(),
            device_id,
            account.account_id,
        );
        
        // Test signing
        let transaction = b"test transaction data";
        let signature = signer.sign(transaction).await.unwrap();
        
        // Signature should be valid format
        assert_eq!(signature.to_bytes().len(), 64);
    }
}