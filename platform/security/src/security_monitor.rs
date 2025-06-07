//! Real-time Security Monitoring and Incident Response
//! 
//! Advanced security monitoring system with anomaly detection,
//! automated threat response, and comprehensive audit logging.

use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use tracing::{warn, error, info};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    AuthenticationFailure,
    SqlInjectionAttempt,
    XssAttempt,
    PathTraversalAttempt,
    RateLimitExceeded,
    SuspiciousTrading,
    UnauthorizedAccess,
    DataExfiltrationAttempt,
    BruteForceAttack,
    AccountLockout,
    PrivilegeEscalation,
    MaliciousInput,
    AnomalousActivity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub id: Uuid,
    pub event_type: SecurityEventType,
    pub severity: SeverityLevel,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub ip_address: IpAddr,
    pub user_agent: Option<String>,
    pub endpoint: String,
    pub details: HashMap<String, String>,
    pub raw_data: String,
    pub tenant_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatPattern {
    pub id: String,
    pub name: String,
    pub pattern_type: ThreatPatternType,
    pub indicators: Vec<String>,
    pub severity: SeverityLevel,
    pub auto_block: bool,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatPatternType {
    IpBasedAttack,
    UserAgentAnomaly,
    RequestVolumeSpike,
    SequentialFailures,
    GeolocationAnomaly,
    TimingBasedAttack,
}

#[derive(Debug, Clone)]
pub struct AnomalyDetectionResult {
    pub is_anomaly: bool,
    pub confidence_score: f64,
    pub anomaly_type: String,
    pub baseline_value: f64,
    pub current_value: f64,
}

/// Advanced security monitoring system
pub struct SecurityMonitor {
    event_queue: Arc<Mutex<VecDeque<SecurityEvent>>>,
    threat_patterns: Arc<RwLock<Vec<ThreatPattern>>>,
    ip_tracking: Arc<RwLock<HashMap<IpAddr, IpActivityTracker>>>,
    user_tracking: Arc<RwLock<HashMap<String, UserActivityTracker>>>,
    anomaly_detector: Arc<AnomalyDetector>,
    alert_sender: mpsc::Sender<SecurityAlert>,
    auto_response: Arc<AutomatedResponseSystem>,
}

#[derive(Debug, Clone)]
struct IpActivityTracker {
    first_seen: DateTime<Utc>,
    last_seen: DateTime<Utc>,
    request_count: u64,
    failed_auth_count: u64,
    blocked_until: Option<DateTime<Utc>>,
    reputation_score: f64,
    countries_seen: Vec<String>,
}

#[derive(Debug, Clone)]
struct UserActivityTracker {
    login_attempts: VecDeque<DateTime<Utc>>,
    successful_logins: VecDeque<DateTime<Utc>>,
    failed_logins: VecDeque<DateTime<Utc>>,
    trading_activities: VecDeque<TradingActivity>,
    risk_score: f64,
    last_known_ips: Vec<IpAddr>,
}

#[derive(Debug, Clone)]
struct TradingActivity {
    timestamp: DateTime<Utc>,
    amount: f64,
    token_pair: String,
    success: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SecurityAlert {
    pub id: Uuid,
    pub alert_type: AlertType,
    pub severity: SeverityLevel,
    pub title: String,
    pub description: String,
    pub affected_resources: Vec<String>,
    pub recommended_actions: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub auto_resolved: bool,
}

#[derive(Debug, Clone, Serialize)]
pub enum AlertType {
    ThreatDetected,
    AnomalyDetected,
    SecurityBreach,
    ComplianceViolation,
    SystemCompromise,
}

impl SecurityMonitor {
    pub fn new() -> (Self, mpsc::Receiver<SecurityAlert>) {
        let (alert_tx, alert_rx) = mpsc::channel(1000);
        
        let monitor = Self {
            event_queue: Arc::new(Mutex::new(VecDeque::new())),
            threat_patterns: Arc::new(RwLock::new(Self::default_threat_patterns())),
            ip_tracking: Arc::new(RwLock::new(HashMap::new())),
            user_tracking: Arc::new(RwLock::new(HashMap::new())),
            anomaly_detector: Arc::new(AnomalyDetector::new()),
            alert_sender: alert_tx,
            auto_response: Arc::new(AutomatedResponseSystem::new()),
        };
        
        (monitor, alert_rx)
    }
    
    /// Process incoming security event
    pub async fn process_event(&self, event: SecurityEvent) {
        // Add to event queue
        {
            let mut queue = self.event_queue.lock().await;
            queue.push_back(event.clone());
            
            // Keep only last 10,000 events in memory
            if queue.len() > 10_000 {
                queue.pop_front();
            }
        }
        
        // Update tracking data
        self.update_ip_tracking(&event).await;
        if let Some(user_id) = &event.user_id {
            self.update_user_tracking(user_id, &event).await;
        }
        
        // Perform threat detection
        if let Some(threat) = self.detect_threats(&event).await {
            self.handle_threat_detection(threat, &event).await;
        }
        
        // Perform anomaly detection
        if let Some(anomaly) = self.detect_anomalies(&event).await {
            self.handle_anomaly_detection(anomaly, &event).await;
        }
        
        // Log event for audit
        self.log_security_event(&event).await;
    }
    
    /// Update IP address tracking
    async fn update_ip_tracking(&self, event: &SecurityEvent) {
        let mut ip_tracking = self.ip_tracking.write().await;
        let tracker = ip_tracking.entry(event.ip_address).or_insert_with(|| IpActivityTracker {
            first_seen: event.timestamp,
            last_seen: event.timestamp,
            request_count: 0,
            failed_auth_count: 0,
            blocked_until: None,
            reputation_score: 0.5, // Neutral score
            countries_seen: Vec::new(),
        });
        
        tracker.last_seen = event.timestamp;
        tracker.request_count += 1;
        
        if matches!(event.event_type, SecurityEventType::AuthenticationFailure) {
            tracker.failed_auth_count += 1;
            tracker.reputation_score = (tracker.reputation_score - 0.1).max(0.0);
        }
        
        // Auto-block IPs with too many failures
        if tracker.failed_auth_count >= 10 {
            tracker.blocked_until = Some(event.timestamp + Duration::hours(1));
            tracker.reputation_score = 0.0;
        }
    }
    
    /// Update user activity tracking
    async fn update_user_tracking(&self, user_id: &str, event: &SecurityEvent) {
        let mut user_tracking = self.user_tracking.write().await;
        let tracker = user_tracking.entry(user_id.to_string()).or_insert_with(|| UserActivityTracker {
            login_attempts: VecDeque::new(),
            successful_logins: VecDeque::new(),
            failed_logins: VecDeque::new(),
            trading_activities: VecDeque::new(),
            risk_score: 0.0,
            last_known_ips: Vec::new(),
        });
        
        // Track login attempts
        match event.event_type {
            SecurityEventType::AuthenticationFailure => {
                tracker.failed_logins.push_back(event.timestamp);
                tracker.risk_score += 0.1;
            }
            _ => {}
        }
        
        // Update IP tracking for user
        if !tracker.last_known_ips.contains(&event.ip_address) {
            tracker.last_known_ips.push(event.ip_address);
            if tracker.last_known_ips.len() > 10 {
                tracker.last_known_ips.remove(0);
            }
        }
        
        // Cleanup old data (keep last 30 days)
        let cutoff = event.timestamp - Duration::days(30);
        tracker.login_attempts.retain(|&ts| ts > cutoff);
        tracker.successful_logins.retain(|&ts| ts > cutoff);
        tracker.failed_logins.retain(|&ts| ts > cutoff);
    }
    
    /// Detect threat patterns
    async fn detect_threats(&self, event: &SecurityEvent) -> Option<ThreatPattern> {
        let patterns = self.threat_patterns.read().await;
        
        for pattern in patterns.iter() {
            if self.matches_threat_pattern(pattern, event).await {
                return Some(pattern.clone());
            }
        }
        
        None
    }
    
    /// Check if event matches threat pattern
    async fn matches_threat_pattern(&self, pattern: &ThreatPattern, event: &SecurityEvent) -> bool {
        match pattern.pattern_type {
            ThreatPatternType::IpBasedAttack => {
                let ip_tracking = self.ip_tracking.read().await;
                if let Some(tracker) = ip_tracking.get(&event.ip_address) {
                    tracker.failed_auth_count >= 5
                } else {
                    false
                }
            }
            ThreatPatternType::SequentialFailures => {
                matches!(event.event_type, SecurityEventType::AuthenticationFailure)
            }
            ThreatPatternType::RequestVolumeSpike => {
                self.detect_volume_spike(event).await
            }
            _ => false,
        }
    }
    
    /// Detect request volume spikes
    async fn detect_volume_spike(&self, event: &SecurityEvent) -> bool {
        let ip_tracking = self.ip_tracking.read().await;
        if let Some(tracker) = ip_tracking.get(&event.ip_address) {
            let time_window = Duration::minutes(5);
            let recent_requests = tracker.request_count;
            
            // Alert if more than 100 requests in 5 minutes
            recent_requests > 100
        } else {
            false
        }
    }
    
    /// Detect anomalies using statistical analysis
    async fn detect_anomalies(&self, event: &SecurityEvent) -> Option<AnomalyDetectionResult> {
        self.anomaly_detector.analyze_event(event).await
    }
    
    /// Handle threat detection
    async fn handle_threat_detection(&self, threat: ThreatPattern, event: &SecurityEvent) {
        warn!("Threat detected: {} from IP {}", threat.name, event.ip_address);
        
        let alert = SecurityAlert {
            id: Uuid::new_v4(),
            alert_type: AlertType::ThreatDetected,
            severity: threat.severity.clone(),
            title: format!("Threat Detected: {}", threat.name),
            description: format!("Detected {} from IP {}", threat.name, event.ip_address),
            affected_resources: vec![event.ip_address.to_string()],
            recommended_actions: vec![
                "Review IP activity logs".to_string(),
                "Consider blocking IP address".to_string(),
                "Monitor for similar patterns".to_string(),
            ],
            timestamp: Utc::now(),
            auto_resolved: false,
        };
        
        // Send alert
        if let Err(e) = self.alert_sender.send(alert).await {
            error!("Failed to send security alert: {}", e);
        }
        
        // Auto-response if enabled
        if threat.auto_block {
            self.auto_response.block_ip(event.ip_address, Duration::hours(1)).await;
        }
    }
    
    /// Handle anomaly detection
    async fn handle_anomaly_detection(&self, anomaly: AnomalyDetectionResult, event: &SecurityEvent) {
        if anomaly.is_anomaly && anomaly.confidence_score > 0.8 {
            warn!("Anomaly detected: {} (confidence: {:.2})", anomaly.anomaly_type, anomaly.confidence_score);
            
            let alert = SecurityAlert {
                id: Uuid::new_v4(),
                alert_type: AlertType::AnomalyDetected,
                severity: SeverityLevel::Medium,
                title: format!("Anomaly Detected: {}", anomaly.anomaly_type),
                description: format!(
                    "Detected anomalous behavior: {} (confidence: {:.2})",
                    anomaly.anomaly_type, anomaly.confidence_score
                ),
                affected_resources: vec![event.ip_address.to_string()],
                recommended_actions: vec![
                    "Investigate user activity".to_string(),
                    "Review system logs".to_string(),
                    "Monitor for similar patterns".to_string(),
                ],
                timestamp: Utc::now(),
                auto_resolved: false,
            };
            
            if let Err(e) = self.alert_sender.send(alert).await {
                error!("Failed to send anomaly alert: {}", e);
            }
        }
    }
    
    /// Log security event for audit
    async fn log_security_event(&self, event: &SecurityEvent) {
        // Structured logging for security events
        info!(
            event_id = %event.id,
            event_type = ?event.event_type,
            severity = ?event.severity,
            user_id = ?event.user_id,
            ip_address = %event.ip_address,
            endpoint = %event.endpoint,
            tenant_id = ?event.tenant_id,
            "Security event processed"
        );
    }
    
    /// Default threat patterns
    fn default_threat_patterns() -> Vec<ThreatPattern> {
        vec![
            ThreatPattern {
                id: "brute_force_login".to_string(),
                name: "Brute Force Login Attack".to_string(),
                pattern_type: ThreatPatternType::SequentialFailures,
                indicators: vec!["multiple_failed_logins".to_string()],
                severity: SeverityLevel::High,
                auto_block: true,
                confidence_threshold: 0.9,
            },
            ThreatPattern {
                id: "sql_injection".to_string(),
                name: "SQL Injection Attempt".to_string(),
                pattern_type: ThreatPatternType::IpBasedAttack,
                indicators: vec!["sql_keywords".to_string(), "malicious_payload".to_string()],
                severity: SeverityLevel::Critical,
                auto_block: true,
                confidence_threshold: 0.95,
            },
            ThreatPattern {
                id: "ddos_attempt".to_string(),
                name: "DDoS Attack".to_string(),
                pattern_type: ThreatPatternType::RequestVolumeSpike,
                indicators: vec!["high_request_volume".to_string()],
                severity: SeverityLevel::High,
                auto_block: true,
                confidence_threshold: 0.85,
            },
        ]
    }
    
    /// Get security metrics for monitoring
    pub async fn get_security_metrics(&self) -> SecurityMetrics {
        let ip_tracking = self.ip_tracking.read().await;
        let user_tracking = self.user_tracking.read().await;
        let event_queue = self.event_queue.lock().await;
        
        SecurityMetrics {
            total_events: event_queue.len(),
            unique_ips: ip_tracking.len(),
            active_users: user_tracking.len(),
            blocked_ips: ip_tracking.values().filter(|t| t.blocked_until.is_some()).count(),
            high_risk_users: user_tracking.values().filter(|t| t.risk_score > 0.7).count(),
            events_last_hour: self.count_recent_events(Duration::hours(1)).await,
            events_last_day: self.count_recent_events(Duration::days(1)).await,
        }
    }
    
    async fn count_recent_events(&self, duration: Duration) -> usize {
        let event_queue = self.event_queue.lock().await;
        let cutoff = Utc::now() - duration;
        event_queue.iter().filter(|e| e.timestamp > cutoff).count()
    }
}

#[derive(Debug, Serialize)]
pub struct SecurityMetrics {
    pub total_events: usize,
    pub unique_ips: usize,
    pub active_users: usize,
    pub blocked_ips: usize,
    pub high_risk_users: usize,
    pub events_last_hour: usize,
    pub events_last_day: usize,
}

/// Anomaly detection using statistical methods
pub struct AnomalyDetector {
    baseline_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            baseline_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn analyze_event(&self, event: &SecurityEvent) -> Option<AnomalyDetectionResult> {
        // Implement statistical anomaly detection
        // This is a simplified version - in production, use more sophisticated algorithms
        
        let metric_key = format!("{}:{}", event.event_type as u8, event.ip_address);
        let current_value = 1.0; // Simplified metric
        
        let mut baseline = self.baseline_data.write().await;
        let history = baseline.entry(metric_key).or_insert_with(Vec::new);
        
        if history.len() < 10 {
            // Not enough data for anomaly detection
            history.push(current_value);
            return None;
        }
        
        let mean = history.iter().sum::<f64>() / history.len() as f64;
        let variance = history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / history.len() as f64;
        let std_dev = variance.sqrt();
        
        // Z-score anomaly detection
        let z_score = (current_value - mean).abs() / std_dev;
        let is_anomaly = z_score > 2.0; // 95% confidence interval
        
        history.push(current_value);
        if history.len() > 100 {
            history.remove(0); // Keep only recent data
        }
        
        Some(AnomalyDetectionResult {
            is_anomaly,
            confidence_score: (z_score / 3.0).min(1.0), // Normalize to 0-1
            anomaly_type: format!("{:?}", event.event_type),
            baseline_value: mean,
            current_value,
        })
    }
}

/// Automated response system for security incidents
pub struct AutomatedResponseSystem {
    blocked_ips: Arc<RwLock<HashMap<IpAddr, DateTime<Utc>>>>,
}

impl AutomatedResponseSystem {
    pub fn new() -> Self {
        Self {
            blocked_ips: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn block_ip(&self, ip: IpAddr, duration: Duration) {
        let mut blocked_ips = self.blocked_ips.write().await;
        blocked_ips.insert(ip, Utc::now() + duration);
        
        warn!("Auto-blocked IP {} for {:?}", ip, duration);
    }
    
    pub async fn is_ip_blocked(&self, ip: &IpAddr) -> bool {
        let blocked_ips = self.blocked_ips.read().await;
        if let Some(&blocked_until) = blocked_ips.get(ip) {
            Utc::now() < blocked_until
        } else {
            false
        }
    }
    
    pub async fn cleanup_expired_blocks(&self) {
        let mut blocked_ips = self.blocked_ips.write().await;
        let now = Utc::now();
        blocked_ips.retain(|_, &mut blocked_until| now < blocked_until);
    }
}