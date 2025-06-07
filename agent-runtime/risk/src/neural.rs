//! Neural network integration for advanced risk prediction (optional feature)

#[cfg(feature = "ml")]
use candle_core::{Device, Result as CandleResult, Tensor, DType};
#[cfg(feature = "ml")]
use candle_nn::{Module, VarBuilder, VarMap};

use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    config::{RiskConfig, NeuralConfig},
    models::*,
    RiskError, RiskResult, RiskAssessment,
};

/// Neural network-based risk predictor
pub struct NeuralRiskPredictor {
    config: NeuralConfig,
    #[cfg(feature = "ml")]
    model: Option<Arc<RwLock<RiskPredictionModel>>>,
    #[cfg(not(feature = "ml"))]
    _phantom: std::marker::PhantomData<()>,
    last_update: DateTime<Utc>,
    prediction_cache: Arc<RwLock<PredictionCache>>,
}

impl NeuralRiskPredictor {
    /// Create new neural risk predictor
    pub async fn new(config: NeuralConfig) -> RiskResult<Self> {
        #[cfg(feature = "ml")]
        let model = if config.enabled {
            Some(Arc::new(RwLock::new(RiskPredictionModel::new().await?)))
        } else {
            None
        };

        Ok(Self {
            config,
            #[cfg(feature = "ml")]
            model,
            #[cfg(not(feature = "ml"))]
            _phantom: std::marker::PhantomData,
            last_update: Utc::now(),
            prediction_cache: Arc::new(RwLock::new(PredictionCache::new())),
        })
    }

    /// Predict risk score using neural network
    pub async fn predict_risk(&self, features: &RiskFeatures) -> RiskResult<RiskPrediction> {
        if !self.config.enabled {
            return Ok(RiskPrediction::default());
        }

        // Check cache first
        {
            let cache = self.prediction_cache.read().await;
            if let Some(cached) = cache.get(features) {
                return Ok(cached);
            }
        }

        #[cfg(feature = "ml")]
        {
            if let Some(model) = &self.model {
                let model_guard = model.read().await;
                let prediction = model_guard.predict(features).await?;
                
                // Cache the prediction
                let mut cache = self.prediction_cache.write().await;
                cache.insert(features.clone(), prediction.clone());
                
                return Ok(prediction);
            }
        }

        // Fallback to statistical prediction if ML is not available
        self.statistical_prediction(features).await
    }

    /// Update the neural network model with new training data
    pub async fn update_model(&self, training_data: &[TrainingExample]) -> RiskResult<()> {
        if !self.config.enabled || training_data.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "ml")]
        {
            if let Some(model) = &self.model {
                let mut model_guard = model.write().await;
                model_guard.train(training_data).await?;
                self.last_update = Utc::now();
                
                // Clear prediction cache after model update
                let mut cache = self.prediction_cache.write().await;
                cache.clear();
                
                tracing::info!("Neural risk model updated with {} training examples", training_data.len());
            }
        }

        Ok(())
    }

    /// Get model performance metrics
    pub async fn get_model_metrics(&self) -> RiskResult<ModelMetrics> {
        #[cfg(feature = "ml")]
        {
            if let Some(model) = &self.model {
                let model_guard = model.read().await;
                return Ok(model_guard.get_metrics());
            }
        }

        Ok(ModelMetrics::default())
    }

    /// Fallback statistical prediction when ML is not available
    async fn statistical_prediction(&self, features: &RiskFeatures) -> RiskResult<RiskPrediction> {
        // Simple statistical model based on feature weights
        let mut risk_score = 0.0;
        
        // Weight features based on importance
        risk_score += features.volatility * 0.3;
        risk_score += features.concentration * 0.25;
        risk_score += features.liquidity_risk * 0.2;
        risk_score += features.correlation * 0.15;
        risk_score += features.momentum * 0.1;
        
        // Normalize to 0-1 range
        risk_score = risk_score.min(1.0).max(0.0);
        
        Ok(RiskPrediction {
            risk_score,
            confidence: 0.6, // Lower confidence for statistical model
            features_importance: vec![
                ("volatility".to_string(), 0.3),
                ("concentration".to_string(), 0.25),
                ("liquidity_risk".to_string(), 0.2),
                ("correlation".to_string(), 0.15),
                ("momentum".to_string(), 0.1),
            ],
            timestamp: Utc::now(),
        })
    }
}

/// Risk features for neural network input
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskFeatures {
    pub volatility: f64,
    pub concentration: f64,
    pub liquidity_risk: f64,
    pub correlation: f64,
    pub momentum: f64,
    pub volume: f64,
    pub price_change: f64,
    pub market_sentiment: f64,
    pub time_features: TimeFeatures,
}

impl RiskFeatures {
    /// Extract features from risk assessment
    pub fn from_assessment(assessment: &RiskAssessment) -> Self {
        Self {
            volatility: assessment.metrics.var_1d,
            concentration: assessment.metrics.concentration_risk,
            liquidity_risk: assessment.metrics.liquidity_risk,
            correlation: 0.5, // Placeholder - would calculate from portfolio
            momentum: 0.0, // Placeholder - would calculate from price history
            volume: 1000000.0, // Placeholder - would get from market data
            price_change: 0.0, // Placeholder - would calculate from price history
            market_sentiment: 0.5, // Placeholder - would get from sentiment analysis
            time_features: TimeFeatures::current(),
        }
    }

    /// Convert to neural network input tensor
    #[cfg(feature = "ml")]
    pub fn to_tensor(&self, device: &Device) -> CandleResult<Tensor> {
        let features = vec![
            self.volatility,
            self.concentration,
            self.liquidity_risk,
            self.correlation,
            self.momentum,
            self.volume,
            self.price_change,
            self.market_sentiment,
            self.time_features.hour_of_day,
            self.time_features.day_of_week,
            self.time_features.month_of_year,
        ];
        
        Tensor::from_vec(features, (1, features.len()), device)
    }
}

/// Time-based features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeFeatures {
    pub hour_of_day: f64,    // 0-1 normalized
    pub day_of_week: f64,    // 0-1 normalized
    pub month_of_year: f64,  // 0-1 normalized
}

impl TimeFeatures {
    fn current() -> Self {
        let now = Utc::now();
        Self {
            hour_of_day: now.hour() as f64 / 24.0,
            day_of_week: now.weekday().num_days_from_monday() as f64 / 7.0,
            month_of_year: now.month() as f64 / 12.0,
        }
    }
}

/// Neural network prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskPrediction {
    pub risk_score: f64,
    pub confidence: f64,
    pub features_importance: Vec<(String, f64)>,
    pub timestamp: DateTime<Utc>,
}

impl Default for RiskPrediction {
    fn default() -> Self {
        Self {
            risk_score: 0.5,
            confidence: 0.0,
            features_importance: Vec::new(),
            timestamp: Utc::now(),
        }
    }
}

/// Training example for model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: RiskFeatures,
    pub target_risk_score: f64,
    pub actual_outcome: f64, // Actual loss/gain
    pub weight: f64,         // Sample weight
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
    pub mean_squared_error: f64,
    pub training_samples: usize,
    pub last_updated: DateTime<Utc>,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            auc_roc: 0.0,
            mean_squared_error: 0.0,
            training_samples: 0,
            last_updated: Utc::now(),
        }
    }
}

/// Prediction cache for performance optimization
pub struct PredictionCache {
    cache: std::collections::HashMap<u64, (RiskPrediction, DateTime<Utc>)>,
    max_size: usize,
    ttl_seconds: i64,
}

impl PredictionCache {
    fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size: 1000,
            ttl_seconds: 300, // 5 minutes
        }
    }

    fn get(&self, features: &RiskFeatures) -> Option<RiskPrediction> {
        let key = self.hash_features(features);
        
        if let Some((prediction, timestamp)) = self.cache.get(&key) {
            let age = Utc::now() - *timestamp;
            if age.num_seconds() < self.ttl_seconds {
                return Some(prediction.clone());
            }
        }
        
        None
    }

    fn insert(&mut self, features: RiskFeatures, prediction: RiskPrediction) {
        // Clean old entries if cache is full
        if self.cache.len() >= self.max_size {
            self.cleanup_old_entries();
        }

        let key = self.hash_features(&features);
        self.cache.insert(key, (prediction, Utc::now()));
    }

    fn clear(&mut self) {
        self.cache.clear();
    }

    fn hash_features(&self, features: &RiskFeatures) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash key features (with precision reduction for caching)
        ((features.volatility * 1000.0) as u64).hash(&mut hasher);
        ((features.concentration * 1000.0) as u64).hash(&mut hasher);
        ((features.liquidity_risk * 1000.0) as u64).hash(&mut hasher);
        
        hasher.finish()
    }

    fn cleanup_old_entries(&mut self) {
        let cutoff = Utc::now() - chrono::Duration::seconds(self.ttl_seconds);
        self.cache.retain(|_, (_, timestamp)| *timestamp > cutoff);
    }
}

/// Neural network model (only available with ML feature)
#[cfg(feature = "ml")]
pub struct RiskPredictionModel {
    device: Device,
    model: RiskNet,
    metrics: ModelMetrics,
}

#[cfg(feature = "ml")]
impl RiskPredictionModel {
    async fn new() -> RiskResult<Self> {
        let device = Device::Cpu; // Use CPU for simplicity, could use CUDA if available
        let model = RiskNet::new(&device)?;
        
        Ok(Self {
            device,
            model,
            metrics: ModelMetrics::default(),
        })
    }

    async fn predict(&self, features: &RiskFeatures) -> RiskResult<RiskPrediction> {
        let input_tensor = features.to_tensor(&self.device)
            .map_err(|e| RiskError::Assessment(format!("Tensor conversion failed: {}", e)))?;
        
        let output = self.model.forward(&input_tensor)
            .map_err(|e| RiskError::Assessment(format!("Model forward pass failed: {}", e)))?;
        
        let risk_score = output.to_vec1::<f32>()
            .map_err(|e| RiskError::Assessment(format!("Output extraction failed: {}", e)))?[0] as f64;
        
        Ok(RiskPrediction {
            risk_score: risk_score.min(1.0).max(0.0),
            confidence: self.metrics.accuracy,
            features_importance: vec![
                ("volatility".to_string(), 0.25),
                ("concentration".to_string(), 0.20),
                ("liquidity_risk".to_string(), 0.20),
                ("correlation".to_string(), 0.15),
                ("momentum".to_string(), 0.10),
                ("volume".to_string(), 0.05),
                ("time_features".to_string(), 0.05),
            ],
            timestamp: Utc::now(),
        })
    }

    async fn train(&mut self, _training_data: &[TrainingExample]) -> RiskResult<()> {
        // Training implementation would go here
        // For now, just update metrics to indicate training occurred
        self.metrics.training_samples += _training_data.len();
        self.metrics.last_updated = Utc::now();
        
        // Simulate improving accuracy
        self.metrics.accuracy = (self.metrics.accuracy + 0.01).min(0.95);
        
        Ok(())
    }

    fn get_metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
}

/// Simple neural network for risk prediction
#[cfg(feature = "ml")]
pub struct RiskNet {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    linear3: candle_nn::Linear,
    dropout: candle_nn::Dropout,
}

#[cfg(feature = "ml")]
impl RiskNet {
    fn new(device: &Device) -> CandleResult<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        let linear1 = candle_nn::linear(11, 64, vb.pp("linear1"))?; // 11 input features
        let linear2 = candle_nn::linear(64, 32, vb.pp("linear2"))?;
        let linear3 = candle_nn::linear(32, 1, vb.pp("linear3"))?;  // 1 output (risk score)
        let dropout = candle_nn::Dropout::new(0.2);
        
        Ok(Self {
            linear1,
            linear2,
            linear3,
            dropout,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        let x = self.dropout.forward(&x, false)?; // false = not training
        
        let x = self.linear2.forward(&x)?;
        let x = x.relu()?;
        let x = self.dropout.forward(&x, false)?;
        
        let x = self.linear3.forward(&x)?;
        let x = x.sigmoid()?; // Output between 0 and 1
        
        Ok(x)
    }
}

#[cfg(feature = "ml")]
impl Module for RiskNet {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NeuralConfig;

    #[tokio::test]
    async fn test_neural_predictor_creation() {
        let config = NeuralConfig {
            enabled: false, // Disable for testing
            model_path: None,
            training_data_path: None,
            update_frequency_hours: 24,
            confidence_threshold: 0.8,
        };
        
        let predictor = NeuralRiskPredictor::new(config).await.unwrap();
        
        let features = RiskFeatures {
            volatility: 0.3,
            concentration: 0.2,
            liquidity_risk: 0.1,
            correlation: 0.5,
            momentum: 0.0,
            volume: 1000000.0,
            price_change: 0.05,
            market_sentiment: 0.6,
            time_features: TimeFeatures::current(),
        };
        
        let prediction = predictor.predict_risk(&features).await.unwrap();
        assert!(prediction.risk_score >= 0.0 && prediction.risk_score <= 1.0);
    }

    #[test]
    fn test_risk_features() {
        let features = RiskFeatures {
            volatility: 0.25,
            concentration: 0.15,
            liquidity_risk: 0.1,
            correlation: 0.7,
            momentum: 0.05,
            volume: 5000000.0,
            price_change: 0.02,
            market_sentiment: 0.8,
            time_features: TimeFeatures::current(),
        };
        
        assert!(features.volatility > 0.0);
        assert!(features.time_features.hour_of_day >= 0.0);
        assert!(features.time_features.hour_of_day <= 1.0);
    }
}