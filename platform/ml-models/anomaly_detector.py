# File: platform/ml-models/anomaly_detector.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import asyncio
import aioredis
import json
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnomalyScore:
    """Detailed anomaly detection result"""
    timestamp: datetime
    score: float
    confidence: float
    anomaly_type: str
    severity: str  # low, medium, high, critical
    features: Dict[str, float]
    explanation: str
    related_addresses: List[str]
    recommended_actions: List[str]
    
class TransformerAutoencoder(nn.Module):
    """Advanced Transformer-based autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Anomaly scoring head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x_proj = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        x_pos = self.pos_encoder(x_proj.transpose(0, 1))  # (seq_len, batch, hidden_dim)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x_pos)  # (seq_len, batch, hidden_dim)
        
        # Attention pooling for anomaly detection
        attn_output, attn_weights = self.attention_pool(
            encoded.mean(0).unsqueeze(0),  # Query
            encoded,  # Key
            encoded   # Value
        )
        
        # Decode
        decoded = self.decoder(encoded.transpose(0, 1))  # (batch, seq_len, input_dim)
        
        # Anomaly score
        anomaly_score = self.anomaly_head(attn_output.squeeze(0))  # (batch, 1)
        
        return decoded, anomaly_score, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class IsolationForestGPU:
    """GPU-accelerated Isolation Forest for real-time anomaly detection"""
    
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, 
                 contamination: float = 0.1, device: str = 'cuda'):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trees = []
        self.threshold = None
        
    def fit(self, X: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples = X_tensor.shape[0]
        
        for _ in range(self.n_estimators):
            # Sample subset
            sample_idx = torch.randperm(n_samples)[:self.max_samples]
            X_subset = X_tensor[sample_idx]
            
            # Build isolation tree
            tree = self._build_tree(X_subset, 0, self._get_max_depth(self.max_samples))
            self.trees.append(tree)
        
        # Calculate threshold
        scores = self.decision_function(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
    def _build_tree(self, X: torch.Tensor, depth: int, max_depth: int) -> Dict:
        n_samples, n_features = X.shape
        
        if depth >= max_depth or n_samples <= 1:
            return {'n_samples': n_samples, 'is_leaf': True}
        
        # Random feature and split
        feature = torch.randint(0, n_features, (1,)).item()
        min_val = X[:, feature].min().item()
        max_val = X[:, feature].max().item()
        
        if min_val == max_val:
            return {'n_samples': n_samples, 'is_leaf': True}
        
        split_val = torch.rand(1).item() * (max_val - min_val) + min_val
        
        left_mask = X[:, feature] < split_val
        right_mask = ~left_mask
        
        return {
            'is_leaf': False,
            'feature': feature,
            'threshold': split_val,
            'left': self._build_tree(X[left_mask], depth + 1, max_depth),
            'right': self._build_tree(X[right_mask], depth + 1, max_depth)
        }
    
    def _get_max_depth(self, n_samples: int) -> int:
        return int(np.ceil(np.log2(n_samples)))
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        scores = []
        
        for i in range(X_tensor.shape[0]):
            path_lengths = []
            for tree in self.trees:
                path_length = self._path_length(X_tensor[i], tree, 0)
                path_lengths.append(path_length)
            
            avg_path_length = np.mean(path_lengths)
            scores.append(avg_path_length)
        
        return np.array(scores)
    
    def _path_length(self, x: torch.Tensor, tree: Dict, current_depth: int) -> float:
        if tree['is_leaf']:
            n = tree['n_samples']
            if n <= 1:
                return current_depth
            else:
                return current_depth + self._c(n)
        
        if x[tree['feature']] < tree['threshold']:
            return self._path_length(x, tree['left'], current_depth + 1)
        else:
            return self._path_length(x, tree['right'], current_depth + 1)
    
    def _c(self, n: int) -> float:
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return (scores < self.threshold).astype(int)

class CryptoAnomalyDetector:
    """Main anomaly detection system for crypto transactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.transformer_model = TransformerAutoencoder(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        self.isolation_forest = IsolationForestGPU(
            n_estimators=config['n_estimators'],
            contamination=config['contamination']
        )
        
        # Feature engineering
        self.scaler = StandardScaler()
        self.feature_window = deque(maxlen=config['feature_window_size'])
        
        # Redis for real-time updates
        self.redis_client = None
        
        # Thresholds
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
        
    async def initialize(self):
        """Initialize connections and load models"""
        self.redis_client = await aioredis.create_redis_pool('redis://localhost')
        
        # Load pre-trained models if available
        try:
            checkpoint = torch.load(self.config['model_path'])
            self.transformer_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded pre-trained transformer model")
        except:
            logger.info("No pre-trained model found, using random initialization")
    
    def extract_features(self, transaction_data: Dict) -> np.ndarray:
        """Extract advanced features from transaction data"""
        features = []
        
        # Basic transaction features
        features.extend([
            transaction_data.get('amount', 0),
            transaction_data.get('gas_fee', 0),
            transaction_data.get('nonce', 0),
            len(transaction_data.get('input_data', '')),
        ])
        
        # Time-based features
        timestamp = datetime.fromisoformat(transaction_data['timestamp'])
        features.extend([
            timestamp.hour,
            timestamp.weekday(),
            timestamp.day,
            int(timestamp.timestamp())
        ])
        
        # Address features
        from_addr = transaction_data.get('from_address', '')
        to_addr = transaction_data.get('to_address', '')
        
        features.extend([
            self._address_activity_score(from_addr),
            self._address_activity_score(to_addr),
            self._is_contract(to_addr),
            self._is_known_exchange(from_addr),
            self._is_known_exchange(to_addr),
        ])
        
        # Network features
        features.extend([
            transaction_data.get('block_confirmations', 0),
            transaction_data.get('mempool_time', 0),
            self._get_network_congestion(),
        ])
        
        # Historical patterns
        if len(self.feature_window) > 0:
            recent_amounts = [tx['amount'] for tx in self.feature_window]
            features.extend([
                np.mean(recent_amounts),
                np.std(recent_amounts),
                np.max(recent_amounts),
                transaction_data['amount'] / (np.mean(recent_amounts) + 1e-8)
            ])
        else:
            features.extend([0, 0, 0, 1])
        
        return np.array(features)
    
    def _address_activity_score(self, address: str) -> float:
        """Calculate activity score for an address"""
        # In production, query from database
        # Placeholder implementation
        return np.random.random()
    
    def _is_contract(self, address: str) -> float:
        """Check if address is a smart contract"""
        # Placeholder
        return 1.0 if len(address) > 40 else 0.0
    
    def _is_known_exchange(self, address: str) -> float:
        """Check if address belongs to known exchange"""
        known_exchanges = {
            '0xexchange1': 1.0,
            '0xexchange2': 1.0,
            # Add more
        }
        return known_exchanges.get(address, 0.0)
    
    def _get_network_congestion(self) -> float:
        """Get current network congestion level"""
        # Placeholder - in production, query from network
        return np.random.random()
    
    async def detect_anomaly(self, transaction_data: Dict) -> AnomalyScore:
        """Main anomaly detection pipeline"""
        # Extract features
        features = self.extract_features(transaction_data)
        
        # Add to feature window
        self.feature_window.append(transaction_data)
        
        # Prepare sequence for transformer
        if len(self.feature_window) >= 10:
            sequence = np.array([self.extract_features(tx) for tx in self.feature_window])
            sequence = self.scaler.fit_transform(sequence)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Transformer prediction
            with torch.no_grad():
                reconstructed, anomaly_score, attention = self.transformer_model(sequence_tensor)
                transformer_score = anomaly_score.cpu().numpy()[0, 0]
                reconstruction_error = F.mse_loss(reconstructed, sequence_tensor).cpu().numpy()
        else:
            transformer_score = 0.5
            reconstruction_error = 0.0
        
        # Isolation Forest prediction
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        isolation_score = self.isolation_forest.decision_function(features_scaled)[0]
        
        # Combine scores
        final_score = 0.6 * transformer_score + 0.4 * (1 - isolation_score / 100)
        
        # Determine severity
        severity = self._get_severity(final_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            transaction_data, features, final_score, transformer_score, isolation_score
        )
        
        # Get related addresses
        related_addresses = await self._find_related_addresses(transaction_data)
        
        # Recommended actions
        actions = self._get_recommended_actions(severity, transaction_data)
        
        return AnomalyScore(
            timestamp=datetime.now(),
            score=final_score,
            confidence=0.85 if len(self.feature_window) >= 10 else 0.5,
            anomaly_type=self._classify_anomaly_type(features, final_score),
            severity=severity,
            features={
                'transformer_score': transformer_score,
                'isolation_score': isolation_score,
                'reconstruction_error': float(reconstruction_error),
                'amount_ratio': features[-1] if len(self.feature_window) > 0 else 1.0
            },
            explanation=explanation,
            related_addresses=related_addresses,
            recommended_actions=actions
        )
    
    def _get_severity(self, score: float) -> str:
        """Determine anomaly severity"""
        for severity, threshold in sorted(self.severity_thresholds.items(), 
                                        key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return severity
        return 'low'
    
    def _generate_explanation(self, transaction_data: Dict, features: np.ndarray, 
                            final_score: float, transformer_score: float, 
                            isolation_score: float) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        if final_score > 0.7:
            explanations.append(f"High anomaly score of {final_score:.2f} detected")
        
        # Amount anomaly
        if len(self.feature_window) > 0 and features[-1] > 10:
            explanations.append(f"Transaction amount is {features[-1]:.1f}x higher than recent average")
        
        # Time anomaly
        hour = int(features[4])
        if hour < 6 or hour > 22:
            explanations.append(f"Unusual transaction time: {hour}:00")
        
        # Network anomaly
        if transformer_score > 0.8:
            explanations.append("Transformer model detected unusual pattern in transaction sequence")
        
        if isolation_score < 10:
            explanations.append("Transaction is isolated from normal behavior clusters")
        
        return "; ".join(explanations) if explanations else "Normal transaction pattern"
    
    async def _find_related_addresses(self, transaction_data: Dict) -> List[str]:
        """Find addresses related to the anomalous transaction"""
        related = []
        
        # In production, query graph database for connected addresses
        # Placeholder implementation
        if transaction_data.get('from_address'):
            related.append(transaction_data['from_address'])
        if transaction_data.get('to_address'):
            related.append(transaction_data['to_address'])
        
        return related
    
    def _classify_anomaly_type(self, features: np.ndarray, score: float) -> str:
        """Classify the type of anomaly"""
        if features[0] > 1000000:  # Large amount
            return "whale_movement"
        elif features[-1] > 20:  # Sudden spike
            return "sudden_spike"
        elif features[4] < 4 or features[4] > 23:  # Odd hours
            return "timing_anomaly"
        elif score > 0.8:
            return "pattern_anomaly"
        else:
            return "general_anomaly"
    
    def _get_recommended_actions(self, severity: str, transaction_data: Dict) -> List[str]:
        """Generate recommended actions based on anomaly"""
        actions = []
        
        if severity == 'critical':
            actions.extend([
                "Immediate investigation required",
                "Consider freezing related addresses",
                "Alert security team"
            ])
        elif severity == 'high':
            actions.extend([
                "Monitor related addresses closely",
                "Review transaction history",
                "Check for similar patterns"
            ])
        elif severity == 'medium':
            actions.extend([
                "Add to watchlist",
                "Analyze transaction context"
            ])
        else:
            actions.append("Continue monitoring")
        
        return actions
    
    async def train_online(self, transaction_batch: List[Dict], labels: Optional[List[int]] = None):
        """Online training for continuous improvement"""
        # Extract features for batch
        features_batch = np.array([self.extract_features(tx) for tx in transaction_batch])
        features_scaled = self.scaler.fit_transform(features_batch)
        
        # Update Isolation Forest
        self.isolation_forest.fit(features_scaled)
        
        # Update Transformer if labels provided
        if labels is not None and len(self.feature_window) >= 10:
            sequences = []
            targets = []
            
            for i in range(len(transaction_batch) - 10):
                seq = features_scaled[i:i+10]
                sequences.append(seq)
                targets.append(labels[i+9])
            
            if sequences:
                self._train_transformer(sequences, targets)
    
    def _train_transformer(self, sequences: List[np.ndarray], labels: List[int]):
        """Train transformer model"""
        dataset = TensorDataset(
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        self.transformer_model.train()
        for epoch in range(10):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                _, anomaly_scores, _ = self.transformer_model(batch_x)
                loss = criterion(anomaly_scores.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
        
        self.transformer_model.eval()
    
    async def save_model(self, path: str):
        """Save trained models"""
        torch.save({
            'model_state_dict': self.transformer_model.state_dict(),
            'scaler': self.scaler,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

# Example usage
async def main():
    config = {
        'input_dim': 20,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'n_estimators': 100,
        'contamination': 0.1,
        'feature_window_size': 100,
        'model_path': 'models/anomaly_detector.pth'
    }
    
    detector = CryptoAnomalyDetector(config)
    await detector.initialize()
    
    # Example transaction
    transaction = {
        'amount': 1000000,
        'from_address': '0xabc123',
        'to_address': '0xdef456',
        'timestamp': datetime.now().isoformat(),
        'gas_fee': 50,
        'nonce': 10,
        'input_data': '0x',
        'block_confirmations': 12,
        'mempool_time': 30
    }
    
    anomaly_score = await detector.detect_anomaly(transaction)
    print(f"Anomaly Score: {anomaly_score.score:.3f}")
    print(f"Severity: {anomaly_score.severity}")
    print(f"Explanation: {anomaly_score.explanation}")
    print(f"Recommended Actions: {', '.join(anomaly_score.recommended_actions)}")

if __name__ == "__main__":
    asyncio.run(main())