import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from sklearn.preprocessing import StandardScaler


@dataclass
class RugPrediction:
    probability: float
    risk_factors: Dict[str, float]
    explanation: str
    confidence: float


class AdvancedRugDetector(nn.Module):

    def __init__(self,
                 feature_dim: int = 128,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3):
        super().__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.BatchNorm1d(hidden_dims[0]))

        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dims[0],
                                               num_heads=8,
                                               dropout=dropout_rate)

        # Classification head
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dims[i + 1])
            ])

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.classifier = nn.Sequential(*layers)

        # Explanation head
        self.explainer = nn.Linear(hidden_dims[-1], feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        features = self.feature_extractor(x)

        # Apply attention
        attended_features, attention_weights = self.attention(
            features.unsqueeze(0), features.unsqueeze(0),
            features.unsqueeze(0))
        attended_features = attended_features.squeeze(0)

        # Classification
        logits = self.classifier(attended_features)

        # Feature importance
        importance = self.explainer(attended_features)

        return torch.sigmoid(logits), importance


class RugDetectorPipeline:

    def __init__(self, model_path: str, scaler_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Feature names for explainability
        self.feature_names = [
            'creator_liquidity_share',
            'liquidity_lock_days',
            'contract_upgradeable',
            'deployer_age_days',
            'deployer_previous_rugs',
            'initial_liquidity_usd',
            'holder_concentration',
            'metadata_completeness',
            'audit_status',
            'social_presence_score',
            # ... more features
        ]

    def extract_features(self, token_data: Dict) -> np.ndarray:
        """Extract features from raw token data"""
        features = []

        # Liquidity metrics
        features.append(token_data.get('creator_liquidity_share', 0))
        features.append(token_data.get('liquidity_lock_days', 0))

        # Contract analysis
        features.append(1 if token_data.get('upgradeable', False) else 0)
        features.append(token_data.get('deployer_age_days', 0))

        # Historical analysis
        deployer_history = self.analyze_deployer_history(
            token_data.get('deployer_address'))
        features.append(deployer_history.get('previous_rugs', 0))

        # Market metrics
        features.append(token_data.get('initial_liquidity_usd', 0))
        features.append(
            self.calculate_holder_concentration(token_data.get('holders', [])))

        # Social signals
        features.append(self.calculate_metadata_score(token_data))
        features.append(1 if token_data.get('audit_report') else 0)
        features.append(self.calculate_social_score(token_data))

        # Technical indicators
        features.extend(self.extract_technical_features(token_data))

        return np.array(features)

    def predict(self, token_data: Dict) -> RugPrediction:
        """Make prediction with explanation"""
        # Extract and scale features
        raw_features = self.extract_features(token_data)
        scaled_features = self.scaler.transform([raw_features])

        # Convert to tensor
        x = torch.FloatTensor(scaled_features).to(self.device)

        # Get prediction and importance
        with torch.no_grad():
            prob, importance = self.model(x)

        prob = prob.cpu().numpy()[0, 0]
        importance = importance.cpu().numpy()[0]

        # Generate risk factors
        risk_factors = {}
        for i, (name, imp) in enumerate(zip(self.feature_names, importance)):
            if abs(imp) > 0.1:  # Significant factors only
                risk_factors[name] = float(imp)

        # Generate explanation
        explanation = self.generate_explanation(prob, risk_factors,
                                                raw_features)

        # Calculate confidence based on feature completeness
        confidence = self.calculate_confidence(token_data)

        return RugPrediction(probability=float(prob),
                             risk_factors=risk_factors,
                             explanation=explanation,
                             confidence=confidence)

    def generate_explanation(self, prob: float, risk_factors: Dict[str, float],
                             features: np.ndarray) -> str:
        """Generate human-readable explanation"""
        if prob > 0.8:
            level = "CRITICAL"
            desc = "Extremely high risk of rug pull"
        elif prob > 0.6:
            level = "HIGH"
            desc = "Significant rug pull indicators detected"
        elif prob > 0.4:
            level = "MODERATE"
            desc = "Some concerning patterns identified"
        else:
            level = "LOW"
            desc = "No major red flags detected"

        # Top risk factors
        top_risks = sorted(risk_factors.items(),
                           key=lambda x: abs(x[1]),
                           reverse=True)[:3]

        factors_text = []
        for factor, importance in top_risks:
            if importance > 0:
                factors_text.append(
                    f"• {factor.replace('_', ' ').title()}: High risk")
            else:
                factors_text.append(
                    f"• {factor.replace('_', ' ').title()}: Protective")

        explanation = f"""
{level} RISK ({prob:.1%}): {desc}

Key factors:
{chr(10).join(factors_text)}
        """.strip()

        return explanation

    def update_model(self, feedback_data: List[Dict]):
        """Online learning from user feedback"""
        # Prepare training data from feedback
        X, y = [], []
        for feedback in feedback_data:
            features = self.extract_features(feedback['token_data'])
            label = 1 if feedback['was_rug'] else 0
            X.append(features)
            y.append(label)

        X = self.scaler.transform(X)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # Fine-tune model
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.BCELoss()

        for epoch in range(10):
            optimizer.zero_grad()
            outputs, _ = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        self.model.eval()
