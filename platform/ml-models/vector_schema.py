"""
Vector schema definitions for Prowzi ML models and embeddings.
Defines the structure and metadata for vector databases and similarity search.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import json

class VectorType(Enum):
    """Types of vectors used in the system."""
    TRANSACTION_EMBEDDING = "transaction_embedding"
    MARKET_SENTIMENT = "market_sentiment"
    CODE_ANALYSIS = "code_analysis"
    AGENT_BEHAVIOR = "agent_behavior"
    RISK_FEATURES = "risk_features"
    USER_PREFERENCES = "user_preferences"
    MISSION_CONTEXT = "mission_context"

class EmbeddingModel(Enum):
    """Supported embedding models."""
    OPENAI_ADA_002 = "text-embedding-ada-002"
    SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"
    COHERE_EMBED = "embed-english-v2.0"
    CUSTOM_FINANCIAL = "prowzi-financial-v1"
    CUSTOM_CRYPTO = "prowzi-crypto-v1"

@dataclass
class VectorMetadata:
    """Metadata associated with a vector."""
    vector_id: str
    vector_type: VectorType
    embedding_model: EmbeddingModel
    created_at: datetime
    updated_at: Optional[datetime] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransactionVector:
    """Vector representation of blockchain transactions."""
    vector: np.ndarray
    metadata: VectorMetadata
    
    # Transaction-specific fields
    transaction_hash: str
    blockchain: str
    from_address: str
    to_address: str
    value: float
    gas_fee: float
    timestamp: datetime
    
    # Derived features
    transaction_type: str  # transfer, swap, mint, burn, etc.
    risk_score: float
    anomaly_score: float
    cluster_id: Optional[str] = None

@dataclass
class MarketSentimentVector:
    """Vector representation of market sentiment data."""
    vector: np.ndarray
    metadata: VectorMetadata
    
    # Sentiment-specific fields
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float
    source_type: str  # twitter, reddit, news, etc.
    source_url: Optional[str] = None
    text_content: Optional[str] = None
    
    # Market context
    price_at_time: float
    volume_24h: float
    market_cap: float

@dataclass
class CodeAnalysisVector:
    """Vector representation of smart contract or code analysis."""
    vector: np.ndarray
    metadata: VectorMetadata
    
    # Code-specific fields
    contract_address: Optional[str] = None
    repository_url: Optional[str] = None
    code_hash: str = ""
    language: str = ""
    
    # Analysis results
    vulnerability_score: float = 0.0
    complexity_score: float = 0.0
    quality_score: float = 0.0
    security_issues: List[str] = field(default_factory=list)
    gas_optimization_score: float = 0.0

@dataclass
class AgentBehaviorVector:
    """Vector representation of agent behavior patterns."""
    vector: np.ndarray
    metadata: VectorMetadata
    
    # Agent-specific fields
    agent_id: str
    agent_type: str
    behavior_pattern: str
    
    # Performance metrics
    success_rate: float
    average_response_time: float
    resource_efficiency: float
    
    # Behavioral features
    decision_consistency: float
    adaptation_rate: float
    collaboration_score: float

@dataclass
class RiskFeatureVector:
    """Vector representation of risk assessment features."""
    vector: np.ndarray
    metadata: VectorMetadata
    
    # Risk-specific fields
    entity_id: str  # transaction, address, contract, etc.
    entity_type: str
    risk_category: str
    
    # Risk scores
    overall_risk: float
    liquidity_risk: float
    volatility_risk: float
    counterparty_risk: float
    regulatory_risk: float
    
    # Contributing factors
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    vector_id: str
    similarity_score: float
    distance: float
    metadata: VectorMetadata
    vector_data: Union[TransactionVector, MarketSentimentVector, CodeAnalysisVector, 
                      AgentBehaviorVector, RiskFeatureVector]

class VectorSchema:
    """Schema definition for vector operations."""
    
    # Standard dimensions for different vector types
    DIMENSIONS = {
        VectorType.TRANSACTION_EMBEDDING: 1536,  # OpenAI ada-002 size
        VectorType.MARKET_SENTIMENT: 512,
        VectorType.CODE_ANALYSIS: 768,
        VectorType.AGENT_BEHAVIOR: 256,
        VectorType.RISK_FEATURES: 384,
        VectorType.USER_PREFERENCES: 128,
        VectorType.MISSION_CONTEXT: 512,
    }
    
    # Required metadata fields for each vector type
    REQUIRED_METADATA = {
        VectorType.TRANSACTION_EMBEDDING: [
            'transaction_hash', 'blockchain', 'from_address', 'to_address'
        ],
        VectorType.MARKET_SENTIMENT: [
            'symbol', 'sentiment_score', 'source_type'
        ],
        VectorType.CODE_ANALYSIS: [
            'code_hash', 'language'
        ],
        VectorType.AGENT_BEHAVIOR: [
            'agent_id', 'agent_type', 'behavior_pattern'
        ],
        VectorType.RISK_FEATURES: [
            'entity_id', 'entity_type', 'risk_category'
        ],
    }
    
    @classmethod
    def get_dimension(cls, vector_type: VectorType) -> int:
        """Get the expected dimension for a vector type."""
        return cls.DIMENSIONS.get(vector_type, 512)
    
    @classmethod
    def validate_vector(cls, vector: np.ndarray, vector_type: VectorType) -> bool:
        """Validate that a vector matches the expected schema."""
        expected_dim = cls.get_dimension(vector_type)
        return vector.shape == (expected_dim,) and vector.dtype == np.float32
    
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any], vector_type: VectorType) -> bool:
        """Validate that metadata contains required fields."""
        required_fields = cls.REQUIRED_METADATA.get(vector_type, [])
        return all(field in metadata for field in required_fields)
    
    @classmethod
    def create_index_config(cls, vector_type: VectorType) -> Dict[str, Any]:
        """Create index configuration for a vector type."""
        return {
            "dimension": cls.get_dimension(vector_type),
            "metric": "cosine",  # Default similarity metric
            "pod_type": "p1",
            "metadata_config": {
                "indexed": cls._get_indexed_metadata_fields(vector_type)
            }
        }
    
    @classmethod
    def _get_indexed_metadata_fields(cls, vector_type: VectorType) -> List[str]:
        """Get metadata fields that should be indexed for fast filtering."""
        common_fields = ["vector_type", "created_at", "source", "tags"]
        
        type_specific = {
            VectorType.TRANSACTION_EMBEDDING: [
                "blockchain", "transaction_type", "risk_score"
            ],
            VectorType.MARKET_SENTIMENT: [
                "symbol", "sentiment_score", "source_type"
            ],
            VectorType.CODE_ANALYSIS: [
                "language", "vulnerability_score", "security_issues"
            ],
            VectorType.AGENT_BEHAVIOR: [
                "agent_id", "agent_type", "success_rate"
            ],
            VectorType.RISK_FEATURES: [
                "entity_type", "risk_category", "overall_risk"
            ],
        }
        
        return common_fields + type_specific.get(vector_type, [])

class VectorDatabase:
    """Abstract interface for vector database operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def upsert_vector(self, vector_id: str, vector: np.ndarray, 
                           metadata: Dict[str, Any]) -> bool:
        """Insert or update a vector with metadata."""
        raise NotImplementedError
    
    async def search_similar(self, query_vector: np.ndarray, 
                           vector_type: VectorType,
                           top_k: int = 10,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        raise NotImplementedError
    
    async def get_vector(self, vector_id: str) -> Optional[VectorSearchResult]:
        """Retrieve a specific vector by ID."""
        raise NotImplementedError
    
    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        raise NotImplementedError
    
    async def create_index(self, index_name: str, vector_type: VectorType) -> bool:
        """Create an index for a vector type."""
        raise NotImplementedError

class PineconeVectorDB(VectorDatabase):
    """Pinecone implementation of vector database."""
    
    def __init__(self, api_key: str, environment: str):
        import pinecone
        super().__init__(f"pinecone://{environment}")
        pinecone.init(api_key=api_key, environment=environment)
        self.pinecone = pinecone
    
    async def upsert_vector(self, vector_id: str, vector: np.ndarray, 
                           metadata: Dict[str, Any]) -> bool:
        """Upsert vector to Pinecone."""
        try:
            index = self.pinecone.Index(metadata.get('index_name', 'default'))
            index.upsert([(vector_id, vector.tolist(), metadata)])
            return True
        except Exception as e:
            print(f"Error upserting vector: {e}")
            return False
    
    async def search_similar(self, query_vector: np.ndarray, 
                           vector_type: VectorType,
                           top_k: int = 10,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors in Pinecone."""
        try:
            index_name = f"prowzi-{vector_type.value}"
            index = self.pinecone.Index(index_name)
            
            results = index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                filter=filter_metadata,
                include_metadata=True
            )
            
            search_results = []
            for match in results['matches']:
                search_results.append(VectorSearchResult(
                    vector_id=match['id'],
                    similarity_score=match['score'],
                    distance=1 - match['score'],  # Convert similarity to distance
                    metadata=VectorMetadata(**match['metadata']),
                    vector_data=None  # Would need to reconstruct from metadata
                ))
            
            return search_results
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []

class WeaviateVectorDB(VectorDatabase):
    """Weaviate implementation of vector database."""
    
    def __init__(self, url: str, auth_config: Optional[Dict] = None):
        import weaviate
        super().__init__(url)
        self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
    
    async def upsert_vector(self, vector_id: str, vector: np.ndarray, 
                           metadata: Dict[str, Any]) -> bool:
        """Upsert vector to Weaviate."""
        try:
            class_name = metadata.get('class_name', 'ProwziVector')
            
            data_object = {
                **metadata,
                'vector_id': vector_id
            }
            
            self.client.data_object.create(
                data_object,
                class_name,
                uuid=vector_id,
                vector=vector.tolist()
            )
            return True
        except Exception as e:
            print(f"Error upserting vector: {e}")
            return False
    
    async def search_similar(self, query_vector: np.ndarray, 
                           vector_type: VectorType,
                           top_k: int = 10,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors in Weaviate."""
        try:
            class_name = f"Prowzi{vector_type.value.title().replace('_', '')}"
            
            query = (
                self.client.query
                .get(class_name, ["vector_id", "_additional {certainty}"])
                .with_near_vector({"vector": query_vector.tolist()})
                .with_limit(top_k)
            )
            
            if filter_metadata:
                where_filter = self._build_where_filter(filter_metadata)
                query = query.with_where(where_filter)
            
            results = query.do()
            
            search_results = []
            for item in results['data']['Get'][class_name]:
                search_results.append(VectorSearchResult(
                    vector_id=item['vector_id'],
                    similarity_score=item['_additional']['certainty'],
                    distance=1 - item['_additional']['certainty'],
                    metadata=VectorMetadata(**item),
                    vector_data=None
                ))
            
            return search_results
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def _build_where_filter(self, filter_metadata: Dict[str, Any]) -> Dict:
        """Build Weaviate where filter from metadata."""
        # Simplified filter building - would need more sophisticated logic
        conditions = []
        for key, value in filter_metadata.items():
            conditions.append({
                "path": [key],
                "operator": "Equal",
                "valueString" if isinstance(value, str) else "valueNumber": value
            })
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {
                "operator": "And",
                "operands": conditions
            }

# Utility functions for vector operations
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

def create_embedding_from_text(text: str, model: EmbeddingModel) -> np.ndarray:
    """Create embedding from text using specified model."""
    # This would integrate with actual embedding services
    # For now, return a placeholder
    dimension = VectorSchema.DIMENSIONS.get(VectorType.TRANSACTION_EMBEDDING, 512)
    return np.random.random(dimension).astype(np.float32)

# Export main classes and functions
__all__ = [
    'VectorType', 'EmbeddingModel', 'VectorMetadata',
    'TransactionVector', 'MarketSentimentVector', 'CodeAnalysisVector',
    'AgentBehaviorVector', 'RiskFeatureVector', 'VectorSearchResult',
    'VectorSchema', 'VectorDatabase', 'PineconeVectorDB', 'WeaviateVectorDB',
    'normalize_vector', 'cosine_similarity', 'euclidean_distance', 'create_embedding_from_text'
]