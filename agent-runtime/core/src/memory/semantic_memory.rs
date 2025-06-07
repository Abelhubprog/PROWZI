//! Semantic Memory Implementation
//!
//! Long-term knowledge storage with sophisticated clustering, graph relationships,
//! and knowledge distillation for autonomous trading intelligence.

use super::{
    MemoryId, MemoryContent, MemoryContext, MemoryError, RetrievedMemory, MemorySource,
    MemoryFeedback, SemanticMemoryStats, KnowledgeCluster, ConceptNode, MarketContext
};
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for semantic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    pub max_concepts: usize,
    pub max_clusters: usize,
    pub similarity_threshold: f32,
    pub knowledge_decay_days: i64,
    pub cluster_merge_threshold: f32,
    pub concept_activation_threshold: f32,
    pub max_concept_connections: usize,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            max_concepts: 100_000,
            max_clusters: 1_000,
            similarity_threshold: 0.85,
            knowledge_decay_days: 365,
            cluster_merge_threshold: 0.90,
            concept_activation_threshold: 0.3,
            max_concept_connections: 50,
        }
    }
}

/// Graph edge representing relationship between concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEdge {
    pub source_id: MemoryId,
    pub target_id: MemoryId,
    pub relationship_type: RelationshipType,
    pub strength: f32,
    pub created_at: DateTime<Utc>,
    pub last_activated: DateTime<Utc>,
    pub activation_count: u64,
}

/// Types of relationships between concepts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RelationshipType {
    Causal,
    Temporal,
    Similarity,
    Opposition,
    Dependency,
    Correlation,
    Category,
    Custom(String),
}

/// Knowledge distillation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDistillation {
    pub summary: String,
    pub key_insights: Vec<String>,
    pub confidence: f32,
    pub supporting_evidence: Vec<MemoryId>,
    pub contradictory_evidence: Vec<MemoryId>,
    pub created_at: DateTime<Utc>,
}

/// Semantic memory implementation with knowledge graphs and clustering
pub struct SemanticMemory {
    config: SemanticConfig,
    concepts: DashMap<MemoryId, ConceptNode>,
    clusters: DashMap<String, KnowledgeCluster>,
    concept_graph: Arc<RwLock<HashMap<MemoryId, Vec<ConceptEdge>>>>,
    embeddings: DashMap<MemoryId, Vec<f32>>,
    distilled_knowledge: DashMap<String, KnowledgeDistillation>,
    stats: SemanticMemoryStats,
    next_id: AtomicU64,
}

impl SemanticMemory {
    /// Create new semantic memory instance
    pub fn new(config: SemanticConfig) -> Self {
        Self {
            config,
            concepts: DashMap::new(),
            clusters: DashMap::new(),
            concept_graph: Arc::new(RwLock::new(HashMap::new())),
            embeddings: DashMap::new(),
            distilled_knowledge: DashMap::new(),
            stats: SemanticMemoryStats::default(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Store a concept in semantic memory
    pub async fn store_concept(
        &self,
        content: MemoryContent,
        context: MemoryContext,
        embedding: Vec<f32>,
    ) -> Result<MemoryId, MemoryError> {
        let concept_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let memory_id = MemoryId(format!("semantic_{}", concept_id));

        // Create concept node
        let concept = ConceptNode {
            id: memory_id.clone(),
            content: content.clone(),
            context: context.clone(),
            activation_level: 1.0,
            creation_time: Utc::now(),
            last_access: Utc::now(),
            access_count: 1,
            importance_score: self.calculate_importance(&content, &context).await?,
            tags: self.extract_tags(&content).await?,
            relationships: Vec::new(),
        };

        // Store concept and embedding
        self.concepts.insert(memory_id.clone(), concept);
        self.embeddings.insert(memory_id.clone(), embedding.clone());

        // Find similar concepts and create relationships
        self.create_semantic_relationships(&memory_id, &embedding).await?;

        // Update or create clusters
        self.update_clusters(&memory_id, &embedding).await?;

        // Update statistics
        self.stats.total_concepts.fetch_add(1, Ordering::SeqCst);
        self.stats.storage_operations.fetch_add(1, Ordering::SeqCst);

        debug!("Stored concept {} in semantic memory", memory_id.0);
        Ok(memory_id)
    }

    /// Retrieve concepts by semantic similarity
    pub async fn retrieve_similar(
        &self,
        query_embedding: &[f32],
        max_results: usize,
        min_similarity: f32,
    ) -> Result<Vec<RetrievedMemory>, MemoryError> {
        let mut similarities = Vec::new();

        // Calculate similarities with all stored concepts
        for entry in self.embeddings.iter() {
            let similarity = self.cosine_similarity(query_embedding, entry.value())?;
            if similarity >= min_similarity {
                similarities.push((entry.key().clone(), similarity));
            }
        }

        // Sort by similarity and take top results
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(max_results);

        let mut results = Vec::new();
        for (memory_id, similarity) in similarities {
            if let Some(concept) = self.concepts.get(&memory_id) {
                let retrieved = RetrievedMemory {
                    id: memory_id.clone(),
                    content: concept.content.clone(),
                    context: concept.context.clone(),
                    relevance_score: similarity,
                    retrieval_time: Utc::now(),
                    access_count: concept.access_count,
                    last_access: concept.last_access,
                    source: MemorySource::Semantic,
                };
                results.push(retrieved);

                // Update access statistics
                if let Some(mut concept) = self.concepts.get_mut(&memory_id) {
                    concept.last_access = Utc::now();
                    concept.access_count += 1;
                    concept.activation_level = (concept.activation_level * 0.9 + similarity * 0.1).min(1.0);
                }
            }
        }

        self.stats.retrieval_operations.fetch_add(1, Ordering::SeqCst);
        debug!("Retrieved {} similar concepts", results.len());
        Ok(results)
    }

    /// Retrieve concepts by cluster
    pub async fn retrieve_by_cluster(
        &self,
        cluster_name: &str,
        max_results: usize,
    ) -> Result<Vec<RetrievedMemory>, MemoryError> {
        let cluster = self.clusters.get(cluster_name)
            .ok_or_else(|| MemoryError::NotFound(format!("Cluster: {}", cluster_name)))?;

        let mut results = Vec::new();
        for concept_id in cluster.concept_ids.iter().take(max_results) {
            if let Some(concept) = self.concepts.get(concept_id) {
                let retrieved = RetrievedMemory {
                    id: concept_id.clone(),
                    content: concept.content.clone(),
                    context: concept.context.clone(),
                    relevance_score: concept.importance_score,
                    retrieval_time: Utc::now(),
                    access_count: concept.access_count,
                    last_access: concept.last_access,
                    source: MemorySource::Semantic,
                };
                results.push(retrieved);
            }
        }

        debug!("Retrieved {} concepts from cluster {}", results.len(), cluster_name);
        Ok(results)
    }

    /// Retrieve related concepts through graph traversal
    pub async fn retrieve_related(
        &self,
        concept_id: &MemoryId,
        relationship_types: &[RelationshipType],
        max_depth: usize,
        max_results: usize,
    ) -> Result<Vec<RetrievedMemory>, MemoryError> {
        let graph = self.concept_graph.read().await;
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut results = Vec::new();

        queue.push_back((concept_id.clone(), 0));
        visited.insert(concept_id.clone());

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth || results.len() >= max_results {
                break;
            }

            if let Some(edges) = graph.get(&current_id) {
                for edge in edges {
                    if relationship_types.is_empty() || relationship_types.contains(&edge.relationship_type) {
                        if !visited.contains(&edge.target_id) {
                            visited.insert(edge.target_id.clone());
                            queue.push_back((edge.target_id.clone(), depth + 1));

                            if let Some(concept) = self.concepts.get(&edge.target_id) {
                                let retrieved = RetrievedMemory {
                                    id: edge.target_id.clone(),
                                    content: concept.content.clone(),
                                    context: concept.context.clone(),
                                    relevance_score: edge.strength,
                                    retrieval_time: Utc::now(),
                                    access_count: concept.access_count,
                                    last_access: concept.last_access,
                                    source: MemorySource::Semantic,
                                };
                                results.push(retrieved);
                            }
                        }
                    }
                }
            }
        }

        debug!("Retrieved {} related concepts for {}", results.len(), concept_id.0);
        Ok(results)
    }

    /// Distill knowledge from a set of concepts
    pub async fn distill_knowledge(
        &self,
        concept_ids: &[MemoryId],
        domain: &str,
    ) -> Result<KnowledgeDistillation, MemoryError> {
        let mut concepts = Vec::new();
        for id in concept_ids {
            if let Some(concept) = self.concepts.get(id) {
                concepts.push(concept.clone());
            }
        }

        if concepts.is_empty() {
            return Err(MemoryError::InvalidOperation("No concepts found for distillation".to_string()));
        }

        // Analyze concepts and extract patterns
        let key_insights = self.extract_insights(&concepts).await?;
        let summary = self.generate_summary(&concepts, &key_insights).await?;
        let confidence = self.calculate_distillation_confidence(&concepts).await?;

        // Find supporting and contradictory evidence
        let (supporting, contradictory) = self.categorize_evidence(&concepts).await?;

        let distillation = KnowledgeDistillation {
            summary,
            key_insights,
            confidence,
            supporting_evidence: supporting,
            contradictory_evidence: contradictory,
            created_at: Utc::now(),
        };

        // Cache the distillation
        self.distilled_knowledge.insert(domain.to_string(), distillation.clone());

        info!("Distilled knowledge for domain: {}", domain);
        Ok(distillation)
    }

    /// Update concept relationships based on feedback
    pub async fn update_relationships(
        &self,
        feedback: &MemoryFeedback,
    ) -> Result<(), MemoryError> {
        let mut graph = self.concept_graph.write().await;

        match feedback {
            MemoryFeedback::Helpful { memory_id, context } => {
                // Strengthen relationships with concepts used in similar contexts
                self.strengthen_contextual_relationships(&mut graph, memory_id, context).await?;
            }
            MemoryFeedback::Irrelevant { memory_id, .. } => {
                // Weaken relationships that led to irrelevant retrievals
                self.weaken_relationships(&mut graph, memory_id, 0.1).await?;
            }
            MemoryFeedback::Outdated { memory_id, .. } => {
                // Mark concept for decay or removal
                if let Some(mut concept) = self.concepts.get_mut(memory_id) {
                    concept.importance_score *= 0.5;
                    concept.activation_level *= 0.5;
                }
            }
        }

        Ok(())
    }

    /// Perform maintenance operations
    pub async fn maintenance(&self) -> Result<(), MemoryError> {
        // Decay old concepts
        self.decay_concepts().await?;

        // Merge similar clusters
        self.merge_clusters().await?;

        // Prune weak relationships
        self.prune_relationships().await?;

        // Update statistics
        self.update_statistics().await?;

        info!("Completed semantic memory maintenance");
        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> SemanticMemoryStats {
        let mut stats = self.stats.clone();
        stats.concept_count = self.concepts.len();
        stats.cluster_count = self.clusters.len();
        stats.average_cluster_size = if stats.cluster_count > 0 {
            self.clusters.iter()
                .map(|cluster| cluster.concept_ids.len())
                .sum::<usize>() as f32 / stats.cluster_count as f32
        } else {
            0.0
        };
        stats
    }

    // Private helper methods

    async fn calculate_importance(&self, content: &MemoryContent, context: &MemoryContext) -> Result<f32, MemoryError> {
        let mut importance = 0.5; // Base importance

        // Factor in recency
        let age_hours = Utc::now().signed_duration_since(context.timestamp).num_hours();
        importance += (1.0 / (1.0 + age_hours as f32 / 24.0)) * 0.3;

        // Factor in content type and urgency
        match &context.mission_context {
            Some(mission) if mission.contains("urgent") => importance += 0.2,
            Some(mission) if mission.contains("critical") => importance += 0.3,
            _ => {}
        }

        Ok(importance.min(1.0))
    }

    async fn extract_tags(&self, content: &MemoryContent) -> Result<Vec<String>, MemoryError> {
        let mut tags = Vec::new();

        // Extract domain-specific tags
        match content {
            MemoryContent::Trading { action, .. } => {
                tags.push("trading".to_string());
                tags.push(format!("action_{:?}", action).to_lowercase());
            }
            MemoryContent::Market { symbol, .. } => {
                tags.push("market".to_string());
                tags.push(format!("symbol_{}", symbol));
            }
            MemoryContent::Analysis { category, .. } => {
                tags.push("analysis".to_string());
                tags.push(format!("category_{}", category));
            }
            MemoryContent::Alert { severity, .. } => {
                tags.push("alert".to_string());
                tags.push(format!("severity_{:?}", severity).to_lowercase());
            }
            _ => {
                tags.push("general".to_string());
            }
        }

        Ok(tags)
    }

    async fn create_semantic_relationships(
        &self,
        memory_id: &MemoryId,
        embedding: &[f32],
    ) -> Result<(), MemoryError> {
        let mut relationships = Vec::new();

        // Find similar concepts
        for entry in self.embeddings.iter() {
            if entry.key() == memory_id {
                continue;
            }

            let similarity = self.cosine_similarity(embedding, entry.value())?;
            if similarity > self.config.similarity_threshold {
                let relationship = ConceptEdge {
                    source_id: memory_id.clone(),
                    target_id: entry.key().clone(),
                    relationship_type: RelationshipType::Similarity,
                    strength: similarity,
                    created_at: Utc::now(),
                    last_activated: Utc::now(),
                    activation_count: 1,
                };
                relationships.push(relationship);

                if relationships.len() >= self.config.max_concept_connections {
                    break;
                }
            }
        }

        // Add relationships to graph
        let mut graph = self.concept_graph.write().await;
        graph.insert(memory_id.clone(), relationships);

        Ok(())
    }

    async fn update_clusters(&self, memory_id: &MemoryId, embedding: &[f32]) -> Result<(), MemoryError> {
        let mut best_cluster = None;
        let mut best_similarity = 0.0;

        // Find best matching cluster
        for cluster_entry in self.clusters.iter() {
            let similarity = self.cosine_similarity(embedding, &cluster_entry.centroid)?;
            if similarity > best_similarity && similarity > self.config.similarity_threshold {
                best_similarity = similarity;
                best_cluster = Some(cluster_entry.key().clone());
            }
        }

        match best_cluster {
            Some(cluster_name) => {
                // Add to existing cluster
                if let Some(mut cluster) = self.clusters.get_mut(&cluster_name) {
                    cluster.concept_ids.push(memory_id.clone());
                    cluster.size = cluster.concept_ids.len();
                    // Update centroid (simplified - should use proper centroid calculation)
                    for (i, val) in embedding.iter().enumerate() {
                        if i < cluster.centroid.len() {
                            cluster.centroid[i] = (cluster.centroid[i] * (cluster.size - 1) as f32 + val) / cluster.size as f32;
                        }
                    }
                }
            }
            None => {
                // Create new cluster
                let cluster_name = format!("cluster_{}", self.clusters.len());
                let cluster = KnowledgeCluster {
                    id: cluster_name.clone(),
                    concept_ids: vec![memory_id.clone()],
                    centroid: embedding.to_vec(),
                    size: 1,
                    coherence_score: 1.0,
                    created_at: Utc::now(),
                    last_updated: Utc::now(),
                };
                self.clusters.insert(cluster_name, cluster);
            }
        }

        Ok(())
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32, MemoryError> {
        if a.len() != b.len() {
            return Err(MemoryError::InvalidOperation("Vector dimension mismatch".to_string()));
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    async fn extract_insights(&self, concepts: &[ConceptNode]) -> Result<Vec<String>, MemoryError> {
        let mut insights = Vec::new();

        // Group concepts by tags and analyze patterns
        let mut tag_groups: HashMap<String, Vec<&ConceptNode>> = HashMap::new();
        for concept in concepts {
            for tag in &concept.tags {
                tag_groups.entry(tag.clone()).or_default().push(concept);
            }
        }

        // Generate insights from patterns
        for (tag, group_concepts) in tag_groups {
            if group_concepts.len() >= 3 {
                insights.push(format!("Significant activity in {} domain with {} related concepts", tag, group_concepts.len()));
            }
        }

        // Temporal patterns
        let recent_concepts = concepts.iter()
            .filter(|c| Utc::now().signed_duration_since(c.creation_time).num_hours() < 24)
            .count();

        if recent_concepts > concepts.len() / 2 {
            insights.push("High recent activity detected".to_string());
        }

        Ok(insights)
    }

    async fn generate_summary(&self, concepts: &[ConceptNode], insights: &[String]) -> Result<String, MemoryError> {
        let concept_count = concepts.len();
        let avg_importance = concepts.iter().map(|c| c.importance_score).sum::<f32>() / concept_count as f32;

        let mut summary = format!("Knowledge distillation from {} concepts (avg importance: {:.2}). ", concept_count, avg_importance);
        
        if !insights.is_empty() {
            summary.push_str("Key patterns: ");
            summary.push_str(&insights.join(", "));
            summary.push('.');
        }

        Ok(summary)
    }

    async fn calculate_distillation_confidence(&self, concepts: &[ConceptNode]) -> Result<f32, MemoryError> {
        let avg_importance = concepts.iter().map(|c| c.importance_score).sum::<f32>() / concepts.len() as f32;
        let avg_access_count = concepts.iter().map(|c| c.access_count as f32).sum::<f32>() / concepts.len() as f32;
        
        // Confidence based on importance and usage
        let confidence = (avg_importance * 0.6 + (avg_access_count / 10.0).min(1.0) * 0.4).min(1.0);
        Ok(confidence)
    }

    async fn categorize_evidence(&self, concepts: &[ConceptNode]) -> Result<(Vec<MemoryId>, Vec<MemoryId>), MemoryError> {
        let mut supporting = Vec::new();
        let mut contradictory = Vec::new();

        // Simple heuristic: high importance = supporting, low = contradictory
        for concept in concepts {
            if concept.importance_score > 0.7 {
                supporting.push(concept.id.clone());
            } else if concept.importance_score < 0.3 {
                contradictory.push(concept.id.clone());
            }
        }

        Ok((supporting, contradictory))
    }

    async fn strengthen_contextual_relationships(
        &self,
        graph: &mut HashMap<MemoryId, Vec<ConceptEdge>>,
        memory_id: &MemoryId,
        _context: &MemoryContext,
    ) -> Result<(), MemoryError> {
        if let Some(edges) = graph.get_mut(memory_id) {
            for edge in edges.iter_mut() {
                edge.strength = (edge.strength * 1.1).min(1.0);
                edge.last_activated = Utc::now();
                edge.activation_count += 1;
            }
        }
        Ok(())
    }

    async fn weaken_relationships(
        &self,
        graph: &mut HashMap<MemoryId, Vec<ConceptEdge>>,
        memory_id: &MemoryId,
        factor: f32,
    ) -> Result<(), MemoryError> {
        if let Some(edges) = graph.get_mut(memory_id) {
            for edge in edges.iter_mut() {
                edge.strength *= 1.0 - factor;
            }
        }
        Ok(())
    }

    async fn decay_concepts(&self) -> Result<(), MemoryError> {
        let cutoff = Utc::now() - Duration::days(self.config.knowledge_decay_days);
        let mut to_remove = Vec::new();

        for entry in self.concepts.iter() {
            if entry.last_access < cutoff && entry.importance_score < 0.1 {
                to_remove.push(entry.key().clone());
            }
        }

        for id in to_remove {
            self.concepts.remove(&id);
            self.embeddings.remove(&id);
        }

        Ok(())
    }

    async fn merge_clusters(&self) -> Result<(), MemoryError> {
        let cluster_pairs: Vec<_> = self.clusters.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        for i in 0..cluster_pairs.len() {
            for j in i + 1..cluster_pairs.len() {
                let similarity = self.cosine_similarity(&cluster_pairs[i].1.centroid, &cluster_pairs[j].1.centroid)?;
                if similarity > self.config.cluster_merge_threshold {
                    // Merge clusters (simplified implementation)
                    debug!("Would merge clusters {} and {}", cluster_pairs[i].0, cluster_pairs[j].0);
                }
            }
        }

        Ok(())
    }

    async fn prune_relationships(&self) -> Result<(), MemoryError> {
        let mut graph = self.concept_graph.write().await;
        
        for edges in graph.values_mut() {
            edges.retain(|edge| edge.strength > 0.1);
        }

        Ok(())
    }

    async fn update_statistics(&self) -> Result<(), MemoryError> {
        // Update concept and cluster counts
        self.stats.concept_count = self.concepts.len();
        self.stats.cluster_count = self.clusters.len();
        
        // Calculate average cluster size
        if self.stats.cluster_count > 0 {
            let total_concepts_in_clusters: usize = self.clusters.iter()
                .map(|cluster| cluster.concept_ids.len())
                .sum();
            self.stats.average_cluster_size = total_concepts_in_clusters as f32 / self.stats.cluster_count as f32;
        }

        Ok(())
    }
}