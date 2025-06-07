use prowzi_messages::{EnrichedEvent, EVIScores, Domain};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

pub struct EVICalculator {
    weights: EVIWeights,
    vector_client: VectorDbClient,
    knowledge_graph: KnowledgeGraphClient,
}

impl EVICalculator {
    pub async fn calculate(&self, event: &EnrichedEvent) -> EVIScores {
        // Parallel calculation of all scores
        let (freshness, novelty, impact, confidence, gap) = tokio::join!(
            self.calculate_freshness(event),
            self.calculate_novelty(event),
            self.calculate_impact(event),
            self.calculate_confidence(event),
            self.calculate_gap(event),
        );

        EVIScores {
            freshness,
            novelty,
            impact,
            confidence,
            gap,
        }
    }

    async fn calculate_freshness(&self, event: &EnrichedEvent) -> f32 {
        let now = Utc::now();
        let event_time = DateTime::from_timestamp(event.timestamp / 1000, 0).unwrap();
        let age = now.signed_duration_since(event_time);

        // Domain-specific half-life
        let half_life_minutes = match event.domain {
            Domain::Crypto => match event.source.as_str() {
                "solana_mempool" => 10.0,
                "eth_mempool" => 15.0,
                _ => 60.0,
            },
            Domain::Ai => match event.source.as_str() {
                "arxiv" => 720.0, // 12 hours
                "github_events" => 180.0,
                _ => 360.0,
            },
        };

        // Exponential decay
        let decay_factor = 0.5_f32.powf(age.num_minutes() as f32 / half_life_minutes);
        decay_factor.max(0.0).min(1.0)
    }

    async fn calculate_novelty(&self, event: &EnrichedEvent) -> f32 {
        // Check if we have embeddings
        if event.payload.embeddings.is_empty() {
            return 0.5; // Default if no embeddings
        }

        // Find k-nearest neighbors
        let similar = self.vector_client
            .search_similar(&event.payload.embeddings, 10)
            .await
            .unwrap_or_default();

        if similar.is_empty() {
            return 1.0; // Completely novel
        }

        // Calculate average distance to nearest neighbors
        let avg_similarity: f32 = similar.iter()
            .map(|s| s.similarity)
            .sum::<f32>() / similar.len() as f32;

        // Invert similarity to get novelty
        1.0 - avg_similarity
    }

    async fn calculate_impact(&self, event: &EnrichedEvent) -> f32 {
        match event.domain {
            Domain::Crypto => self.calculate_crypto_impact(event).await,
            Domain::Ai => self.calculate_ai_impact(event).await,
        }
    }

    async fn calculate_crypto_impact(&self, event: &EnrichedEvent) -> f32 {
        let metrics = &event.payload.extracted.metrics;

        match event.source.as_str() {
            "solana_mempool" => {
                // Token launch impact based on liquidity
                let liquidity = metrics.get("initial_liquidity_usd").unwrap_or(&0.0);
                let holder_count = metrics.get("holder_count").unwrap_or(&0.0);

                let liquidity_score = (*liquidity / 1_000_000.0).min(1.0); // $1M = max score
                let holder_score = (*holder_count / 1000.0).min(1.0); // 1k holders = max

                (liquidity_score * 0.7 + holder_score * 0.3).min(1.0)
            }
            _ => 0.5, // Default medium impact
        }
    }

    async fn calculate_ai_impact(&self, event: &EnrichedEvent) -> f32 {
        let metrics = &event.payload.extracted.metrics;

        match event.source.as_str() {
            "arxiv" => {
                // Paper impact based on potential citations
                let has_code = metrics.get("has_code").unwrap_or(&0.0) > 0.0;
                let reference_count = metrics.get("reference_count").unwrap_or(&0.0);
                let is_benchmark = metrics.get("is_benchmark").unwrap_or(&0.0) > 0.0;

                let mut score = 0.3; // Base score
                if has_code { score += 0.2; }
                if *reference_count > 50.0 { score += 0.2; }
                if is_benchmark { score += 0.3; }

                score.min(1.0)
            }
            "github_events" => {
                // Repository impact
                let stars = metrics.get("stars").unwrap_or(&0.0);
                let forks = metrics.get("forks").unwrap_or(&0.0);

                let star_score = (*stars / 10_000.0).min(1.0);
                let fork_score = (*forks / 1_000.0).min(1.0);

                (star_score * 0.6 + fork_score * 0.4).min(1.0)
            }
            _ => 0.5,
        }
    }

    async fn calculate_confidence(&self, event: &EnrichedEvent) -> f32 {
        let mut confidence = 0.5; // Base confidence

        // Source reputation
        let source_reputation = match event.source.as_str() {
            "solana_mempool" => 0.9, // Direct blockchain data
            "eth_mempool" => 0.9,
            "github_events" => 0.85,
            "arxiv" => 0.95,
            "twitter" => 0.6,
            _ => 0.7,
        };

        confidence = confidence.max(source_reputation);

        // Schema validity
        if event.payload.extracted.entities.is_empty() {
            confidence *= 0.8; // Penalize if no entities extracted
        }

        // Cross-verification bonus
        if event.payload.extracted.metrics.contains_key("cross_verified") {
            confidence = (confidence + 0.1).min(1.0);
        }

        confidence
    }

    async fn calculate_gap(&self, event: &EnrichedEvent) -> f32 {
        // Query knowledge graph for coverage
        let topic_coverage = self.knowledge_graph
            .get_topic_coverage(&event.topic_hints)
            .await
            .unwrap_or(0.5);

        // Invert coverage to get gap
        1.0 - topic_coverage
    }
}
```

**agent-runtime/evaluator/src/banding.rs**:

```rust
use prowzi_messages::{EVIEnvelope, Band};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use std::time::{Duration, Instant};

pub struct BandingQueue {
    tubes: HashMap<Band, UrgencyTube>,
    dedup_cache: Arc<RwLock<DedupCache>>,
}

pub struct UrgencyTube {
    band: Band,
    queue: Arc<RwLock<VecDeque<EVIEnvelope>>>,
    sender: mpsc::Sender<EVIEnvelope>,
    receiver: Arc<RwLock<mpsc::Receiver<EVIEnvelope>>>,
}

pub struct DedupCache {
    entries: HashMap<String, Instant>,
    ttl: Duration,
}

impl BandingQueue {
    pub fn new() -> Self {
        let mut tubes = HashMap::new();

        for band in [Band::Instant, Band::SameDay, Band::Weekly, Band::Archive] {
            let (tx, rx) = mpsc::channel(10000);
            tubes.insert(band.clone(), UrgencyTube {
                band: band.clone(),
                queue: Arc::new(RwLock::new(VecDeque::new())),
                sender: tx,
                receiver: Arc::new(RwLock::new(rx)),
            });
        }

        Self {
            tubes,
            dedup_cache: Arc::new(RwLock::new(DedupCache {
                entries: HashMap::new(),
                ttl: Duration::from_secs(60),
            })),
        }
    }

    pub async fn enqueue(&self, envelope: EVIEnvelope) {
        let band = &envelope.band;

        if let Some(tube) = self.tubes.get(band) {
            // Try to send without blocking
            if let Err(_) = tube.sender.try_send(envelope.clone()) {
                // Queue is full, apply backpressure
                tracing::warn!("Band {:?} queue full, applying backpressure", band);

                // Add to overflow queue
                let mut queue = tube.queue.write().await;
                queue.push_back(envelope);

                // Trim if too large
                if queue.len() > 50000 {
                    queue.pop_front();
                    metrics::QUEUE_OVERFLOW.inc();
                }
            }
        }
    }

    pub async fn subscribe(&self, band: Band) -> mpsc::Receiver<EVIEnvelope> {
        let (tx, rx) = mpsc::channel(1000);

        // Spawn background task to feed subscriber
        let tube = self.tubes.get(&band).unwrap().clone();
        tokio::spawn(async move {
            loop {
                // Try channel first
                let mut receiver = tube.receiver.write().await;
                if let Ok(envelope) = receiver.try_recv() {
                    let _ = tx.send(envelope).await;
                    continue;
                }

                // Then check overflow queue
                let mut queue = tube.queue.write().await;
                if let Some(envelope) = queue.pop_front() {
                    let _ = tx.send(envelope).await;
                    continue;
                }

                // Nothing available, wait a bit
                drop(queue);
                drop(receiver);
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        rx
    }

    pub async fn is_duplicate(&self, envelope: &EVIEnvelope) -> bool {
        let mut cache = self.dedup_cache.write().await;
        let now = Instant::now();

        // Clean expired entries
        cache.entries.retain(|_, time| now.duration_since(*time) < cache.ttl);

        // Check if exists
        if cache.entries.contains_key(&envelope.event_id) {
            return true;
        }

        // Add to cache
        cache.entries.insert(envelope.event_id.clone(), now);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_banding_queue_throughput() {
        let queue = BandingQueue::new();
        let start = Instant::now();

        // Generate test events
        let mut handles = vec![];

        for i in 0..10000 {
            let q = queue.clone();
            let handle = tokio::spawn(async move {
                let envelope = EVIEnvelope {
                    event_id: format!("test-{}", i),
                    scores: Default::default(),
                    total_evi: 0.8,
                    band: Band::Instant,
                    explanations: HashMap::new(),
                };

                q.enqueue(envelope).await;
            });
            handles.push(handle);
        }

        futures::future::join_all(handles).await;

        let duration = start.elapsed();
        assert!(duration < Duration::from_millis(150), "P99 latency exceeded");

        println!("Processed 10k events in {:?}", duration);
    }
}
