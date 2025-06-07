use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{dijkstra, strongly_connected_components};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct WalletNode {
    pub address: String,
    pub balance: f64,
    pub first_seen: i64,
    pub transaction_count: u64,
    pub labels: HashSet<String>,
}

#[derive(Debug, Clone)]
pub struct TransactionEdge {
    pub amount: f64,
    pub token: String,
    pub timestamp: i64,
    pub tx_hash: String,
}

pub struct WalletGraphAnalyzer {
    graph: DiGraph<WalletNode, TransactionEdge>,
    address_to_node: HashMap<String, NodeIndex>,
    patterns: PatternDetector,
}

impl WalletGraphAnalyzer {
    pub async fn analyze_wallet_cluster(
        &mut self,
        root_address: &str,
        depth: usize,
    ) -> WalletClusterAnalysis {
        // Build subgraph around wallet
        let subgraph = self.extract_subgraph(root_address, depth).await;

        // Detect patterns
        let wash_trading = self.detect_wash_trading(&subgraph);
        let sybil_clusters = self.detect_sybil_attacks(&subgraph);
        let mixing_patterns = self.detect_mixing_services(&subgraph);

        // Calculate metrics
        let centrality_scores = self.calculate_centrality(&subgraph);
        let clustering_coefficient = self.calculate_clustering(&subgraph);

        // Identify whale movements
        let whale_transfers = self.identify_whale_movements(&subgraph);

        // Risk assessment
        let risk_score = self.calculate_cluster_risk(
            &wash_trading,
            &sybil_clusters,
            &mixing_patterns,
            &whale_transfers,
        );

        WalletClusterAnalysis {
            root_wallet: root_address.to_string(),
            total_wallets: subgraph.node_count(),
            total_volume: self.calculate_total_volume(&subgraph),
            wash_trading_probability: wash_trading.probability,
            sybil_score: sybil_clusters.score,
            mixing_detected: !mixing_patterns.is_empty(),
            whale_movements: whale_transfers,
            risk_score,
            visualization_data: self.prepare_visualization(&subgraph),
        }
    }

    fn detect_wash_trading(&self, graph: &DiGraph<WalletNode, TransactionEdge>) -> WashTradingResult {
        let mut cycles = Vec::new();
        let sccs = strongly_connected_components(graph);

        for component in sccs {
            if component.len() < 3 {
                continue;
            }

            // Check for circular flows
            let mut total_flow = 0.0;
            let mut cycle_count = 0;

            for &node in &component {
                for edge in graph.edges(node) {
                    let target = edge.target();
                    if component.contains(&target) {
                        total_flow += edge.weight().amount;
                        cycle_count += 1;
                    }
                }
            }

            if cycle_count > component.len() * 2 {
                cycles.push(WashTradingCycle {
                    wallets: component.iter()
                        .map(|&n| graph[n].address.clone())
                        .collect(),
                    volume: total_flow,
                    transaction_count: cycle_count,
                });
            }
        }

        let probability = if cycles.is_empty() {
            0.0
        } else {
            (cycles.len() as f64 / graph.node_count() as f64).min(1.0)
        };

        WashTradingResult {
            probability,
            cycles,
        }
    }

    fn detect_sybil_attacks(&self, graph: &DiGraph<WalletNode, TransactionEdge>) -> SybilAnalysis {
        let mut sybil_groups = Vec::new();

        // Group wallets by creation time
        let mut time_clusters: HashMap<i64, Vec<NodeIndex>> = HashMap::new();
        for node in graph.node_indices() {
            let wallet = &graph[node];
            let time_bucket = wallet.first_seen / 3600000; // Hour buckets
            time_clusters.entry(time_bucket).or_default().push(node);
        }

        // Check for coordinated behavior
        for (_, nodes) in time_clusters.iter() {
            if nodes.len() < 5 {
                continue;
            }

            // Check transaction patterns
            let mut pattern_similarity = 0.0;
            for i in 0..nodes.len() {
                for j in i+1..nodes.len() {
                    let sim = self.calculate_behavior_similarity(
                        graph,
                        nodes[i],
                        nodes[j]
                    );
                    pattern_similarity += sim;
                }
            }

            pattern_similarity /= (nodes.len() * (nodes.len() - 1)) as f64 / 2.0;

            if pattern_similarity > 0.8 {
                sybil_groups.push(SybilGroup {
                    wallets: nodes.iter()
                        .map(|&n| graph[n].address.clone())
                        .collect(),
                    similarity_score: pattern_similarity,
                    creation_window: 3600, // 1 hour
                });
            }
        }

        SybilAnalysis {
            score: (sybil_groups.len() as f64 / graph.node_count() as f64).min(1.0),
            groups: sybil_groups,
        }
    }

    pub async fn track_whale_movements(&mut self) -> Vec<WhaleAlert> {
        let mut alerts = Vec::new();

        // Query recent large transactions
        let large_txs = self.query_large_transactions(1_000_000.0).await;

        for tx in large_txs {
            // Check if it's a known whale
            let whale_info = self.identify_whale(&tx.from_address).await;

            if let Some(whale) = whale_info {
                // Analyze transaction context
                let context = self.analyze_transaction_context(&tx).await;

                // Generate alert based on patterns
                if context.is_accumulating {
                    alerts.push(WhaleAlert {
                        whale_address: whale.address,
                        action: WhaleAction::Accumulating,
                        amount: tx.amount,
                        token: tx.token,
                        impact_estimate: self.estimate_market_impact(&tx),
                        confidence: context.confidence,
                    });
                } else if context.is_distributing {
                    alerts.push(WhaleAlert {
                        whale_address: whale.address,
                        action: WhaleAction::Distributing,
                        amount: tx.amount,
                        token: tx.token,
                        impact_estimate: self.estimate_market_impact(&tx),
                        confidence: context.confidence,
                    });
                }
            }
        }

        alerts
    }
}
